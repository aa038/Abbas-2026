"""
Observe the mock planet catalog 
-------------------------------------------
This script observes the full planet catalog under a 3-month uniform cadence for demographics studies, 
similar to the other demographics model, but then applies a simple noise model on top of the working angles/contrast constraints.
The script assumes a set of IWA, OWA and contrast floor, all of which are tunable by the user.

Input:
    Data/Planet Generation/SAG13 Planet Catalog.csv      # The full planet catalog from Fig 1
    Data/Planet Generation/HWO Stars.csv

Output:
    1. Observing Log.csv                                 # The observing log for the chosen telescope configuration
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

# solve_orbit is a local library that will need to be installed
# Installation intructions can be found in REQUIREMENTS.md in the root directory
from solve_orbit import solve_orbit

# ------------------------ Constants ------------------------- #
# Telescope Parameters
CONTRAST_FLOOR = 1e-10         # Coronagraph contrast floor
PEAK_WAVELENGTH = 0.6e-6       # In metres. From the Habex Final Report (Gaudi et al. 2020)
THROUGHPUT = 0.27              # From the Habex Final Report (Gaudi et al. 2020)
IWA = 0.06                     # Inner working angle in arcsec
OWA = 1                        # Outer working angle in arcsec
TEL_APERTURE = 6               # Telescope primary mirror aperture in m (From the Habex Final Report)

# Area of the primary mirror
A = np.pi * (TEL_APERTURE/2) ** 2

# Mission Parameters
mission_clock = 2035           # Mission start time (in years)
mission_end = 2060             # Mission end time (in years)
maxObs = 8                     # Max number of times a star will be revisited for observations 
timeStep = 3/12                # Spacing between observations in fractional years (This is scaled to the stellar HZ; See Eqn 5 in Abbas et al. 2026)

# Physical Constants
SOLARLUM = 3.83e26             # Solar luminosity in J/s
EARTHRADII = 6.371e6           # Radius of the Earth in metres (Planet radii in the catalog are in R_E)
AU_TO_M = 1.5e11               # 1 AU in metres (Planet SMAs are in AU)
PC_TO_M = 3.086e16             # 1 pc in metres (Distances to stars are in pc)

h = 6.626e-34                  # Planck's constant (J·s)
c = 3e8                        # Speed of light (m/s)
# ------------------------------------------------------------- #

# ----------------- EXOSIMS-lite noise model ------------------ #
# Approximate Johnson-Cousins Vega zero-points.
# Values are in Jy
FILTERS = {
    "U": {"lambda_m": 0.365e-6, "Fnu0_Jy": 1810.0},
    "B": {"lambda_m": 0.445e-6, "Fnu0_Jy": 4260.0},
    "V": {"lambda_m": 0.551e-6, "Fnu0_Jy": 3640.0},
    "R": {"lambda_m": 0.658e-6, "Fnu0_Jy": 3080.0},
    "I": {"lambda_m": 0.806e-6, "Fnu0_Jy": 2550.0},
}


USE_ETC_FILTER = True

# Detection policy
SNR_DETECTION = 7.0             # Common broadband detection threshold; can also test 10
T_VISIT_HR = 3.0               # Per-epoch exposure-time budget
T_VISIT_S = T_VISIT_HR * 3600.0

# Bandpass
BANDPASS_FRAC = 0.20            # fractional bandwidth, e.g. 20%
BANDPASS_NM = BANDPASS_FRAC * (PEAK_WAVELENGTH * 1e9)

# Simple photon-flux normalizations
# Approximate V=0 flux density near 550-600 nm
V0_FLUX_W_M2_NM = 3.6e-11       # W / m^2 / nm

# Approximate solar spectral irradiance near 0.6 micron at 1 AU
# Used because the catalog has L_sol and d_pc, but not necessarily V magnitudes.
SOLAR_SPEC_IRRAD_W_M2_NM = 1.8  # W / m^2 / nm at 1 AU

# Zodiacal / exozodiacal surface brightness
MU_ZODI = 23.0                  # mag / arcsec^2
MU_EXOZODI_1ZODI = 22.0         # mag / arcsec^2 for one exozodi, simple approximation
N_EXOZODI = 3.0                 # fixed value for diagnostic

# Detector terms
DARK_CURRENT = 1e-4             # e- / pix / s
READ_NOISE = 0.0                # e- / pix / read; set nonzero for sensitivity test
CIC = 1e-3                      # e- / pix / read
T_FRAME_S = 1000.0              # effective frame time
N_PIX = 6.0                     # photometric aperture in pixels, simple diagnostic

# Speckle systematic term
SPECKLE_STABILITY = 0.03         
# Also test SPECKLE_STABILITY = 0.01 or 0.05 as a conservative systematic floor

# Optional post-processing gain
POSTPROCESSING_GAIN = 1.0       # set >1 to reduce residual stellar leakage
# -------------------------------------------------------------- #


def HZ(L):
    """
    Calculate inner and outer edges of the habitable zone.

    Parameters:
    L (float or np.array): Stellar luminosity in solar units.

    Returns:
    HZ_inner (float or np.array): Inner edge of the HZ (in AU).
    HZ_outer (float or np.array): Outer edge of the HZ (in AU).
    """
    HZ_inner = np.sqrt(L / 1.78)
    HZ_outer = np.sqrt(L / 0.32)
    return HZ_inner, HZ_outer


def RankStars(starsDF, weights = [-5, -10, 5, 10]):
    """
    Rank stars based on distance, brightness, and spectral type to choose for observations.
    This is a simple formulation, since target prioritization was beyond the scope of this work.
    Contact me if you would like to collaborate on implementing a more detailed algorithm.
    
    Parameters:
    starsDF (pd.DataFrame): DataFrame containing star and planet properties.
    
    Returns:
    pd.DataFrame: Ranked DataFrame with a 'Score' column.
    """

    # Weights
    wDist = weights[0]        # Weight for the distance
    wLum = weights[1]         # Weight for luminosity
    wSpecType = weights[2]    # Weight for spectral type
    wExpTime = weights[3]     # Weight for exposure time (prefer shorter exposure times)

    # Spectral type factor mapping
    # We prefer F and G stars since the HZ will be larger, farther out and we have the potential for more SNR
    spectral_type_map = {'F': 3, 'G': 3, 'K': 2, 'M': 1} 

    # Add spectral type factor
    spectralTypeFactor = starsDF['Spec'].str[0].map(spectral_type_map).fillna(0)

    # Normalize the exposure times
    normExpTime = (starsDF['tExp(s)'] - starsDF['tExp(s)'].min()) / (starsDF['tExp(s)'].max() - starsDF['tExp(s)'].min())

    # Compute scores
    starsDF['Score'] = (
        wDist / starsDF['d_pc'] +  # Closer stars rank higher
        wLum * starsDF['L_sol'] +  # Brighter stars rank higher
        wSpecType * spectralTypeFactor +  # Adjust for spectral type
        wExpTime * normExpTime # Adjust for exposure time
    )

    # Sort stars by score
    ranked_stars = starsDF.sort_values(by='Score', ascending=False).reset_index(drop=True)
    return ranked_stars

def IntegrationTime(L, d, A, throughput = THROUGHPUT, contrast = CONTRAST_FLOOR, SNR = 100, wavelength = PEAK_WAVELENGTH):

    # Convert distance from pc to metres
    d = d * PC_TO_M

    # Flux from the star
    starFlux = (L * SOLARLUM) / (4 * np.pi * d**2)

    # Convert to photon flux
    photon_energy = h * c / wavelength  # Energy of a photon (J)
    starPhotonFlux = starFlux / photon_energy  # Photon flux (photons/m^2/s)

    # Using the max contrast that can be achieved, we can get the minimum flux for the planet we can detect
    planetPhotonFlux = starPhotonFlux * contrast

    # Compute the exposure time, assuming the signal limited case
    tExp = (SNR ** 2) * (starPhotonFlux*contrast + planetPhotonFlux) / (planetPhotonFlux**2 * A * throughput)

    return tExp

def contrastCheck(DetParams, albedo, Rp, coronagraphContrast = CONTRAST_FLOOR):

    # Compute the 3D distance 
    r3D = DetParams[2] * AU_TO_M

    # Contrast
    contrast = albedo * (Rp/r3D)**2 * DetParams[3]

    if contrast > coronagraphContrast:
        return True
    
    return False

def sepCheck(DetParams, IWA = IWA, OWA = OWA):

    # Check if the planet is within the working angle
    if IWA < DetParams[0] < OWA:
        return True
    
    return False

def planet_contrast(DetParams, albedo, Rp_m):
    """
    Reflected-light planet/star contrast.
    """

    # 3D planet-star separation
    r3D_m      = DetParams[2] * AU_TO_M
    # Phase function
    phase_func = DetParams[3]

    # Contrast computed using the 3D distance, phase function and albedo
    contrast = albedo * (Rp_m / r3D_m)**2 * phase_func

    return contrast

def diffraction_aperture(wavelength_m=PEAK_WAVELENGTH, diameter_m=TEL_APERTURE, aperture_radius_lamD=0.7):
    """
    Approximate the angular area on the sky of the photometric aperture,
    for a diffraction-limited planet image.

    We assume a Stark 2014-like simple photometric aperture around the planet PSF = 0.7 as default
    """

    # Angular size of one diffraction element in arcsec
    lam_over_D_arcsec = (wavelength_m / diameter_m) * 206265.0

    # Use a circular photometric aperture to measure the planet's light
    area_arcsec2 = np.pi * (aperture_radius_lamD * lam_over_D_arcsec)**2

    return area_arcsec2

def separation_dependent_throughput(sep_arcsec, iwa=IWA, max_throughput=THROUGHPUT, rolloff_power=4):
    """
    Approximate coronagraphic core throughput as a smooth function of separation.

    The adopted IWA is treated as the half-throughput point:
        T(sep = IWA) = 0.5 * max_throughput

    This follows the common coronagraph convention that IWA is the separation
    where off-axis throughput reaches half of its asymptotic value.

    Parameters
    ----------
    sep_arcsec : float or array
        Planet-star angular separation in arcsec.
    iwa : float
        Inner working angle in arcsec, interpreted as the half-throughput point.
    max_throughput : float
        Asymptotic planet-core throughput at large separation.
    rolloff_power : float
        Controls how sharply throughput rises near the IWA.
        Larger values give a sharper transition.

    Returns
    -------
    throughput : float or array
        Separation-dependent planet-core throughput.
    """

    # Define separation in units of IWA
    x = sep_arcsec / iwa

    # Using x, construct an analytical expression for sep dependent throughput
    throughput = x**rolloff_power / (1.0 + x**rolloff_power)

    return throughput

def separation_dependent_raw_contrast(sep_arcsec, iwa=IWA, contrast_floor=CONTRAST_FLOOR, 
    contrast_at_iwa_factor=10.0, rolloff_width_iwa=0.5, rolloff_power=2.0):
    """
    Approximate a separation-dependent raw coronagraph contrast curve.

    The contrast approaches raw_contrast_floor at large separations, but is
    degraded near the IWA by a smooth multiplicative excess.

    This is an analytic surrogate for the spatially dependent raw contrast
    curves used in higher-fidelity yield calculations. It should be replaced
    by a real coronagraph contrast-vs-separation curve when available.

    Parameters
    ----------
    sep_arcsec : float or array
        Planet-star angular separation in arcsec.

    iwa : float
        Inner working angle in arcsec.

    contrast_floor : float
        Asymptotic raw contrast at large separations.

    contrast_at_iwa_factor : float
        Multiplicative degradation at the IWA. For example, 10 means the raw
        contrast at the IWA is 10 times worse than the large-separation floor.

    rolloff_width_iwa : float
        Width of the contrast roll-off in units of IWA. Smaller values make the
        contrast approach the floor more quickly.

    rolloff_power : float
        Shape parameter for the roll-off.

    Returns
    -------
    raw_contrast : float or array
        Separation-dependent raw contrast.
    """
    # Define separation in units of IWA
    x = sep_arcsec / iwa

    # Excess contrast degradation near the IWA.
    # At x = 1, contrast = contrast_at_iwa_factor * raw_contrast_floor.
    # At large x, contrast = raw_contrast_floor.
    dx = x - 1.0
    excess = (contrast_at_iwa_factor - 1.0) * np.exp(-(dx / rolloff_width_iwa)**rolloff_power)

    raw_contrast = contrast_floor * (1.0 + excess)

    return raw_contrast

def count_rate_from_mag(mag, band="V", bandpass_frac=BANDPASS_FRAC, area_m2=A, throughput=THROUGHPUT):
    """
    Convert an apparent magnitude in a given band into an approximate
    photon/electron count rate.

    Assumes Vega-system Johnson-Cousins magnitudes.
    """

    # Load the wavelength and flux details
    filt     = FILTERS[band]
    lambda_m = filt["lambda_m"]
    Fnu0_Jy  = filt["Fnu0_Jy"]

    # Convert the magnitude to flux density in W/m^2/Hz
    Fnu = Fnu0_Jy * 1e-26 * 10.0**(-0.4 * mag)

    # Bandwidth in Hz, using dnu = c/lambda^2 dlambda
    dlambda_m = bandpass_frac * lambda_m
    dnu = c * dlambda_m / lambda_m**2

    photon_energy = h * c / lambda_m

    return Fnu * dnu * area_m2 * throughput / photon_energy

def surface_brightness_count_rate(mu_mag_arcsec2, omega_arcsec2, band="V", bandpass_frac=BANDPASS_FRAC, area_m2=A, throughput=THROUGHPUT):
    """
    Convert surface brightness in mag/arcsec^2 into count rate inside
    a photometric aperture of area omega_arcsec2.
    """
    filt = FILTERS[band]
    lambda_m = filt["lambda_m"]
    Fnu0_Jy = filt["Fnu0_Jy"]

    # Convert the magnitude to flux density in W/m^2/Hz
    Fnu = Fnu0_Jy * 1e-26 * 10.0**(-0.4 * mu_mag_arcsec2)

    # Bandwidth in the V-band
    dlambda_m = bandpass_frac * lambda_m
    dnu = c * dlambda_m / lambda_m**2

    photon_energy = h * c / lambda_m

    return Fnu * dnu * area_m2 * throughput * omega_arcsec2 / photon_energy


def count_rates(V_mag, planet_star_contrast, sep_arcsec):
    """
    EXOSIMS-inspired count-rate model.

    Returns
    -------
    Cp : float
        Planet signal count rate [e-/s].
    Cb : float
        Stochastic background count rate [e-/s].
    Csp : float
        Residual speckle systematic count rate [e-/s].
    """

    # Compute the separation dependent throughput and contrast
    tau          = separation_dependent_throughput(sep_arcsec)
    raw_contrast = separation_dependent_raw_contrast(sep_arcsec)

    # Compute the signal in the V-band
    C_star = count_rate_from_mag(V_mag)

    # Planet signal
    Cp = C_star * planet_star_contrast * tau

    # Residual stellar leakage / speckle halo
    C_speckle = C_star * raw_contrast / POSTPROCESSING_GAIN

    # Local zodi and exozodi effects
    # Angular area of the photometric aperture
    omega_ap  = diffraction_aperture()
    # Counts from the zodi and exo-zodi in this aperture
    C_zodi    = surface_brightness_count_rate(MU_ZODI, omega_ap)
    C_exozodi = N_EXOZODI * surface_brightness_count_rate(MU_EXOZODI_1ZODI, omega_ap)

    # Detector terms
    C_dark = N_PIX * DARK_CURRENT
    C_read = N_PIX * READ_NOISE**2 / T_FRAME_S
    C_cic  = N_PIX * CIC / T_FRAME_S

    Cb = C_speckle + C_zodi + C_exozodi + C_dark + C_read + C_cic

    # Residual systematic speckle floor
    Csp = SPECKLE_STABILITY * C_speckle

    return Cp, Cb, Csp


def integration_time(V_mag, planet_star_contrast, sep_arcsec, snr=SNR_DETECTION):
    """
    Required exposure time using an EXOSIMS-like Cp, Cb, Csp model.

    SNR = Cp*t / sqrt((Cp + Cb)*t + (Csp*t)^2)

    Solving for t gives:
    t = SNR^2 * (Cp + Cb) / (Cp^2 - SNR^2*Csp^2)
    """
    Cp, Cb, Csp = count_rates(V_mag, planet_star_contrast, sep_arcsec)

    denom = Cp**2 - (snr * Csp)**2

    t_req_s = snr**2 * (Cp + Cb) / denom

    return t_req_s, Cp, Cb, Csp


def detectionCheck(DetParams, albedo, Rp_m, V_mag, t_visit_s=T_VISIT_S):
    """
    Full detection check:
    1. Within working angles
    2. Above nominal hard contrast floor
    3. Reaches S/N threshold within per-visit exposure time
    """
    # IWA/OWA Check
    sep_ok = sepCheck(DetParams)

    # Contrast floor check
    Cp_contrast = planet_contrast(DetParams, albedo, Rp_m)
    contrast_ok = Cp_contrast > CONTRAST_FLOOR

    if not (sep_ok and contrast_ok):
        return False, Cp_contrast, np.inf, 0.0, np.inf, np.inf

    # Run the integration time required for SNR = 7
    t_req_s, Cp, Cb, Csp = integration_time(V_mag, planet_star_contrast=Cp_contrast, sep_arcsec=DetParams[0])

    detected = t_req_s <= t_visit_s

    return detected, Cp_contrast, t_req_s, Cp, Cb, Csp


def calculate_characteristic_hz_period(L_sol, M_sol):
    """
    Period for a planet in the HZ scaled to the stellar luminosity

    E.g., For an Earth-like planet around a Sun-like star, 3 months = 1/4 of the period.
    For an F star, the HZ is farther out => a period >3 months corresponds to 1/4 of the period.
    For an M star, the HZ is closer => a period <3 months corresponds to 1/4 of the period.

    This scaling is done to ensure planets around different stars are observed at similar orbital positions.

    See Eqn 5 in Abbas et al. 2026 and the associated text for more detail
    """
    HZ_in, HZ_out = HZ(L_sol)                # HZ boundaries for the current star
    a_hz = np.sqrt(HZ_in * HZ_out)           # Characteristic SMA in AU

    return np.sqrt((a_hz**3) / M_sol)        # Characteristic HZ period in years


# ---------------------------------- I/O ------------------------------------ #
curr_dir   = Path(__file__).resolve().parent
parent_dir = curr_dir.parent
data_dir   = parent_dir / "Data" / "Planet Generation"

# Load the list of stars with planets around them
planets_df = pd.read_csv(data_dir / "SAG13 Planet Catalog.csv")
planets_df = planets_df[planets_df['P_yr'] < 10]

# Load the star details
stars      = pd.read_csv(data_dir / "HWO Stars.csv")
# --------------------------------------------------------------------------- #

# ---------------- Setting up the simulated observations -------------------- #
# For simulated observations, we need the following:
#   1. List of stars to observe
#   2. Rank them 
#   3. Create variables to keep track of
#       - How many times a star has been observed
#       - When it was observed last (since we observe after a minimum of timeStep months)
#       - How many times each planet around the star was observed 

# ---- 1. Star catalog (subset of the planet catalog) ---- #
stars_df = planets_df.drop_duplicates("HDName")[["HDName", "Spec", "d_pc", "L_sol", "M_sol"]].copy()

# Keep only the photometry columns we need from the HWO star table
stars_phot = stars[["HDName", "V"]].copy()

# Merge V magnitude onto the stars used in this simulation
stars_df = stars_df.merge(stars_phot, on="HDName", how="left", validate="many_to_one")

# Sanity check
missing_v = stars_df[stars_df["V"].isna()]
if len(missing_v) > 0:
    print("Stars missing V magnitudes after merge:")
    print(missing_v[["HDName", "Spec", "d_pc", "L_sol", "M_sol"]])
    raise ValueError(f"{len(missing_v)} stars are missing V magnitudes.")

# ---- 2. Compute the exposure time required for each star, and rank the stars ---- #
stars_df['tExp(s)'] = IntegrationTime(stars_df['L_sol'], stars_df['d_pc'], A)
stars_df            = RankStars(stars_df)

# Compute the characteristic HZ period for each star (See Abbas et al. Eqn 5)
stars_df['hz_period'] = calculate_characteristic_hz_period(stars_df['L_sol'], stars_df['M_sol'])

# Empty array to keep track of how many times the planet has been detected
planets_df["NDet"] = np.zeros(len(planets_df))

# ---- 3. Timekeeping variables ---- #
# Empty arrays to keep track of:
# - How many times a star has been observed
# - When the stars was last observed
stars_df["NObs"]    = np.zeros(len(stars_df))
stars_df["LastObs"] = np.ones(len(stars_df)) * (-np.inf)

# Empty dataframe to hold the observing log
ObsLog = pd.DataFrame(columns = ['PlanetID', 'StarName', 'SMA_AU', 'ecc', 'Rp_REarth', 'Mp_MEarth', 'd_pc', 'M_sol', 'L_sol', 'Sep', 'PA', 
                                 '3dDist', 'PhaseFunc', 'PlanetContrast', 'tReq_s', 'tReq_hr', 'Cp_e_s', 'Cb_e_s', 'Csp_e_s', 'DetStatus_Ideal',
                                 'DetStatus_ETC','DetStatus', 'NDet', 'NObs', 'LastObs'])

# Timer
start_time  = time.time()
obs_counter = 0

# Maximum number of possible observations
# If there are 100 stars set to be observed 8 times each, max_possible_obs = 100 * 8 = 800
# Useful for some conditional statements in the observing loop to ensure we do not quit the loop too early
max_possible_obs = len(stars_df) * maxObs  

while mission_clock < mission_end:

    # Creating our target list. We want stars that:
    # 1. Have not been observed more than NObs times
    # 2. Have been observed more than one timeStep (the minimum wait time between obs) ago
    eligible_stars = stars_df[
        (stars_df['NObs'] < maxObs) &
        (mission_clock >= stars_df['LastObs'] + stars_df['hz_period'] * timeStep)
    ]

    # If the target list is empty, that could be due to two reasons
    # 1. All the stars have been observed the required number of times. In that case, we're done!
    # 2. It has not been long enough since we last observed all the stars. In that case, move time forward by a day and try again
    if eligible_stars.empty:
        if (stars_df['NObs'] == maxObs).all():
            break
        mission_clock += 1/365  # Advance by 1 day
        continue

    # Pick highest-ranked eligible star
    star      = eligible_stars.sort_values("Score", ascending=False).iloc[0]
    star_name = star["HDName"]
    star_mass = star["M_sol"]
    star_lum  = star["L_sol"]
    star_dist = star["d_pc"]
    V_mag     = star["V"]
    tExp      = star["tExp(s)"]

    # All planets around this star
    star_planets = planets_df[planets_df['HDName'] == star_name]

    # Update star-level bookkeeping
    stars_df.loc[stars_df["HDName"] == star_name, "LastObs"] = mission_clock
    stars_df.loc[stars_df["HDName"] == star_name, "NObs"]   += 1

    # --------------  Progress Bar  ---------------------- #
    obs_counter += 1

    if obs_counter % len(stars_df) == 0:
        elapsed = time.time() - start_time
        avg_time = elapsed / obs_counter
        remaining = (max_possible_obs - obs_counter) * avg_time
        print(f"[{mission_clock:.2f}] {obs_counter} obs done | Est. {remaining:.1f} sec remaining")
    # ---------------------------------------------------- #

    # Loop through all the planets around this star and update the Observing Log
    for _, planet in star_planets.iterrows():
        Obs_row = []

        # ObsLog Cols 1. & 2. 
        # Planet ID and Star ID
        Obs_row.append(planet['PlanetID']) 
        Obs_row.append(star_name)

        # Planet orbital and physical parameters
        sma    = planet['sma_AU']
        ecc    = planet['ecc']
        inc    = np.degrees(planet['inc_rad'])
        aop    = np.degrees(planet['aop_rad'])
        pan    = np.degrees(planet['pan_rad'])
        epp    = planet['epp_yr']
        P      = planet['P_yr']
        d      = planet['d_pc']
        albedo = planet['albedo']
        Rp     = planet['Rp_Rearth'] * EARTHRADII
        Mp     = planet['Mp_Mearth']

        # Add core identifiers and physical parameters
        # ObsLog Cols 3-9. 
        # SMA (in AU), Ecc, Rp (in R_E), M_p (in M_E), distance to the planet (in pc), Stellar mass (in M_solar), Stellar luminosity (in L_solar)
        Obs_row.extend([sma, ecc, planet['Rp_Rearth'], Mp, d, star_mass, star_lum])

        # Compute sep, PA, 3D dist and phase function
        # DetParams = [2D Sep (in arcsec), Position Angle (in deg), 3D star-planet sep (in AU), Orbital Phase Function]
        # solve_orbit is a local library that will have to be installed. 
        # See detailed in REQUIREMENTS.md in the root directory
        DetParams = solve_orbit(sma, ecc, inc, aop, pan, epp, P, mission_clock, d)

        # ObsLog Cols 10-13. 
        # 2D Sep (in arcsec), PA (in degrees), 3D planet-star sep (in AU), Phase Func
        Obs_row.extend(DetParams)

        # ---------------- Detection model ---------------- #
        # Idealized detection: working angles + hard contrast floor
        check_sep      = sepCheck(DetParams)
        pl_contrast    = planet_contrast(DetParams, albedo, Rp)
        check_contrast = pl_contrast > CONTRAST_FLOOR
        det_ideal      = int(check_sep and check_contrast)

        # EXOSIMS-lite exposure-time filter
        det_etc, pl_contrast, t_req_s, Cp, Cb, Csp = detectionCheck(DetParams, albedo, Rp, V_mag)
        det_etc = int(det_etc)

        # Choose which detection status drives NDet and downstream orbit fitting
        if USE_ETC_FILTER:
            det_status = det_etc
        else:
            det_status = det_ideal

        Obs_row.extend([
            pl_contrast,
            t_req_s,
            t_req_s / 3600.0 if np.isfinite(t_req_s) else np.inf,
            Cp,
            Cb,
            Csp,
            det_ideal,
            det_etc,
            det_status,
        ])

        if det_status == 1:
            planets_df.loc[planets_df['PlanetID'] == planet['PlanetID'], 'NDet'] += 1
        # ------------------------------------------------- #

        # Add tracking metadata
        Obs_row.append(int(planets_df.loc[planets_df["PlanetID"] == planet["PlanetID"], "NDet"].iloc[0]))
        Obs_row.append(int(stars_df.loc[stars_df["HDName"] == star_name, "NObs"].iloc[0]))
        Obs_row.append(mission_clock)

        # Save observation
        ObsLog.loc[len(ObsLog)] = Obs_row

    # Advance clock by half a day 
    # We assume the total program time with any overheads takes 0.5 days
    mission_clock += 0.5/365

round_cols = ["SMA_AU", "ecc", "Rp_REarth", "Mp_MEarth", "d_pc",
              "M_sol", "L_sol", "Sep", "PA", "3dDist", "PhaseFunc", "LastObs"]

ObsLog[round_cols] = ObsLog[round_cols].round(3)

ObsLog.to_csv(curr_dir / "1. Observing Log - 3 hr.csv", index=False)


    


