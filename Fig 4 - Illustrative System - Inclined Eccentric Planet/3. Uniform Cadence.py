"""
Illustrative System - Uniform Cadence (Part 3 of 5)
---------------------------------------------------
Runs a uniform revisit cadence on the demo short-period, near face-on planet
(Fig. 4a). Loops through mission time, selects the next eligible target, calls
`solve_orbit(...)` to get on-sky separation/phase, applies simple detectability
checks (IWA/OWA + contrast), and logs each visit.

Inputs:
    2. Planet Catalog w Mass.csv   # From Part 2 (Same directory)

Outputs:
    3. Observing Log.csv           # Visit-by-visit log with detections

Notes:
- Telescope/mirror area and basic constants are defined in-file.
- This script is the "uniform cadence" baseline; the adaptive variant
  lives in Part 4 and reads/writes analogous files.
"""

# ------------------------ Scientific Assumptions --------------------------- #
# - ONE planet generated around ONE star
# - Planet observed uniformly every 3 months (scaled by the HZ period; See Sec 3.3 and Eqn 5 in Abbas et al. 2026)
# - Telescope parameters shown below
# - Planet is considered detected if:
# -     (a) 2D star-planet sep is within the working angles
# -     (b) star-planet contrast is above the defined contrast floor
# - The mission modelling assumes idealised conditions, with minimal noise modelling. The paper focuses mostly on observing cadence and scheduling 
# --------------------------------------------------------------------------- #

import numpy as np
import pandas as pd
from pathlib import Path

# orbituary is a local library that will need to be installed
# Installation intructions can be found in REQUIREMENTS.md in the root directory
from orbituary.solve_orbit import solve_orbit

SEED = 42
rng  = np.random.default_rng(SEED)

# ----------------- Constants ------------------ #
# Telescope Parameters
PEAK_WAVELENGTH = 0.6e-6       # In metres. From the Habex Final Report (Gaudi et al. 2020)
THROUGHPUT      = 0.27         # From the Habex Final Report (Gaudi et al. 2020)
CONTRAST_FLOOR  = 2.5e-11      # Deeper of the two contrast floors used in the ExEP Mission Star List (Mamajek & Stapelfeldt 2024)
IWA             = 0.06         # Inner working angle in arcsec
OWA             = 1            # Outer working angle in arcsec
TEL_APERTURE    = 6            # Telescope primary mirror aperture in m (From the Habex Final Report)

# Area of the primary mirror
A = np.pi * (TEL_APERTURE/2) ** 2


# Mission Parameters
mission_clock = 2035           # Mission start time (in years)
mission_end   = 2045           # Mission end time (in years)
maxObs        = 8              # Max number of times a star will be revisited for observations 
timeStep      = 3/12           # Spacing between observations in fractional years (This is scaled to the stellar HZ; See Eqn 5 in Abbas et al. 2026)

# Physical Constants
SOLARLUM   = 3.83e26           # Solar luminosity in J/s
EARTHRADII = 6.371e6           # Radius of the Earth in metres (Planet radii in the catalog are in R_E)
AU_TO_M    = 1.5e11            # 1 AU in metres (Planet SMAs are in AU)
PC_TO_M    = 3.086e16          # 1 pc in metres (Distances to stars are in pc)

h = 6.626e-34                  # Planck's constant (J·s)
c = 3e8                        # Speed of light (m/s)

SEP_UNCERTAINTY = 0.003
# ---------------------------------------------- #


def HZ(L):
    """
    Return inner/outer HZ edges (AU) for stellar luminosity L (in L_sun).
    """

    hz_inner = np.sqrt(L / 1.78)
    hz_outer = np.sqrt(L / 0.32)

    return hz_inner, hz_outer


def RankStars(starsDF, weights = [1, 2, 1, 2]):
    """
    Rank stars based on distance, brightness, and spectral type to choose for observations.
    This is a simple formulation, since nuanced target prioritization is beyond the scope of this work.
    
    Parameters:
    starsDF (pd.DataFrame): DataFrame containing star and planet properties.
    
    Returns:
    pd.DataFrame: Ranked DataFrame with a 'Score' column.
    """

    wDist, wLum, wSpecType, wExpTime = weights

    # We prefer F and G stars since the HZ will be larger, farther out and we have the potential for more SNR
    spectral_type_map  = {"F": 3, "G": 3, "K": 2, "M": 1}
    spectralTypeFactor = starsDF["Spec"].str[0].map(spectral_type_map).fillna(0) / 3

    def normalize(values):
        values = values.astype(float)
        span   = values.max() - values.min()

        if span == 0:
            return pd.Series(0.0, index=values.index)

        return (values - values.min()) / span

    distance_score   = 1.0 - normalize(starsDF["d_pc"])          # Closer stars rank higher
    luminosity_score = normalize(starsDF["L_sol"])               # Brighter stars rank higher

    starsDF["Score"] = wDist * distance_score + wLum * luminosity_score + wSpecType * spectralTypeFactor 

    # Sort stars by score
    ranked_stars = starsDF.sort_values(by='Score', ascending=False).reset_index(drop=True)
    return ranked_stars


def contrastCheck(DetParams, albedo, Rp, coronagraphContrast = CONTRAST_FLOOR):
    """
    Check if the observed planet contrast is above the coronagraph contrast floor.

    This is check 1/2 for planet detectability
    """

    # Compute the 3D distance between the planet and star in metres
    r3D = DetParams[2] * AU_TO_M

    # Compute the planet - star contrast
    contrast = albedo * (Rp/r3D)**2 * DetParams[3]

    if contrast > coronagraphContrast:
        return True
    
    return False

def sepCheck(DetParams, IWA = IWA, OWA = OWA):
    """
    Check is the 2D planet - star separation is within telescope working angles

    This is check 2/2 for planet detectability
    """

    # Check if the planet is within the working angle
    if IWA < DetParams[0] < OWA:
        return True
    
    return False

def characteristic_hz_period_years(L_sol, M_sol):
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
curr_dir = Path(__file__).resolve().parent

# Load the simulated planet catalog with mass (Result from Part 2)
planets_df = pd.read_csv(curr_dir / "2. Planet Catalog w Mass.csv")
# --------------------------------------------------------------------------- #

# ---------------- Setting up the simulated observations -------------------- #
# For simulated observations, we need the following:
#   1. List of stars to observe
#   2. Rank them 
#   3. Create variables to keep track of
#       - How many times a star has been observed
#       - When it was observed last (since we observe after a minimum of timeStep months)
#       - How many times the planet was detected 
#         (This gets more complex for multi-planet systems, but this example is a single planet system by design.
#          Multi-planet systems are handled in Scripts related to Figs. 5 and 6)

# ---- 1. Star catalog (subset of the planet catalog) ---- #
stars_df = planets_df.drop_duplicates("HDName")[["HDName", "Spec", "d_pc", "L_sol", "M_sol"]].copy()

# ---- 2. Rank the stars for scheduling ---- #
stars_df = RankStars(stars_df)

# Compute the characteristic HZ period for each star (See Abbas et al. Eqn 5)
stars_df['hz_period'] = characteristic_hz_period_years(stars_df['L_sol'], stars_df['M_sol'])

# ---- 3. Timekeeping variables ---- #
# Empty arrays to keep track of:
# - How many times a star has been observed
# - How many times a planet has been detected
# - When the stars was last observed
stars_df["NObs"]    = np.zeros(len(stars_df))
planets_df["NDet"]  = np.zeros(len(planets_df))
stars_df["LastObs"] = np.ones(len(stars_df)) * (-np.inf) # -Inf implies that at the start of the simulation, all stars are ready to be observed right away

# Empty dataframe to hold the observing log
ObsLog = pd.DataFrame(columns = ['PlanetID', 'StarName', 'SMA_AU', 'ecc', 'Rp_REarth', 'Mp_MEarth', 'd_pc', 'M_sol', 'L_sol', 'Sep', 'PA', '3dDist', 'PhaseFunc', 'DetStatus', 'NDet', 'NObs', 'LastObs'])

# Maximum number of possible observations
# If there are 100 stars set to be observed 8 times each, max_possible_obs = 100 * 8 = 800
# Useful for some conditional statements in the observing loop to ensure we do not quit the loop too early
max_possible_obs = len(stars_df) * maxObs  
# --------------------------------------------------------------------------- #

# ------------------------- Observing Simulation ---------------------------- #

# mission_clock is the current calendar time
# mission_end is when the observations are slated to end
while mission_clock < mission_end:

    # Creating our target list. We want stars that:
    # 1. Have not been observed more than NObs times
    # 2. Have been observed more than one timeStep (the minimum wait time between obs scaled by the HZ period) ago
    eligible_stars = stars_df[(stars_df['NObs'] < maxObs) & (mission_clock >= stars_df['LastObs'] + stars_df['hz_period'] * timeStep)]

    # If the target list is empty, that could be due to two reasons
    # 1. All the stars have been observed the required number of times. In that case, we're done!
    # 2. It has not been long enough since we last observed all the stars. In that case, move time forward by a day and try again
    if eligible_stars.empty:
        if (stars_df['NObs'] == maxObs).all():
            break

        mission_clock += 1/365
        continue

    # If the target list is not empty, pick the top-ranked star
    star = eligible_stars.sort_values("Score", ascending=False).iloc[0]

    # Physical properties of this star
    star_name = star["HDName"]
    star_mass = star["M_sol"]
    star_lum  = star["L_sol"]

    # Planets around this star
    # In this representative case, there is only 1 planet
    # For multi-planet system handling, see scripts related to Figs. 5 and 6
    star_planets = planets_df[planets_df['HDName'] == star_name]

    # Update star observation timing
    stars_df.loc[stars_df["HDName"] == star_name, 'LastObs'] = mission_clock   # The star was last observed now i.e. mission_clock
    stars_df.loc[stars_df["HDName"] == star_name, 'NObs'] += 1                 # Increment the number of times the star was observed by 1

    # Loop through all the planets around this star
    for _, planet in star_planets.iterrows():

        # Empty list to keep track of the observing log for one single observation
        # This will be appended to ObsLog at the end of the loop iteration
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

        # ObsLog Cols 3-9. 
        # SMA (in AU), Ecc, Rp (in R_E), M_p (in M_E), distance to the planet (in pc), Stellar mass (in M_solar), Stellar luminosity (in L_solar)
        Obs_row.extend([sma, ecc, planet['Rp_Rearth'], Mp, d, star_mass, star_lum])

        # Compute sep, PA, 3D dist and phase function
        # DetParams = [2D Sep (in arcsec), Position Angle (in deg), 3D star-planet sep (in AU), Orbital Phase Function]
        # orbituary is a local library that will have to be installed. 
        # See detailed in REQUIREMENTS.md in the root directory
        DetParams = solve_orbit(sma, ecc, inc, aop, pan, epp, P, mission_clock, d)

        # Run detection checks (Within working angles AND above the contrast floor)
        check1 = sepCheck(DetParams)
        check2 = contrastCheck(DetParams, albedo, Rp)

        sep_true, pa_true, r3d_au, phase_func = DetParams
        sep_logged, pa_logged = sep_true, pa_true

        # Check if the planet is detected
        if check1 and check2:
            planets_df.loc[planets_df['PlanetID'] == planet['PlanetID'], 'NDet'] += 1

            # Isotropic 3 mas uncertainty expressed in separation-PA coordinates.
            pa_uncertainty_deg = np.degrees(SEP_UNCERTAINTY / DetParams[0])
    
            sep_logged = rng.normal(loc=DetParams[0], scale=SEP_UNCERTAINTY)
            pa_logged  = rng.normal(loc=DetParams[1], scale=pa_uncertainty_deg) % 360.0 

            # ObsLog Cols 10-13. 
            # 2D Sep (in arcsec), PA (in degrees), 3D planet-star sep (in AU), Phase Func
            Obs_row.extend([sep_logged, pa_logged, r3d_au, phase_func])

            Obs_row.append(1)  # Detected, ObsLog Col. 14
        else:
            # ObsLog Cols 10-13. 
            # 2D Sep (in arcsec), PA (in degrees), 3D planet-star sep (in AU), Phase Func
            Obs_row.extend([sep_logged, pa_logged, r3d_au, phase_func])

            Obs_row.append(0)  # Not detected, ObsLog Col. 14

        # ObsLog Cols. 15-17: NDet, NObs (star), LastObs (star)
        Obs_row.append(int(planets_df.loc[planets_df['PlanetID'] == planet['PlanetID'], 'NDet'].iloc[0]))
        Obs_row.append(int(stars_df.loc[stars_df["HDName"] == star_name, 'NObs'].iloc[0]))
        Obs_row.append(mission_clock)

        ObsLog.loc[len(ObsLog)] = Obs_row

    # Advance clock by half a day 
    # We assume the total program time with any overheads takes 0.5 days
    mission_clock += 0.5/365
# --------------------------------------------------------------------------- #

# ---------------------------------- HZ Check ------------------------------- #
L = ObsLog['L_sol']
sma = ObsLog['SMA_AU']
ecc = ObsLog['ecc']

# HZ Limits
hz_inner, hz_outer = HZ(L)

# Check if the planet is in the HZ
hz_mask = (sma * (1 + ecc) < hz_outer) & (sma * (1 - ecc) > hz_inner)
ObsLog['HZ'] = hz_mask
# --------------------------------------------------------------------------- #

# Save the observing log as a .csv
ObsLog = ObsLog.round(3)
ObsLog.to_csv(curr_dir / "3. Observing Log.csv", index=False)