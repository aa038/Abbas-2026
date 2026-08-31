"""
Adaptive Observing Log
---------------------------------------------------
Runs an adaptive cadence on the input SAG13 population
(Fig. 1). Loops through mission time, selects the next eligible target, calls
`solve_orbit(...)` to get on-sky separation/phase, applies simple detectability
checks (IWA/OWA + contrast), and logs each visit.

Inputs:
    1. Planet Catalog w Mass.csv   # From Part 2 (Same directory)

Outputs:
    4. Observing Log.csv           # Visit-by-visit log with detections

Notes:
- Telescope/mirror area and basic constants are defined in-file.
- This script is the "adaptive cadence" baseline; the uniform variant
  lives in Part 3 and reads/writes analogous files.
"""

# ------------------------ Scientific Assumptions --------------------------- #
# - Planet observed using an adaptive cadence (See Sec 3.3 in Abbas et al. 2026)
# - Telescope parameters shown below
# - Planet is considered detected if:
# -     (a) 2D star-planet sep is within the working angles
# -     (b) star-planet contrast is above the defined contrast floor
# - The detection probability (p_det in Eqn 6, Abbas et al 2026) is computed in the function detectability_fraction() 
#       - If both planet radius and albedo is known, use both contrast and working angles checks
#       - If not, consider using only working angle checks for realism

# - The mission modelling assumes idealised conditions, with minimal noise modelling. The paper focuses mostly on observing cadence and scheduling 
# --------------------------------------------------------------------------- #

# ------------------------------ WARNING ----------------------------------- #
# Compared to the uniform-cadence baseline (Part 3), this script introduces:
#   - orbit fitting after each detection,
#   - time-dependent detectability forecasts,
#   - objective functions for next-epoch selection,
#   - a scheduler that optimizes information gain subject to detectability constraints.

# As a result, this script is more complex than the uniform cadence baseline,
# and is structured in modular sections for readability.
# It is COMPUTATIONALLY DEMANDING and is typically only run to generate an independent sample,
# or experimenting with alternative cadence strategies.
# Casual users should use the provided data products directly.
# -------------------------------------------------------------------------- #

import pandas as pd
import numpy as np
from pathlib import Path

# orbituary is a local library that will need to be installed
# Installation intructions can be found in REQUIREMENTS.md in the root directory
from orbituary.solve_orbit import solve_orbit
from orbituary.orbituary_interface import fit_orbit

# ----------------------------  Reproducibility  ---------------------------- #
SEED = 42  # set to None for non-deterministic runs
rng  = np.random.default_rng(SEED)
np.random.seed(SEED)                   # For the Orbituary internal draws
# --------------------------------------------------------------------------- #


# -------------------------------  Constants  ------------------------------- #
# Telescope Parameters
CONTRAST_FLOOR        = 1.0e-09        # Contrast floor for this fiducial-case directory
IWA                   = 0.08           # Inner working angle in arcsec
OWA                   = 1              # Outer working angle in arcsec

# Physical constants
SOLARLUM              = 3.83e26
EARTHRADII            = 6.371e6
AU_TO_M               = 1.496e11
PC_TO_M               = 3.086e16
 
# Mission Parameters
maxObs                = 8              # Maximum number of times a star is revisited for observations
SCREENING_EPOCHS      = 3              # Every star receives this many visits before triage
EXO_EARTH_RADIUS_LIMS = (0.8, 1.4)     # R_Earth
TIME_SPACING          = 0.5            # How finely spaced the array of future times is, when scheduling an adaptive epoch (in days)
MIN_FRAC_GAP          = 0.25           # Minimum time to wait before scheduling an adaptie epoch (scaled by the HZ period, see Eqn 4) (in years)
MAX_WAIT_YEARS        = 1.00           # Next adaptive observation must be no more than one calendar year away
P_DET_MIN             = 0.5            # Minimum preferred detectability fraction for an epoch 
TIME_PER_STAR         = 0.5            # Total program time (setup + exposure + readout) for one star in days           

MISSION_START         = 2035.00
MISSION_END           = 2045.00

MCMC_STEPS            = 20000          # No of orbit steps taken by the MCMC sampler (Time complexity scales as O(N)) 
MCMC_BURNIN           = 2000           # MCMC Burnin

# Orbit-fitting assumptions
SEP_UNCERTAINTY       = 0.003         # arcsec
SMA_BOUNDS            = (0.1, 15.0)   # AU
NUM_ORBITS            = 500

# Posterior probability required to classify an orbit as HZ or non-HZ
HZ_CLASSIFICATION_THRESHOLD = 0.95

ORBIT_COLUMNS = ["sma", "ecc", "inc", "aop", "pan", "epp", "period"]
# --------------------------------------------------------------------------- #

# ------------------------------- HZ utilities ------------------------------ #
def HZ(L):
    """
    Return inner/outer HZ edges (AU) for stellar luminosity L (in L_sun).
    """

    hz_inner = np.sqrt(L / 1.78)
    hz_outer = np.sqrt(L / 0.32)

    return hz_inner, hz_outer


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
# --------------------------------------------------------------------------- #

# ---------------------- Star ranking (simple/static) ----------------------- #
def RankStars(starsDF, weights = [1, 2, 1]):
    """
    Rank stars based on distance, brightness, and spectral type to choose for observations.
    This is a simple formulation, since nuanced target prioritization is beyond the scope of this work.
    
    Parameters:
    starsDF (pd.DataFrame): DataFrame containing star and planet properties.
    
    Returns:
    pd.DataFrame: Ranked DataFrame with a 'Score' column.
    """

    wDist, wLum, wSpecType = weights

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
# --------------------------------------------------------------------------- #

# --------------- Section 1: One-visit observation of a star ---------------- #
def sepCheck(sep_arcsec, IWA = IWA, OWA = OWA):
    """
    Check is the 2D planet - star separation is within telescope working angles

    This is check 1/2 for planet detectability
    """
    return (sep_arcsec > IWA) & (sep_arcsec < OWA)


def contrastCheck(phase_fn, Rp_RE, r3D_AU, albedo, contrast_floor = CONTRAST_FLOOR):
    """
    Check if the observed planet contrast is above the coronagraph contrast floor.

    This is check 2/2 for planet detectability
    """
    
    # Compute the planet radius, and distance to the planet in SI units
    Rp_m  = Rp_RE * EARTHRADII
    r3D_m = r3D_AU * AU_TO_M

    # Compute the planet - star contrast
    contrast = albedo * (Rp_m / r3D_m)**2 * phase_fn

    return contrast >= contrast_floor


def observe_star_once(star_name: str, t_year: float):
    """
    Logs one visit for all planets of `star_name` at epoch `t_year`.
    - Always logs a row (DetStatus 0/1) per planet.
    - Increments planet NDet on detection.
    - Increments star NObs.
    """
    global ObsLog, StarVisitLog
    
    # Use the star name to retrieve its physical properties from the stars_df dataframe
    s_idx = star_idx[star_name]
    s_row = stars_df.iloc[s_idx]
    d_pc, L_sol, M_sol = s_row['d_pc'], s_row['L_sol'], s_row['M_sol']

    # Planets for this star
    planets = planets_df[planets_df['HDName'] == star_name]
    
    # Loop through all the planets and log observations
    # NextObs will be filled in later by the scheduler
    new_rows = []   # List to store obs logs for all planets in the system 

    for _, p in planets.iterrows():
        sma     = p['sma_AU']
        ecc     = p['ecc']
        inc_deg = np.degrees(p['inc_rad'])
        aop_deg = np.degrees(p['aop_rad'])
        pan_deg = np.degrees(p['pan_rad'])
        epp_yr  = p['epp_yr']
        P_yr    = p['P_yr']
        Rp_Re   = p['Rp_Rearth']
        Mp_Me   = p['Mp_Mearth']
        albedo  = p['albedo']

        # Compute sep, PA, 3D dist and phase function
        # solve_orbit is part of the local library orbituary, that will need to be installed
        # Installation intructions can be found in REQUIREMENTS.md in the root directory
        Sep_true, PA_true, r3D_AU, PhaseFunc = solve_orbit(sma, ecc, inc_deg, aop_deg, pan_deg, epp_yr, P_yr,t_year, d_pc)

        # Check for detections - within the working angles and above the contrast floor
        det = int(sepCheck(Sep_true) and contrastCheck(PhaseFunc, Rp_Re, r3D_AU, albedo))

        if det:
            # Use the same uncertainty model assumed by fit_planet_orbit()
            pa_uncertainty_deg = np.degrees(SEP_UNCERTAINTY / Sep_true)

            # Generate the measured astrometry
            Sep_arcsec = rng.normal(loc=Sep_true, scale=SEP_UNCERTAINTY)
            PA_deg     = rng.normal(loc=PA_true, scale=pa_uncertainty_deg) % 360.0

            # Separation must remain physically positive
            Sep_arcsec = max(Sep_arcsec, np.finfo(float).eps)

        else:
            # A nondetection does not provide measured astrometry
            Sep_arcsec = np.nan
            PA_deg     = np.nan

        # If detected, increment NDet in the planets dataframe (planets_df)
        # This information will be referenced for the observing log in the next line
        if det:
            planets_df.loc[planets_df['PlanetID'] == p['PlanetID'], 'NDet'] += 1

        new_rows.append([p['PlanetID'], star_name, sma, ecc, Rp_Re, Mp_Me, d_pc, M_sol, L_sol, Sep_arcsec, PA_deg, r3D_AU, PhaseFunc, det,
            int(planets_df.loc[planets_df['PlanetID'] == p['PlanetID'], 'NDet'].iloc[0]), # Number of times this planet has been detected
            int(stars_df.loc[s_idx, 'NObs']) + 1,  # Number of times this star has been observed
            t_year  # Time of observation 
        ])

    # Append all the planets around this star to the ObsLog
    if new_rows:
        ObsLog_batch = pd.DataFrame(new_rows, columns=ObsLog.columns)

        if ObsLog.empty:
            ObsLog = ObsLog_batch
        else:
            ObsLog = pd.concat([ObsLog, ObsLog_batch], ignore_index=True) 

    # Increment star NObs by 1
    stars_df.at[s_idx, 'NObs'] = stars_df.at[s_idx, 'NObs'] + 1
    # Placeholder. NextObs is computed by the scheduler later
    stars_df.at[s_idx, 'NextObs'] = np.nan  
    # Time the star was last observed i.e now!
    stars_df.at[s_idx, 'LastObs'] = t_year

    # A separate star-level log is required for targets that have no simulated
    # P < 10 yr planets and therefore contribute no rows to ObsLog.
    visit_row = pd.DataFrame([{
        'StarName': star_name,
        'NObs': int(stars_df.at[s_idx, 'NObs']),
        'LastObs': float(t_year)
    }])
    StarVisitLog = pd.concat([StarVisitLog, visit_row], ignore_index=True)
# --------------------------------------------------------------------------- #

# ---------- Section 3: Orbit fitting + HZ classification utilities --------- #
def classify_orbit_fully_hz(sma_au, ecc, L_sol):
    """
    For a given sma (in AU) and ecc, classify an orbit as fully within the HZ or not
    """
    # Compute the HZ boundaries (in AU)
    hz_inner, hz_outer = HZ(L_sol)

    # Compute periastron and apastron for the orbit (in AU)
    peri = sma_au * (1.0 - ecc)
    ap   = sma_au * (1.0 + ecc)

    # Orbit is in the HZ if both periastron and apastron are in the HZ
    return (peri >= hz_inner) & (ap <= hz_outer)


def get_planet_observations(planet_id):
    """
    Return times (yr), 2D separations (arcsec), position angles (deg) 
    for *detections only* of a given planet, sorted by time.

    This information will be used to fit orbits, which will inform the adaptive scheduler
    """

    # Retrieve *detections only* for this planet from the observation log
    obs = ObsLog[(ObsLog['PlanetID'] == planet_id) & (ObsLog['DetStatus'] == 1)]
    if obs.empty:
        return np.array([]), np.array([]), np.array([])
    
    # Sort the observations by the time of detection, and return the astrometric data with detection times
    obs = obs.sort_values('LastObs')
    t   = obs['LastObs'].to_numpy()
    sep = obs['Sep'].to_numpy()
    pa  = obs['PA'].to_numpy()

    return t, sep, pa


def fit_planet_orbit(planet_id, star_name):
    """
    Run the fitter on all the available astrometric epochs of the planet

    Returns the samples DataFrame with the 7 orbit columns:
    [sma_AU, ecc, inc_deg, aop_deg, pan_deg, epp_yr, P_yr]
    """
    # Pull the necessary star properties for orbit fitting
    srow = stars_df.loc[stars_df['HDName'] == star_name]
    M_sol = float(srow['M_sol'].iloc[0])
    d_pc  = float(srow['d_pc'].iloc[0])

    # Get the detected astrometric data for the planet
    times, seps, pas = get_planet_observations(planet_id)

    if times.size == 0:
        return None

    try:
        pa_uncertainty = np.degrees(SEP_UNCERTAINTY/seps)

        orbits, diagnostics = fit_orbit(times, seps, pas, M_sol, d_pc,
            sep_uncertainty=SEP_UNCERTAINTY, pa_uncertainty=pa_uncertainty, sma_bounds=SMA_BOUNDS, num_orbits=NUM_ORBITS,
            mcmc_steps=MCMC_STEPS, mcmc_burnin=MCMC_BURNIN, progress=None)
        return orbits
    except Exception as e:
        print(f"fit_orbit failed for PlanetID {planet_id}: {e}")
        return None

    
def hz_mask_for_orbit_posteriors(orbit_fits, L_sol):
    """
    Given posterior the orbital posteriors for a given planet,
    return boolean mask for fully-in-HZ orbits.
    """

    if orbit_fits is None or orbit_fits.empty:
        return np.array([], dtype=bool)

    sma = orbit_fits["sma"].to_numpy()
    ecc = orbit_fits["ecc"].to_numpy()

    # Classify the orbits as fully in the HZ or not
    hz_membership = classify_orbit_fully_hz(sma, ecc, L_sol)

    return hz_membership


def hz_confidence(orbit_fits, L_sol):
    """
    Fraction of orbital posteriors fully inside HZ.
    """

    # Boolean mask for fully HZ orbits
    mask = hz_mask_for_orbit_posteriors(orbit_fits, L_sol)
    
    # Fraction of orbits fully within the HZ
    # Since the mask is Boolean, the mean suffices for this
    conf = float(mask.mean())

    classified_hz = conf >= HZ_CLASSIFICATION_THRESHOLD

    classified_non_hz = conf <= 1.0 - HZ_CLASSIFICATION_THRESHOLD
    is_classified     = classified_hz or classified_non_hz

    return conf, is_classified


def evaluate_star_for_stopping(star_name):
    """
    True only when every detected planet is confidently not an exo-Earth.
    """

    detected = planets_df[(planets_df['HDName'] == star_name) & (planets_df['NDet'] > 0)]

    # No detections means we have not ruled out an exo-Earth
    if detected.empty:
        return False

    L_sol = float(stars_df.iloc[star_idx[star_name]]['L_sol'])
    radius_min, radius_max = EXO_EARTH_RADIUS_LIMS

    for _, planet in detected.iterrows():

        # Planet radius
        radius = float(planet['Rp_Rearth'])

        # A detected planet outside the radius range is non-exo-Earth
        # Assume the radius is known to reasonable precision after 3 epochs
        if not (radius_min <= radius <= radius_max):
            continue

        orbit_fits = fit_planet_orbit(planet['PlanetID'], star_name)

        # Failed or insufficient fit: unresolved, so keep observing
        if orbit_fits is None or orbit_fits.empty:
            return False

        confidence, _ = hz_confidence(orbit_fits, L_sol)

        # At least 5% HZ probability means it is not >95% confirmed non-HZ.
        if confidence >= 1.0 - HZ_CLASSIFICATION_THRESHOLD:
            return False

    return True
# --------------------------------------------------------------------------- #

# --------- Section 4: Objective functions for next-epoch selection --------- #
def xy_from(sep_arcsec, pa_deg):
    """
    Compute (x_sky, y_sky) for the planet from sep and PA
    """

    pa = np.radians(pa_deg)

    x = sep_arcsec * np.sin(pa)  # East
    y = sep_arcsec * np.cos(pa)  # North

    return x, y


def covariance_trace(x, y):
    """
    Compute tr(cov) ~= Var(x) + Var(y) along the sample axis.
    """
    # We want variance across samples
    var_x = np.var(x, axis=0, ddof=0)
    var_y = np.var(y, axis=0, ddof=0)
    return var_x + var_y


def centroid(x, y):
    """
    Mean across samples
    """
    return np.mean(x, axis=0), np.mean(y, axis=0)


def detectability_fraction(sep_arcsec, r3D_AU, phase_func, Rp_RE, albedo, IWA = IWA, OWA = OWA, contrast_floor = CONTRAST_FLOOR):
    """
    Compute the fraction of orbits that are detectable at any given time.

    This is best done using both contrast and 2D separation
    But contrast checks require knowledge of both the planet radius and albedo
    If radius is not constrained, use just the 2D sep constraint
    """

    sep_mask      = sepCheck(sep_arcsec, IWA, OWA)
    contrast_mask = contrastCheck(phase_func, Rp_RE, r3D_AU, albedo, contrast_floor)

    #det_mask = sep_mask & contrast_mask
    det_mask = sep_mask    

    # Total number of orbits
    N = det_mask.shape[0]

    # Find the mean "detectability" at some t across all the orbits
    det_frac = det_mask.sum(axis=0) / N  
    
    return det_frac


def propagate_orbits(orbit_fits, t_grid, d_pc):
    """
    Vectorized propagation for a *single planet* posterior sample set at all times.

    orbit_fits must contain columns in order: [sma, ecc, inc_deg, aop_deg, pan_deg, epp_yr, P_yr]
    Returns (sep_arcsec, pa_deg, r3D_AU, phase_func) arrays of shape (N_samples, T_times)
    """
    
    params = orbit_fits.loc[:, ORBIT_COLUMNS].to_numpy(dtype=float)
    times = np.asarray(t_grid, dtype=float)[None, :]

    return solve_orbit(
        params[:, 0, None],  # sma
        params[:, 1, None],  # ecc
        params[:, 2, None],  # inc
        params[:, 3, None],  # aop
        params[:, 4, None],  # pan
        params[:, 5, None],  # epp
        params[:, 6, None],  # period
        times,
        float(d_pc)
    )


def objective_unclassified(orbit_fits, L_sol, d_pc, t_grid, Rp_Re, albedo):
    """
    For an unclassified planet (Unclassified = 100% of orbital posteriors classified as fully HZ/non-HZ):
        - If both HZ and non-HZ subsets exist: normalized centroid separation x detectability.
        - If only one subset exists: internal spread x detectability (fallback).

    Returns: (J(t), p_det(t)) each shaped like t_grid (See Eqn 6 in Sec 3.3, Abbas et al. 2026)

    Read Section 3.3 in Abbas et al. 2026 for an in-depth description
    """
    # Split posteriors by fully inside the HZ or not
    hz_mask = hz_mask_for_orbit_posteriors(orbit_fits, L_sol)

    # Condition for a failed orbit fit (no available orbits)
    if hz_mask.size == 0:
        T = len(t_grid)
        return np.zeros(T), np.zeros(T)

    # Propagate all orbits forward in time (future times defined in t_grid)
    # These are 2D arrays of size (n_orbits, t_grid) ==> All orbits evaluated at all future times
    sep, pa, r3D, phase = propagate_orbits(orbit_fits, t_grid, d_pc)

    # Average detectability across orbits at each time
    p_det = detectability_fraction(sep, r3D, phase, Rp_Re, albedo)

    # If both HZ and non-HZ posteriors exist, choose a time that maximizes spread between HZ and non-HZ posteriors
    has_hz   = np.any(hz_mask)
    has_nohz = np.any(~hz_mask)

    if has_hz and has_nohz:

        # 2D sky coordinates of the HZ and non-HZ orbits as a function of time
        hz_x, hz_y       = xy_from(sep[hz_mask, :], pa[hz_mask, :])
        nonhz_x, nonhz_y = xy_from(sep[~hz_mask, :], pa[~hz_mask, :])

        # Compute the centroids for each class, as a function of time
        mx_hz_x, mx_hz_y   = centroid(hz_x, hz_y)
        mx_nhz_x, mx_nhz_y = centroid(nonhz_x, nonhz_y)

        # Compute the sky separation between the two centroids, as a function of time
        d_centroid = np.hypot(mx_hz_x - mx_nhz_x, mx_hz_y - mx_nhz_y)  

        # Compute the combined spread in both the HZ and non-HZ orbital solutions
        tr_hz    = covariance_trace(hz_x, hz_y)   
        tr_nonhz = covariance_trace(nonhz_x, nonhz_y)
        denom    = np.sqrt(tr_hz + tr_nonhz)

        # Eqn 5 (also see Appendix A, Eqn A5 and A6) from Abbas et al. 2026
        weighted_sep = np.divide(d_centroid, denom, out=np.zeros_like(d_centroid, dtype=float), where=denom > 0)

        J = weighted_sep * p_det

        # Prevent invalid values from reaching the scheduler
        J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)

        return J, p_det

    # Fallback: Only one class exists
    # Use internal spread of whichever class we have
    use_mask = hz_mask if has_hz else (~hz_mask)

    x, y   = xy_from(sep[use_mask, :], pa[use_mask, :])
    spread = np.sqrt(covariance_trace(x, y))  
    J      = spread * p_det

    # Prevent invalid values from reaching the scheduler
    J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)

    return J, p_det


def objective_classified(orbit_fits, L_sol, d_pc, t_grid, Rp_Re, albedo):
    """
    For a classified planet (entire posterior in one class),
    maximize internal spread x detectability for that class.

    Returns: (J(t), p_det(t)).
    """

    if orbit_fits is None:
        T = len(t_grid)
        return np.zeros(T), np.zeros(T)
    
    # Simplified version of the unclassified planet scheduling
    # See Sec 3.3 and Appendix A, Eqn A7 from Abbas et al. 2026 for detail 
    sep, pa, r3D, phase = propagate_orbits(orbit_fits, t_grid, d_pc)
        
    p_det = detectability_fraction(sep, r3D, phase, Rp_Re, albedo)

    x, y   = xy_from(sep, pa)
    spread = np.sqrt(covariance_trace(x, y)) 

    J = spread * p_det

    # Prevent invalid values from reaching the scheduler
    J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)

    return J, p_det


def pick_best_time(t_grid, J, p_det, pmin=P_DET_MIN):
    """
    Choose best time with guardrail p_det >= pmin. If none meet pmin, pick global max J.

    Returns (t_best, idx_best, met_guardrail: bool)
    """

    # Find all time indices where the detectability constraint is met
    eligible = np.where(p_det >= pmin)[0]

    if eligible.size > 0:
        idx = eligible[np.argmax(J[eligible])]
        return float(t_grid[idx]), int(idx), True
    
    # Fall back to absolute max
    idx = int(np.argmax(J))
    return float(t_grid[idx]), idx, False
# --------------------------------------------------------------------------- #

# ------------------------- Section 5: Scheduler ---------------------------- #
def build_time_grid(last_obs, mission_end, period_char, min_frac=MIN_FRAC_GAP,
                    max_wait_years=MAX_WAIT_YEARS, slot_days=TIME_SPACING):
    """
    Build the array of future times across which the adaptive scheduling is applied
    """

    # Minimum wait is scaled by the characteristic HZ period; maximum wait is
    # an absolute calendar-year limit.
    min_gap = min_frac * period_char
    max_gap = max_wait_years

    if min_gap > max_gap:
        raise ValueError(
            f"Minimum wait ({min_gap:.3f} yr) exceeds the "
            f"maximum wait ({max_gap:.3f} yr)."
        )

    # Use min_gap and max_gap to build the end points of an array of future times
    if np.isnan(last_obs):
        start  = MISSION_START + min_gap
        anchor = MISSION_START
    else:
        start  = last_obs + min_gap
        anchor = last_obs

    stop = min(anchor + max_gap, mission_end)

    # How finely spaced the time array is
    step = slot_days / 365.0

    return np.arange(start, stop + 1e-9, step)


def schedule_next_observation_for_star(star_name, mission_now, aggregate_mode=False):
    """
    Decide NextObs for `star_name`:
    - Build t_grid from the HZ-scaled minimum wait through at most one year
    - Evaluate J(t) per planet with a posterior
    - If aggregate_mode=False: pick the single (planet, t) with max J subject to p_det>=P_DET_MIN
      If aggregate_mode=True: sum J across planets per t, then pick best t.
    - Write stars_df['NextObs'].
    Returns (t_best, met_guardrail, reason_string)
    """

    s_idx = star_idx[star_name]
    s_row = stars_df.iloc[s_idx]
    d_pc  = float(s_row['d_pc'])
    L_sol = float(s_row['L_sol'])
    M_sol = float(s_row['M_sol'])

    P_hz  = characteristic_hz_period_years(L_sol, M_sol) 

    last_obs = s_row['LastObs']
    t_grid   = build_time_grid(last_obs, mission_end=MISSION_END, period_char=P_hz)

    # If we are at mission end, do a minimal fallback
    if t_grid.size == 0:
        stars_df.at[s_idx, 'NextObs'] = np.nan
        return None, False, "no_time_window"

    # Gather planets
    planets = planets_df[planets_df['HDName'] == star_name]

    # No planets detected yet
    if planets.empty:

        # Observe at the earliest, waiting for the minimum predefined wait period
        gap    = MIN_FRAC_GAP * P_hz
        anchor = last_obs if not np.isnan(last_obs) else mission_now
        t_best = anchor + gap
        t_best = min(t_best, MISSION_END)

        stars_df.at[s_idx, 'NextObs'] = t_best
        return t_best, True, "no_planets"

    # If there are multiple planets, compute J(t) for each planet
    per_planet_results = []  # list of dicts with keys: 'PlanetID','J','p_det','classified'
    any_with_posterior = False

    for _, p in planets.iterrows():
        pid   = p['PlanetID']
        Rp_Re = p['Rp_Rearth']
        alb   = p['albedo']

        # Orbit fit if there is atleast one detection
        orbit_fits = fit_planet_orbit(pid, star_name)

        if orbit_fits is None or orbit_fits.empty:
            continue

        # If atleast one planet was orbit fit successfully:
        any_with_posterior = True
        conf, classified = hz_confidence(orbit_fits, L_sol)

        if classified:
            J, p_det = objective_classified(orbit_fits, L_sol, d_pc, t_grid, Rp_Re, alb)
        else:
            J, p_det = objective_unclassified(orbit_fits, L_sol, d_pc, t_grid, Rp_Re, alb)

        per_planet_results.append({
            'PlanetID': pid,
            'J': J,
            'p_det': p_det,
            'classified': bool(classified)
        })

    # If no planet has posteriors yet, wait for the minimum predefined period
    if not any_with_posterior:
        gap    = MIN_FRAC_GAP * P_hz
        anchor = last_obs if not np.isnan(last_obs) else mission_now
        t_best = anchor + gap
        t_best = min(t_best, MISSION_END)

        stars_df.at[s_idx, 'NextObs'] = t_best
        return t_best, True, "no_posteriors"

    # If at least one planet has posteriors, run adaptive scheduling
    # There are 2 modes for multi-planet systems:
    #   1. Aggregate: Find the t that maximizes J (Eqn 6 for all planets)
    #   2. Non-aggregate (Default): Find the best t for the planet with the largest J (most likely to be HZ)

    if not aggregate_mode:
        # Find the planet with the maximum J
        best_tuple = None  # Tuple containing (t_best, idx, met_guardrail, PlanetID, J_val)
        for res in per_planet_results:
            # Compute the best time for this planet, and compute J at t_best
            t_candidate, idx, ok = pick_best_time(t_grid, res['J'], res['p_det'], pmin=P_DET_MIN)
            J_val = float(res['J'][idx])

            # Compae to other planets
            if (best_tuple is None) or (J_val > best_tuple[4]):  # best_tuple[4] = J_val
                best_tuple = (t_candidate, idx, ok, res['PlanetID'], J_val)

        t_best, idx_best, ok_best, pid_best, J_best = best_tuple
        stars_df.at[s_idx, 'NextObs'] = t_best
        return t_best, ok_best, f"best_planet={pid_best}"
    
    else:
        # Sum J across planets per time
        J_sum = np.zeros_like(t_grid, dtype=float)
        p_det_min = np.ones_like(t_grid, dtype=float)

        for res in per_planet_results:
            J_sum    += res['J']
            p_det_min = np.minimum(p_det_min, res["p_det"])

        # Find the time that maximises the combined J (J_sum)
        t_best, idx_best, ok_best = pick_best_time(t_grid, J_sum, p_det_min, pmin=P_DET_MIN)

        stars_df.at[s_idx, 'NextObs'] = t_best
        return t_best, ok_best, "aggregate_mode"
# --------------------------------------------------------------------------- #

# ---------------------------------- I/O ------------------------------------ #
curr_dir   = Path(__file__).resolve().parent
parent_dir = curr_dir.parent.parent.parent
data_dir   = parent_dir / "Planet Generation"

# Load the list of stars with planets around them
planets_df = pd.read_csv(data_dir / "SAG13 Planet Catalog.csv")
planets_df = planets_df[planets_df['P_yr'] < 10]
# --------------------------------------------------------------------------- #

# ---------------- Setting up the simulated observations -------------------- #
# For simulated observations, we need the following:
#   1. List of stars to observe
#   2. Rank them (Stars are observed as they hit they adaptively scheduled observation time. Ranking is used as a tie-breaker)
#   3. Create variables to keep track of
#       - How many times a star has been observed
#       - When it was observed last 
#       - When it should be observed next (solely a feature of adaptive scheduling. Not used in the uniform cadence)
#       - Boolean variable informing us if the all 8 (or maxObs) observations are complete


# ---- 1. Full HWO star catalog ---- #
# Build the queue from the target list 
star_catalog = pd.read_csv(data_dir / "HWO Stars.csv").rename(columns={'SpecType': 'Spec', 'Dist': 'd_pc', 'L': 'L_sol', 'M': 'M_sol'})
stars_df     = star_catalog[['HDName', 'Spec', 'd_pc', 'L_sol', 'M_sol']].copy()

# Precompute HZ edges per star (handy later)
hz_edges = np.array([HZ(L) for L in stars_df['L_sol'].values])
stars_df['HZ_in_AU']  = hz_edges[:,0]
stars_df['HZ_out_AU'] = hz_edges[:,1]

# ---- 2. Static, tie-breaker ranking (we will still schedule by NextObs first) ---- #
stars_df = RankStars(stars_df)

# ---- 3. Timekeeping variables ---- #
# Empty arrays to keep track of:
# - How many times a star has been observed
# - When it should be observed next (computed by the adaptive scheduler)
# - Have we observed this star the required number of times
# - When the stars was last observed
stars_df['NObs']                 = 0
stars_df['NextObs']              = MISSION_START  # Everyone eligible at start
stars_df['Done']                 = False
stars_df['LastObs']              = np.nan

# For easy scaling to multi-planet systems in the future, NDet is defined as a property of a star
# For the single planet uniform cadence, it was defined as a property of a star for computing ease
planets_df['NDet'] = 0
  

# Observation log (one row per attempted epoch per planet)
ObsLog = pd.DataFrame(columns=['PlanetID','StarName','SMA_AU','ecc','Rp_REarth','Mp_MEarth','d_pc','M_sol','L_sol',
    'Sep','PA','3dDist','PhaseFunc','DetStatus','NDet','NObs','LastObs'])

# Observing log per star
StarVisitLog = pd.DataFrame(columns=['StarName', 'NObs', 'LastObs'])

# Convenience indices for fast lookup
star_idx = {name:i for i, name in enumerate(stars_df['HDName'])}
# --------------------------------------------------------------------------- #
    
# ----------------------------- Mission loop -------------------------------- #
def all_done():
    """
    Boolean check to see if all stars have been observed maxObs times i.e. end of the mission
    """
    return bool((stars_df['Done'] | (stars_df['NObs'] >= maxObs)).all())


def next_global_time_after():
    """
    Earliest NextObs among not-done stars
    """
    pending = stars_df[~stars_df['Done'] & (stars_df['NObs'] < maxObs)]

    # If there are no stars, that means we are done with observations
    # Time to end the mission
    if pending.empty:
        return MISSION_END
    
    nxt = pending['NextObs'].min()

    return float(nxt)


def pick_eligible_star(now):
    """
    Pick one star to observe now:
    - Eligible if NextObs <= now and not done
    - Tie-breaker: earlier NextObs, then higher score (defined using RankStars)
    """
    eligible = stars_df[(~stars_df['Done']) & (stars_df['NObs'] < maxObs) & (stars_df['NextObs'] <= now)]
    
    if eligible.empty:
        return None
    
    # Tie-break: Earliest NextObs, then highest score using our simple ranking algorithm
    chosen = eligible.sort_values(['NextObs','Score'], ascending=[True, False]).iloc[0]
    return chosen['HDName']


def run_mission(save_csv=True, verbose=True):
    # Initialise the mission clock
    mission_clock = MISSION_START

    while (mission_clock <= MISSION_END) and (not all_done()):
        # Find any star eligible to observe at current time
        star_name = pick_eligible_star(mission_clock)

        # If there are no stars are ready to be observed,
        # move the clock forward in time until atleast one star is good to go
        if star_name is None:
            # Find the earliest time when a star is ready to be observed
            nxt = next_global_time_after()

            # Is this past the scheduled mission end time?
            if nxt > MISSION_END:
                if verbose:
                    print(f"[{mission_clock:.3f}] No further schedulable observations.")
                break

            mission_clock = nxt
            continue

        # Observe the star
        # This is a 2 parter:
        #   1. Check if a planet around it passes contrast + IWA checks, and update the Obs Log
        #   2. Move the mission clock forward in time
        if verbose:
            print(f"[{mission_clock:.3f}] Observing {star_name} (visit {int(stars_df.loc[star_idx[star_name],'NObs'])+1})")

        # Logs + increments NObs + sets LastObs (This does NOT set the adaptively scheduled NextObs)
        observe_star_once(star_name, mission_clock)  

        # Increment the mission clock
        mission_clock += TIME_PER_STAR / 365

        # After the star has been observed, there are two outcomes:
        #   1. The star was observed the required maxObs times
        #   In this case, the adaptive scheduler does not need to be run. Mark as Done and clear NextObs
        #
        #   2. The star has not hit maxObs yet
        #   Then, run the adaptive scheduler to pick the next best time
        sidx = star_idx[star_name]

        n_obs = int(stars_df.at[sidx, 'NObs'])

        # Maximum number of visits reached
        if n_obs >= maxObs:
            stars_df.at[sidx, 'Done']    = True
            stars_df.at[sidx, 'NextObs'] = np.nan

            if verbose:
                print(f"-> {star_name}: reached {maxObs} visits. Marked done.")

        # Starting at epoch 3, stop whenever all detected planets have been confirmed as non-exo-Earths
        elif n_obs >= SCREENING_EPOCHS and evaluate_star_for_stopping(star_name):

            stars_df.at[sidx, 'Done']    = True
            stars_df.at[sidx, 'NextObs'] = np.nan

            if verbose:
                print(f"-> {star_name}: all detected planets confirmed non-exo-Earth. Marked done.")

        # Otherwise continue with the existing scheduler.
        else:
            t_best, ok_guard, reason = schedule_next_observation_for_star(star_name, mission_clock, aggregate_mode=False)

    if save_csv:
        out = ObsLog.copy()
        num_cols = ['SMA_AU','ecc','Rp_REarth','Mp_MEarth','d_pc','M_sol','L_sol','Sep','PA','3dDist','PhaseFunc','LastObs']
        for col in num_cols:
            if col in out.columns:
                out[col] = np.round(out[col].astype(float), 6)
        out.to_csv(curr_dir / "3a. Observing Log.csv", index=False)
        StarVisitLog.to_csv(curr_dir / "3a. Star Visit Log.csv", index=False)

        if verbose:
            print(f"\nSaved {len(out)} rows to 3a. Observing Log.csv")
        

run_mission(save_csv=True)
# --------------------------------------------------------------------------- #
