"""
Tongue Plot Generator 
-------------------------------------------
This script generates an exo-Earth-only 4D (radius x period x eccentricity x
stars) tongue plot.  Each star uses its realized 3- to 8-epoch schedule, and
cells outside that star's fully-in-HZ region are assigned zero completeness.

Input:
    3a. Star Visit Log.csv                 # One row per star visit from the adaptive scheduler


Output:
    3b. 4D Tongue Plot.npz                 # The 4D tongue plot stored a 4D NumPy array
"""

import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import astropy.units as u
import sys

# solve_orbit is a local library that will need to be installed
# Installation intructions can be found in REQUIREMENTS.md in the root directory
from orbituary.solve_orbit import solve_orbit

curr_dir = Path(__file__).resolve().parent
forecaster_dir = curr_dir.parent.parent.parent / "Forecaster"
sys.path.insert(0, str(forecaster_dir))

import mr_forecast as mr

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> USER-TUNABLE LIMITS <<<<<<<<<<<<<<<<<<<<<<<<<< #
# Telescope constraints
IWA                  = 0.08
OWA                  = 1 
CONTRAST_FLOOR       = 1e-9

# Tongue Plot contraints
# Min and max value for the 3 dimensions
RADIUS_LIMS          = [0.8, 1.4]      # Exo-Earth radius interval (R_E)
PERIOD_LIMS          = [0.03, 6.0]     # Envelope containing the HZs of the HWO target list (yr)
ECC_LIMS             = [1e-4, 0.99]    # Eccentricity limits are (1e-4, 0.99) since the beta function is defined over (0,1), NOT [0,1]

# Number of points in the grid
N_RAD_POINTS         = 41  
N_PER_POINTS         = 101  
N_ECC_POINTS         = 41

# Number of planets per tongue plot grid cell for completeness calculations
NUM_PLANETS_PER_CELL = 100

# Observing parameters
MISSION_START        = 2035
N_EPOCHS             = 8               # Upper bound; realized schedules contain either 3 or 8 epochs.
EPOCH_SPACING        = 3               # Spacing between epochs in months

# Multiprocessing
N_CORES              = 10              # Number of cores this script is run on. This is SOLELY dependent on how many cores you have on your machine
                                       # If you are unsure, set it equal to 1
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #

SEED = 42  # set to None for non-deterministic runs
rng  = np.random.default_rng(SEED)

def Inclinations(num_planets):
    """
    Generate inclination angles with a sin distribution.

    Parameters:
    num_planets (int): Number of planets to generate inclinations for.

    Returns:
    inclinations (np.array): Inclination angles in degrees, from 0 to 180, favoring edge-on orientations (i ~ 90 degrees).
    """
    # Generate uniform numbers from -1 to 1, and take the arccos to get inclination angles with a sine distribution 
    cos_inclinations = np.random.uniform(-1, 1, num_planets)
    inc = np.degrees(np.arccos(cos_inclinations))

    return inc

def OrbitalAngles(num_planets):
    """
    Generate random values for argument of periapsis (AOP) and
    longitude of the ascending node (LAN).

    Parameters:
    num_planets (int): Number of planets to generate angles for.

    Returns:
    aop (np.array): Argument of periapsis in degrees (0 to 360).
    lan (np.array): Longitude of ascending node in degrees (0 to 360).
    """
    aop = np.random.uniform(0, 360, num_planets)  # Argument of periapsis
    lan = np.random.uniform(0, 360, num_planets)  # Longitude of ascending node
    return aop, lan

def EpochOfPeriastronPassage(P, missionStart, num_planets):

    """
    Generate the epoch of periastron passage for each planet.

    Parameters:
    orbital_periods (float or np.array): Orbital period(s) in years.
    mission_start (float): Start year of the mission.

    Returns:
    epochs_of_periastron (float or np.array): Epoch of periastron passage (T₀) in years.
    """
    
    # Generate uniform mean anomalies [0, 2π]
    mean_anomalies = np.random.uniform(0, 2 * np.pi, num_planets)
    
    # Calculate corresponding epochs of periastron passage
    epp = missionStart - (P * mean_anomalies) / (2 * np.pi)
    
    return epp

def assign_albedo(mass_earth):
    """
    Assign geometric albedo by mass class (Earth masses).
    Rough priors: rocky ~0.3, sub-Neptunes dim, giants brighter, brown dwarfs very dim.
    """
    albedo = np.zeros_like(mass_earth, dtype=float)

    small_planet = (mass_earth < 10)
    albedo[small_planet] = np.clip(rng.normal(0.367, 0.05, size=small_planet.sum()), 0.15, 0.60)

    sub_n = (mass_earth >= 10) & (mass_earth < 95.16)
    albedo[sub_n] = np.clip(rng.beta(2.5, 5, size=sub_n.sum()), 0.15, 0.60)  # peak around 0.2

    gas = (mass_earth >= 95.16) & (mass_earth < 317.8)
    albedo[gas] = np.clip(rng.normal(0.45, 0.07, size=gas.sum()), 0.25, 0.60)

    giants = (mass_earth >= 317.8) & (mass_earth < 0.080 * u.M_sun.to(u.M_earth))
    albedo[giants] = np.clip(rng.normal(0.45, 0.07, size=giants.sum()), 0.25, 0.60)

    bd = (mass_earth >= 0.080 * u.M_sun.to(u.M_earth))
    albedo[bd] = rng.uniform(0.01, 0.05, size=bd.sum())

    return albedo

def check_detectability_vectorized(sep_arcsec, r3d_au, phase_func, albedo, Rp_earth,
                                 IWA=IWA, OWA=OWA, coronagraph_contrast=CONTRAST_FLOOR):
    """
    Vectorized detectability check for multiple planets at both epochs.
    
    Parameters
    ----------
    sep_arcsec : ndarray
        Angular separations in arcseconds. Should be a 2D array (n_planets, 2) for both epochs
    r3d_au : ndarray
        3D distances in AU. Should be a 2D array (n_planets, 2) for both epochs
    phase_func : ndarray
        Phase functions. Should be a 2D array (n_planets, 2) for both epochs
    albedo : ndarray
        Planet albedos
    Rp_earth : ndarray
        Planet radii in Earth radii
    IWA : float
        Inner working angle in arcseconds
    OWA : float
        Outer working angle in arcseconds
    coronagraph_contrast : float
        Coronagraph contrast
        
    Returns
    -------
    ndarray
        Boolean array indicating detectability (True if detected in either epoch)
    """
    # Convert Earth radii to meters
    Rp_m = Rp_earth * 6.371e6
    
    # Convert AU to meters
    r3d_m = r3d_au * 1.496e11
    
    # Working angle check for both epochs
    sep_check = (sep_arcsec > IWA) & (sep_arcsec < OWA)
    
    # Contrast check for both epochs
    contrast = albedo[:, np.newaxis] * (Rp_m[:, np.newaxis]/r3d_m)**2 * phase_func
    contrast_check = contrast > coronagraph_contrast
    
    # Combined detectability check for each epoch
    detectability_per_epoch = sep_check & contrast_check
    
    # Planet is considered detected if it's visible in either epoch
    return np.any(detectability_per_epoch, axis=1)

def generate_single_star_completeness(star_row, rad_centers, per_centers, ecc_centers, num_planets_per_cell):
    """
    Generate a completeness map for a single star.
    
    Parameters
    ----------
    star_row : dict
        Dictionary containing star properties (HDName, M, Dist)
    rad_centers : np.array
        Array of radius bin centers to test (in Earth masses)
    per_centers : np.array
        Array of period bin centers to test (in AU)
    ecc_centers : np.array
        Array of eccentricity axis bin centers to test
    num_planets_per_cell : int, optional
        Number of planets to simulate at each grid point
        
    Returns
    -------
    np.array
        2D array of completeness values (0-1) for each mass/SMA grid point
    """

    nR = len(rad_centers)
    nP = len(per_centers)
    nE = len(ecc_centers)

    # Create meshgrid of all mass-sma combinations
    rad_grid, per_grid, ecc_grid = np.meshgrid(rad_centers, per_centers, ecc_centers, indexing='ij')

    # Flatten each into a 1D array (An 2x3 array would turn into a 1D array of length 2x3=6)
    rad_flat = rad_grid.flatten()
    per_flat = per_grid.flatten()
    ecc_flat = ecc_grid.flatten()

    # Calculate total number of planets
    n_grid_points = len(rad_flat)
    total_planets = n_grid_points * num_planets_per_cell

    # Repeat the earlier 1D arrays num_planets_per_cell times
    # If there are 5 planets, the earlier arrays of length 6 would now have size 6x5=30
    radii = np.repeat(rad_flat, num_planets_per_cell)
    P     = np.repeat(per_flat, num_planets_per_cell)
    eccs  = np.repeat(ecc_flat, num_planets_per_cell)

    # Generate orbital parameters for all planets at once
    inc      = Inclinations(total_planets)
    aop, lan = OrbitalAngles(total_planets)
    
    # Compute sma 
    smas = ((P**2) * star_row['M']) ** (1/3)

    # Generate epp
    epp = EpochOfPeriastronPassage(P, missionStart=MISSION_START, num_planets=total_planets)

    # Generate albedos
    # Draw NUM_PLANETS_PER_CELL masses for each radius using P(M | R).
    #
    # We reuse these draws across period/eccentricity cells. This provides the
    # correct conditional albedo distribution while avoiding millions of
    # unnecessary Forecaster calls.
    mass_draw_radii = np.repeat(rad_centers, num_planets_per_cell)

    mass_draws = mr.Rpost2M(
        mass_draw_radii,
        unit="Earth",
        grid_size=1000,
        classify="No"
    )

    # Shape: radius x Monte Carlo realization
    albedo_by_radius = assign_albedo(mass_draws).reshape(nR, num_planets_per_cell)

    # Expand to match the ordering of radii, P, ecc, and Monte Carlo realization:
    # radius x period x eccentricity x realization
    albedos = np.broadcast_to(albedo_by_radius[:, np.newaxis, np.newaxis, :], (nR, nP, nE, num_planets_per_cell)).reshape(total_planets)

    # Initialize completeness array
    completeness_cube = np.zeros((len(rad_centers), len(per_centers), len(ecc_centers)))

    # Array of epochs when each planet is observed
    epochs = np.asarray(star_row["epochs"], dtype=float)
    
    # Process in manageable batches to avoid memory issues
    batch_size = 10000
    for start_idx in range(0, total_planets, batch_size):

        end_idx = min(start_idx + batch_size, total_planets)
        
        # ------------------------------------------------------------------------- #
        # (1) Assemble orbital parameters for this batch of planets
        # These are fixed physical properties for each planet being simulated
        # ------------------------------------------------------------------------- #
        params_batch = np.column_stack([
            smas[start_idx:end_idx],           # sma
            eccs[start_idx:end_idx],           # ecc
            inc[start_idx:end_idx],            # inc
            aop[start_idx:end_idx],            # aop
            lan[start_idx:end_idx],            # lan
            epp[start_idx:end_idx],            # epp
            P[start_idx:end_idx]               # P
        ])

        # ------------------------------------------------------------------------- #
        # (2) Compute orbital positions across all epochs for all planets
        # Outputs:
        #   - sep: sky-projected separation [arcsec]
        #   - r3d: 3D star-planet separation [AU]
        #   - phase: phase function (dimensionless)
        # All these have size (n_batch, n_epochs)
        #
        # Then
        #
        # (3) Check detectability across all epochs
        # Planet is detected if it is visible in ANY epoch (IWA/OWA + contrast)
        # ------------------------------------------------------------------------- #

        # Calculate orbital positions for all epochs
        sep, _, r3d, phase = solve_orbit(
        params_batch[:, 0, None],   # sma
        params_batch[:, 1, None],   # eccentricity
        params_batch[:, 2, None],   # inclination
        params_batch[:, 3, None],   # AOP
        params_batch[:, 4, None],   # PAN
        params_batch[:, 5, None],   # EPP
        params_batch[:, 6, None],   # period
        epochs[None, :],
        float(star_row["Dist"])
        )
            
        # Check if the planet is detected at any of the epochs
        detection = check_detectability_vectorized(
            sep,
            r3d,
            phase,
            albedos[start_idx:end_idx],
            radii[start_idx:end_idx]
        )

        # ------------------------------------------------------------------------- #
        # (4) Assign each detection back to its mass-SMA-ecc grid cell
        # WARNING - Some serious array index gymnastics to follow
        # PROBLEM - We flattened our radius x period x ecc grid to 1D, and we need to put the detected planets back correctly
        # TO start with, we have n_planets = batch_size. These have to be placed in the 3D tongue plot

        # Consider n_radius x n_per x n_ecc = 2 x 3 x 4 = 24 grid cells
        # Since there are 10 planets per cell, that gives 24 x 10 = 240 planets
        #
        # Assume our planet index is 5 (0-indexed)
        #
        # 1. Find the correct mass x sma x ecc grid:
        #       Each grid has num_planets_per_cell planets
        #       Step 1 is to divide flattened planet index by the number of planets in each grid
        #       Planet 5 lives in grid 5 // 10 = 0 
        # 2. Find the right mass grid:
        #       Divide by the result from Step 1 by n_sma x n_ecc (or 3x4)
        #       Planet 5 lives in mass bin 1 // 12 = 0
        # 3. Find the sma x ecc grid
        #       Take the modulus of grid index with n_sma x n_ecc
        #       Planet 5 lives in the sma x ecc grid 0 % 12 = 0
        # 4. Find the sma bin
        #       Divide sma x ecc grid by n_ecc
        #       Planet 5 lives in the sma bin 0 // 4 = 0
        # 5. Find the ecc bin
        #       Take the modulus of the sma x ecc grid with n_ecc 
        #       Planet 5 lives in the ecc bin 0 % 4 = 0
        #  To wrap up, Planet 5 lives in the cell [0, 0, 0], which makes sense since:
        #       - If there are 10 planets per cell, planet 5 (0-indexed) must live in the first cell
        #  - Note: Eccentricity is the fastest moving grid, so planet 11 will live in the cell [0, 0, 1]
        # ------------------------------------------------------------------------- #
        planet_indices = np.arange(start_idx, end_idx)

        # Step 1
        grid_index = planet_indices // num_planets_per_cell
        # Step 2
        rad_idx = grid_index // (nP * nE)
        # Step 3
        rem = grid_index % (nP * nE)
        # Step 4
        per_idx = rem // nE
        # Step 5
        ecc_idx = rem % nE

        # Only keep planets detected in at least one epoch
        m = detection
        # Re-flatten detected (rad,per,ecc) -> 1D cell index
        flat = (rad_idx[m] * (nP * nE) + per_idx[m] * nE + ecc_idx[m])
        # Count detections per cell
        counts = np.bincount(flat, minlength=nR*nP*nE)
        # Add counts back into the 3D completeness cube
        completeness_cube += counts.reshape(nR, nP, nE)

    # Normalize completeness
    completeness_cube /= num_planets_per_cell

    sma_centers = (per_centers**2 * float(star_row["M"])) ** (1/3)

    hz_inner = np.sqrt(float(star_row["L"]) / 1.78)
    hz_outer = np.sqrt(float(star_row["L"]) / 0.32)

    hz_mask = (sma_centers[:, None] * (1.0 - ecc_centers[None, :]) >= hz_inner) & (sma_centers[:, None] * (1.0 + ecc_centers[None, :]) <= hz_outer)

    completeness_cube *= hz_mask[None, :, :]
    
    return completeness_cube

def process_star(args):
    """
    Process a single star for completeness calculations.
    
    Parameters:
    args: tuple containing (index, star_dict, mass_centers, sma_centers, num_planets_per_cell)
    """
    i, star_dict, rad_centers, sma_centers, ecc_centers, num_planets_per_cell = args
    print(f"\nProcessing star {i+1}: {star_dict['HDName']}", flush = True)  
    
    return generate_single_star_completeness(
        star_dict,
        rad_centers,
        sma_centers,
        ecc_centers,
        num_planets_per_cell
    )
    
def generate_survey_completeness(stars_df, rad_centers, per_centers, ecc_centers, num_planets_per_cell, epoch_lookup):
    """
    Generate a survey-wide completeness map by parallel processing stars.
    """
    
    # Number of cores to run the script parallely on
    n_cores = N_CORES
    print(f"Using {n_cores} CPU cores")

    args_list = []

    for i, star in stars_df.iterrows():
        star_name = star["HDName"]

        epochs = epoch_lookup[star_name]

        args_list.append((
            i,
            {
                "HDName": star_name,
                "M": float(star["M"]),
                "L": float(star["L"]),
                "Dist": float(star["Dist"]),
                "epochs": np.asarray(epochs, dtype=float)
            },
            rad_centers,
            per_centers,
            ecc_centers,
            num_planets_per_cell
        ))
    
    # Process stars in parallel
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        completeness_maps = list(executor.map(process_star, args_list))

    # 4D completeness map
    survey_completeness_4d = np.stack(completeness_maps, axis=3)
    
    return survey_completeness_4d


def build_epoch_lookup_from_obslog(obslog, stars_df, max_epochs=N_EPOCHS):
    """
    Build a dictionary mapping each star to its actual observation times.

    The observing log has one row per planet per visit, so we take the unique
    LastObs values for each star.
    """

    star_col = "StarName"
    time_col = "LastObs"

    epoch_lookup = {}

    for star_name in stars_df["HDName"]:
        times = (obslog.loc[obslog[star_col] == star_name, time_col].astype(float).to_numpy())

        # The log has duplicate times because every planet gets a row per visit.
        # Round before unique to avoid tiny floating-point differences.
        times = np.unique(np.round(times, 8))
        times = np.sort(times)

        epoch_lookup[star_name] = times


    return epoch_lookup

if __name__ == '__main__':

    multiprocessing.set_start_method("spawn", force=True)

    # ------------------------------------ I/O ------------------------------------- #
    curr_dir   = Path(__file__).resolve().parent
    parent_dir = curr_dir.parent.parent.parent
    data_dir   = parent_dir / "Planet Generation"

    # Load the stars that will be observed
    stars = pd.read_csv(data_dir / 'HWO Stars.csv')

    # Read in the observing log
    obslog = pd.read_csv(curr_dir / '3a. Observing Log.csv')

    # Build actual adaptive visit schedule for each star
    epoch_lookup = build_epoch_lookup_from_obslog(obslog, stars, max_epochs=N_EPOCHS)
    # ------------------------------------------------------------------------------ #

    # -----------------------------  TONGUE PLOT SETUP  ---------------------------- #

    # Define the radius, period and eccentrcitiy ranges
    radius_min = RADIUS_LIMS[0]     # Minimum planet radius in Earth radii (R_E)
    radius_max = RADIUS_LIMS[1]     # Maximum planet radius in Earth radii (R_E)
    period_min = PERIOD_LIMS[0]     # Minimum period in yrs
    period_max = PERIOD_LIMS[1]     # Maximum period in yrs
    ecc_min    = ECC_LIMS[0]        # Eccentricity limits are (1e-4, 0.99) since the beta function is defined over (0,1), NOT [0,1]
    ecc_max    = ECC_LIMS[1]

    # Number of points in the grid s
    n_rad_points = N_RAD_POINTS 
    n_per_points = N_PER_POINTS  
    n_ecc_points = N_ECC_POINTS

    # Generate logarithmic grids in radius and period, and a linear grid in eccentricity
    rad_grid = np.logspace(np.log10(radius_min), np.log10(radius_max), n_rad_points)
    per_grid = np.logspace(np.log10(period_min), np.log10(period_max), n_per_points)
    ecc_grid = np.linspace(ecc_min, ecc_max, n_ecc_points)

    # Calculate cell centers along all 3 grids
    rad_centers = np.sqrt(rad_grid[:-1] * rad_grid[1:])
    per_centers = np.sqrt(per_grid[:-1] * per_grid[1:])
    ecc_centers = 0.5 * (ecc_grid[:-1] + ecc_grid[1:])
    # ----------------------------- ------------------------------------------------ #

    # Generate the survey-wide completeness
    survey_completeness_4D = generate_survey_completeness(
        stars,
        rad_centers,
        per_centers,
        ecc_centers,
        num_planets_per_cell=NUM_PLANETS_PER_CELL,
        epoch_lookup = epoch_lookup
    )

    np.savez(
    curr_dir / '3b. 4D Tongue Plot.npz',
    completeness=survey_completeness_4D,
    rad_centers=rad_centers,
    per_centers=per_centers,
    ecc_centers=ecc_centers,
    rad_edges=rad_grid,
    per_edges=per_grid,
    ecc_edges=ecc_grid
    )

        




