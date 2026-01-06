"""
Tongue Plot Generator (Part 2 of 9)
-------------------------------------------
This script generates a 4D (radius x period x eccentricity x stars) tongue plot,
for a chosen set of telescope and coronagraph parameters.

Input:
    5a. Observing Log.csv       # The observing log for the chosen telescope configuration (from Part 1)


Output:
    5b. 4D Tongue Plot.npz      # The 4D tongue plot stored a 4D NumPy array
"""

import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# solve_orbit is a local library that will need to be installed
# Installation intructions can be found in REQUIREMENTS.md in the root directory
from solve_orbit import solve_all_epochs_vectorized_full

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> USER-TUNABLE LIMITS <<<<<<<<<<<<<<<<<<<<<<<<<< #
# Telescope constraints
IWA                  = 0.06
OWA                  = 1 
CONTRAST_FLOOR       = 2.5e-11

# Tongue Plot contraints
# Min and max value for the 3 dimensions
RADIUS_LIMS          = [0.01, 3.4]     # Minimum and maximum planet radius in Earth radii (R_E)
PERIOD_LIMS          = [0.01, 40]      # Minimum and maximum period in yrs
ECC_LIMS             = [1e-4, 0.99]    # Eccentricity limits are (1e-4, 0.99) since the beta function is defined over (0,1), NOT [0,1]

# Number of points in the grid
N_RAD_POINTS         = 61  
N_PER_POINTS         = 101  
N_ECC_POINTS         = 21

# Number of planets per tongue plot grid cell for completeness calculations
NUM_PLANETS_PER_CELL = 10

# Observing parameters
MISSION_START        = 2035
N_EPOCHS             = 8               # A planet is considered as detected on the completeness map if it is detected in any of the N_EPCOHS epochs.
EPOCH_SPACING        = 3               # Spacing between epochs in months

# Multiprocessing
N_CORES              = 10              # Number of cores this script is run on. This is SOLELY dependent on how many cores you have on your machine
                                       # If you are unsure, set it equal to 1
                                       # DO NOT FUCK AROUND WITH THIS NUMBER. YOUR COMPUTER WILL FREEZE, AT BEST 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #

def Inclinations(num_planets):
    """
    Generate inclination angles with a sin distribution.

    Parameters:
    num_planets (int): Number of planets to generate inclinations for.

    Returns:
    inclinations (np.array): Inclination angles in degrees, from 0 to 180, favoring edge-on orientations (i ~ 90 degrees).
    """
    # Generate uniform numbers from -1 to 1, and take the arccos to get inclination angles with a sine distribution 
    # (This blew my mind! Taking the arccos of a uniform distrbution gives a sine distribution)
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

def EarthLikeAlbedo(num_planets):
    """
    Assign random albedos to Earth-like planets using a normal distribution.

    Parameters:
    num_planets (int): Number of planets to assign albedos for.

    Returns:
    np.array: Array of albedo values.
    """
    albedos = np.random.normal(0.367, 0.05, num_planets)  # Mean=0.3, StdDev=0.05

    return np.clip(albedos, 0, 1)  

def assign_albedo(mass):
    """Assign geometric albedo based on planet mass in Earth masses."""
    albedo = np.zeros_like(mass)

    solar_mass_earth_units = 332946.0487

    # Rocky planets (< 2.04 M_earth): Earth- or Mars-like
    rocky = mass < 2.04
    albedo[rocky] = np.random.uniform(0.15, 0.35, size=rocky.sum())

    # Sub-Neptunes (2.04 - 95.16 M_earth): often hazy/dark
    subneptune = (mass >= 2.04) & (mass < 95.16)
    albedo[subneptune] = np.random.uniform(0.05, 0.25, size=subneptune.sum())

    # Gas Giants (95.16 - 317.8 M_earth): cloud effects dominate
    gas_giants = (mass >= 95.16) & (mass < 317.8)
    albedo[gas_giants] = np.random.uniform(0.3, 0.6, size=gas_giants.sum())

    # Giant Planets (317.8 - ~0.08 M_sol): Jupiter analogs
    giants = (mass >= 317.8) & (mass < 0.080 * solar_mass_earth_units)
    albedo[giants] = np.random.uniform(0.3, 0.5, size=giants.sum())

    # Brown dwarfs (> 0.08 M_sol)
    bdwarfs = mass >= 0.080 * solar_mass_earth_units
    albedo[bdwarfs] = np.random.uniform(0.01, 0.05, size=bdwarfs.sum())

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
    albedos = EarthLikeAlbedo(total_planets)

    # Initialize completeness array
    completeness_cube = np.zeros((len(rad_centers), len(per_centers), len(ecc_centers)))

    # Define the two observation epochs (initial and 2 months later)
    mission_start  = MISSION_START
    n_epochs       = N_EPOCHS
    cadence_months = EPOCH_SPACING

    # Array of epochs when each planet is observed
    epochs = mission_start + np.arange(n_epochs) * (cadence_months / 12)
    
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
        sep, _, r3d, phase = solve_all_epochs_vectorized_full(
            params_batch,
            epochs,
            star_row['Dist']
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
    
def generate_survey_completeness(stars_df, rad_centers, per_centers, ecc_centers, num_planets_per_cell):
    """
    Generate a survey-wide completeness map by parallel processing stars.
    """
    
    # Number of cores to run the script parallely on
    n_cores = N_CORES
    print(f"Using {n_cores} CPU cores")
    
    # Create args list with just the necessary data from each star
    args_list = [(i, 
                  {
                      'HDName': star['HDName'],
                      'M'     : float(star['M']),       # Convert to native Python float
                      'Dist'  : float(star['Dist'])     # Convert to native Python float
                  }, 
                  rad_centers, 
                  per_centers, 
                  ecc_centers,
                  num_planets_per_cell) 
                 for i, star in stars_df.iterrows()]
    
    # Process stars in parallel
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        completeness_maps = list(executor.map(process_star, args_list))

    # 4D completeness map
    survey_completeness_4d = np.stack(completeness_maps, axis=3)
    
    return survey_completeness_4d

if __name__ == '__main__':

    multiprocessing.set_start_method("spawn", force=True)

    # ------------------------------------ I/O ------------------------------------- #
    curr_dir   = Path(__file__).resolve().parent
    parent_dir = curr_dir.parent.parent
    data_dir   = parent_dir / "Planet Generation"

    # Load the stars that will be observed
    stars = pd.read_csv(data_dir / 'HWO Stars.csv')
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
        num_planets_per_cell=NUM_PLANETS_PER_CELL
    )

    np.savez(
    curr_dir / '5b. 4D Tongue Plot.npz',
    completeness=survey_completeness_4D,
    rad_centers=rad_centers,
    per_centers=per_centers,
    ecc_centers=ecc_centers,
    rad_edges=rad_grid,
    per_edges=per_grid,
    ecc_edges=ecc_grid
    )

        




