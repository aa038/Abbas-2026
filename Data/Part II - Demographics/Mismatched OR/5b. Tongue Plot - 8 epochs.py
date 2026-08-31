import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u

from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.stats import beta
from pathlib import Path

# solve_orbit and PlotStyle are local libraries that will need to be installed
# Installation intructions can be found in REQUIREMENTS.md in the root directory
from orbituary.solve_orbit import solve_orbit
from PlotStyle import plotStyle

from forecaster import optimized_mass_to_radius

plotStyle()

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> USER-TUNABLE LIMITS <<<<<<<<<<<<<<<<<<<<<<<<<< #
# Telescope constraints
IWA                  = 0.06
OWA                  = 1 
CONTRAST_FLOOR       = 1e-10

# Tongue Plot contraints
# Min and max value for the 3 dimensions
MASS_LIMS            = [0.01, 40]     # Minimum and maximum planet mass in Earth masses
SMA_LIMS             = [0.1, 10]       # Minimum and maximum sma in au
ECC_LIMS             = [1e-4, 0.99]    # Eccentricity limits are (1e-4, 0.99) since the beta function is defined over (0,1), NOT [0,1]

# Number of points in the grid
N_MASS_POINTS        = 41  
N_SMA_POINTS         = 101  
N_ECC_POINTS         = 41

# Number of planets per tongue plot grid cell for completeness calculations
NUM_PLANETS_PER_CELL = 100

# Observing parameters
MISSION_START        = 2035
N_EPOCHS             = 8               # A planet is considered as detected on the completeness map if it is detected in any of the N_EPCOHS epochs.
EPOCH_SPACING        = 3               # Spacing between epochs in months

# Multiprocessing
N_CORES              = 10              # Number of cores this script is run on. This is SOLELY dependent on how many cores you have on your machine
                                       # If you are unsure, set it equal to 1
                                       # DO NOT FUCK AROUND WITH THIS NUMBER. YOUR COMPUTER WILL FREEZE, AT BEST 
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


def RadiusFromMass(masses, num_planets, n_samples=1000):
    """
    Generate Earth-like masses and radii with small random variations.

    Parameters:
    masses (np.array): Planet masses in Earth masses.
    num_planets (int): Number of planets to generate.

    Returns:
    radii (np.array): Planet radii in Earth radii.
    """

    masses = np.ones(num_planets) * masses

    # Convert masses to radii using optimized forecaster
    radii = optimized_mass_to_radius(masses)

    return masses, radii


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
                                 IWA=0.06, OWA=1, coronagraph_contrast=1e-10):
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


def generate_single_star_completeness(star_row, mass_centers, sma_centers, ecc_centers, num_planets_per_cell=10000):
    """
    Generate a completeness map for a single star.
    
    Parameters
    ----------
    star_row : dict
        Dictionary containing star properties (HDName, M, Dist)
    mass_centers : np.array
        Array of mass bin centers to test (in Earth masses)
    sma_centers : np.array
        Array of semi-major axis bin centers to test (in AU)
    ecc_centers : np.array
        Array of eccentricity axis bin centers to test
    num_planets_per_cell : int, optional
        Number of planets to simulate at each grid point
        
    Returns
    -------
    np.array
        2D array of completeness values (0-1) for each mass/SMA grid point
    """
    # Create meshgrid of all mass-sma combinations
    mass_grid, sma_grid, ecc_grid = np.meshgrid(mass_centers, sma_centers, ecc_centers, indexing='ij')
    mass_flat = mass_grid.flatten()
    sma_flat = sma_grid.flatten()
    ecc_flat = ecc_grid.flatten()

    # Calculate total number of planets
    n_grid_points = len(mass_flat)
    total_planets = n_grid_points * num_planets_per_cell

    masses = np.repeat(mass_flat, num_planets_per_cell)
    smas = np.repeat(sma_flat, num_planets_per_cell)
    eccs = np.repeat(ecc_flat, num_planets_per_cell)

    # Generate orbital parameters for all planets at once
    inc = Inclinations(total_planets)
    aop, lan = OrbitalAngles(total_planets)
    
    # Compute period (vectorized)
    P = np.sqrt(smas**3 / star_row['M'])  

    # Generate epp
    epp = EpochOfPeriastronPassage(P, missionStart=2035, num_planets=total_planets)

    # Compute the radii from mass
    masses, radii = RadiusFromMass(masses, total_planets)

    # Generate albedos
    albedos = assign_albedo(masses)

    # Initialize completeness array
    completeness_cube = np.zeros((len(mass_centers), len(sma_centers), len(ecc_centers)))

    # Array of epochs when each planet is observed
    epochs = np.asarray(star_row["epochs"], dtype=float)
    
    # Process in manageable batches to avoid memory issues
    batch_size = 10000
    for start_idx in range(0, total_planets, batch_size):

        end_idx = min(start_idx + batch_size, total_planets)
        #if start_idx % (batch_size*50) == 0:               # every 50 batches
            #print(f"    {star_row['HDName']}: {start_idx}/{total_planets}", flush=True)
        
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
        # All are shape (batch_size, n_epochs)
        #
        # Then
        #
        # (3) Check detectability across all epochs
        # Planet is detected if visible in ANY epoch (IWA/OWA + contrast)
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
        float(star_row["Dist"]),
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
        # PROBLEM - We flattened our mass x sma x ecc grid to 1D, and we need to put the detected planets back correctly
        # Assume our planet index is 5 (0-indexed), there are 5 planets per grid and assume the grid was (2, 3, 4)
        # 1. Find the correct mass x sma x ecc grid:
        #       Each grid has num_planets_per_cell planets
        #       Step 1 is to divide flatted planet index by the number of planets in each grid
        #       Planet 5 lives in grid 5 // 5 = 1 
        # 2. Find the right mass grid:
        #       Divide by the result from Step 1 by 3x4 or (n_sma x n_ecc)
        #       Planet 5 lives in mass bin 1 // 12 = 0
        # 3. Find the sma x ecc grid
        #       Take the modulus of grid index with n_sma x n_ecc
        #       Planet 5 lives in the sma x ecc grid 1 % 12 = 1 (remainder of 1 when divided by 12)
        # 4. Find the sma bin
        #       Divide sma x ecc grid by n_ecc
        #       Planet 5 lives in the sma bin 1 // 4 = 0
        # 5. Find the ecc bin
        #       Take the modulus of the sma x ecc grid with n_ecc 
        #       Planet 5 lives in the ecc bin 1 % 4 = 1
        #  To wrap up, Planet 5 lives in the bin [0, 0, 1], which makes sense since:
        #       - If there are 5 planets per bin, planet 5 (0-indexed) must live in the second bin
        #       - Eccentricity is the fastest moving grid
        # ------------------------------------------------------------------------- #
        for i, detected in enumerate(detection):
            planet_index = start_idx + i
            # Step 1 
            grid_index = planet_index // num_planets_per_cell
            # Step 2
            mass_idx = grid_index // (len(sma_centers) * len(ecc_centers))
            # Step 3
            remaining = grid_index % (len(sma_centers) * len(ecc_centers))
            # Step 4
            sma_idx = remaining // len(ecc_centers)
            # Step 5
            ecc_idx = remaining % len(ecc_centers)
            
            if (mass_idx < len(mass_centers) and 
                sma_idx < len(sma_centers) and 
                ecc_idx < len(ecc_centers)):
                completeness_cube[mass_idx, sma_idx, ecc_idx] += detected

    # Normalize completeness
    completeness_cube /= num_planets_per_cell
    
    return completeness_cube

def process_star(args):
    """
    Process a single star for completeness calculations.
    
    Parameters:
    args: tuple containing (index, star_dict, mass_centers, sma_centers, num_planets_per_cell)
    """
    i, star_dict, mass_centers, sma_centers, ecc_centers, num_planets_per_cell = args
    print(f"\nProcessing star {i+1}: {star_dict['HDName']}", flush = True)  # Changed from star.HDName to star_dict['HDName']
    return generate_single_star_completeness(
        star_dict,
        mass_centers,
        sma_centers,
        ecc_centers,
        num_planets_per_cell
    )
    
def generate_survey_completeness(stars_df, mass_centers, sma_centers, ecc_centers, num_planets_per_cell, epoch_lookup):
    """
    Generate a survey-wide completeness map by parallel processing stars.
    """
    
    print(f"Using {N_CORES} CPU cores")
    
    args_list = []
    
    for i, star in stars_df.iterrows():
        star_name = star["HDName"]

        epochs = epoch_lookup[star_name]

        args_list.append((
            i,
            {
                "HDName": star_name,
                "M": float(star["M"]),
                "Dist": float(star["Dist"]),
                "epochs": np.asarray(epochs, dtype=float)
            },
            mass_centers,
            sma_centers,
            ecc_centers,
            num_planets_per_cell
        ))
    
    # Process stars in parallel
    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        completeness_maps = list(executor.map(process_star, args_list))
    
    # Sum all the completeness maps
    survey_completeness = np.sum(completeness_maps, axis=0)

    # 4D completeness map
    survey_completeness_4d = np.stack(completeness_maps, axis=3)
    
    return survey_completeness, survey_completeness_4d


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
        times = obslog.loc[obslog[star_col] == star_name, time_col].dropna().astype(float).to_numpy()

        # The log has duplicate times because every planet gets a row per visit.
        # Round before unique to avoid tiny floating-point differences.
        times = np.unique(np.round(times, 8))
        times = np.sort(times)

        epoch_lookup[star_name] = times


    return epoch_lookup


if __name__ == '__main__':

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    multiprocessing.set_start_method("spawn", force=True)

    # ------------------------------------ I/O ------------------------------------- #
    curr_dir = Path(__file__).resolve().parent
    parent_dir = curr_dir.parent.parent
    data_dir = parent_dir / "Planet Generation"

    stars = pd.read_csv(data_dir / 'HWO Stars.csv')

    # Read in the observing log
    obslog = pd.read_csv(curr_dir / '5a. Observing Log.csv')
    
    # Build actual adaptive visit schedule for each star
    epoch_lookup = build_epoch_lookup_from_obslog(obslog, stars, max_epochs=N_EPOCHS)
    # ------------------------------------------------------------------------------ #

    # -----------------------------  TONGUE PLOT SETUP  ---------------------------- #

    # Define the ranges (kept the same as original)
    mass_min = MASS_LIMS[0]
    mass_max = MASS_LIMS[1]
    sma_min  = SMA_LIMS[0]
    sma_max  = SMA_LIMS[1]
    ecc_min  = ECC_LIMS[0]
    ecc_max  = ECC_LIMS[1]

    # Number of points in the grid 
    n_mass_points = N_MASS_POINTS  
    n_sma_points  = N_SMA_POINTS 
    n_ecc_points  = N_ECC_POINTS

    # Generate logarithmic grids 
    mass_grid = np.logspace(np.log10(mass_min), np.log10(mass_max), n_mass_points)
    sma_grid  = np.logspace(np.log10(sma_min), np.log10(sma_max), n_sma_points)
    ecc_grid  = np.linspace(ecc_min, ecc_max, n_ecc_points)

    # Calculate cell centers
    mass_centers = np.sqrt(mass_grid[:-1] * mass_grid[1:])
    sma_centers  = np.sqrt(sma_grid[:-1] * sma_grid[1:])
    ecc_centers  = 0.5 * (ecc_grid[:-1] + ecc_grid[1:])
    # ----------------------------- ------------------------------------------------ #

    # Generate the survey-wide completeness (new approach)
    survey_completeness, survey_completeness_4D = generate_survey_completeness(
        stars,
        mass_centers,
        sma_centers,
        ecc_centers,
        num_planets_per_cell=NUM_PLANETS_PER_CELL,
        epoch_lookup = epoch_lookup
    )

    np.savez(
    curr_dir / '5b. 4D Tongue Plot.npz',
    completeness=survey_completeness_4D,
    mass_centers=mass_centers,
    sma_centers=sma_centers,
    ecc_centers=ecc_centers,
    mass_edges=mass_grid,
    sma_edges=sma_grid,
    ecc_edges=ecc_grid
    )

    # Modify marginalization
    beta_weights = beta.pdf(ecc_centers, a=0.867, b=3.03)
    beta_weights = beta_weights / np.sum(beta_weights)

    # Marginalize over eccentricity for plotting
    marginalized_completeness = np.average(survey_completeness, axis=2, weights = beta_weights)

    #marginalized_completeness = np.mean(survey_completeness, axis=2)

    # Create figure and axis (kept the same)
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ["white", "yellow", "red", "blue"]
    ylrdblu_cmap = LinearSegmentedColormap.from_list("YlRdBlu", colors, N=1024)

    # Create the heatmap (kept the same)
    im = ax.imshow(marginalized_completeness,
                    origin='lower',
                    aspect='auto',
                    extent=[np.log10(sma_min), np.log10(sma_max), 
                            np.log10(mass_min), np.log10(mass_max)],
                    cmap=ylrdblu_cmap)
    
    # Apply Gaussian smoothing to the completeness data
    smoothing_sigma = 1.0  # Adjust this value to control smoothing amount
    smoothed_completeness = gaussian_filter(marginalized_completeness, sigma=smoothing_sigma)
    
    # Add contours using the smoothed data
    levels = [20, 40, 60, 80, 90, 95]  # Contour levels
    CS = ax.contour(np.linspace(np.log10(sma_min), np.log10(sma_max), survey_completeness.shape[1]),
                    np.linspace(np.log10(mass_min), np.log10(mass_max), survey_completeness.shape[0]),
                    smoothed_completeness,
                    levels=levels,
                    colors='black',
                    linewidths=2)  # Increased line width

    # Label contours with custom format
    fmt = {}
    for level in levels:
        fmt[level] = f'{level} stars'
        
    ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=16)  # Increased font size


    # Add colorbar (modified label to reflect new units)
    cbar = plt.colorbar(im)
    cbar.set_label('Number of Stars', rotation=270, labelpad=15)

    # Set axis labels and ticks (kept the same)
    ax.set_xlabel('Semi-major Axis (AU)')
    ax.set_ylabel('Planet Mass (M$_\oplus$)')

    # Set custom tick positions (kept the same)
    sma_ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10]
    mass_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 50]

    # Convert to log10 for plotting (kept the same)
    sma_tick_positions = np.log10(sma_ticks)
    mass_tick_positions = np.log10(mass_ticks)

    ax.set_xticks(sma_tick_positions)
    ax.set_yticks(mass_tick_positions)

    # Set tick labels (kept the same)
    ax.set_xticklabels(sma_ticks)
    ax.set_yticklabels(mass_ticks)

    plt.tight_layout()
    plt.savefig(curr_dir / '5b. Tongue Plot.png', dpi=300, bbox_inches='tight')
    plt.show()

        




