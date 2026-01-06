import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.stats import beta
from pathlib import Path

from solve_orbit import solve_orbit_vectorized_full
from PlotStyle import plotStyle
from forecaster import optimized_mass_to_radius

plotStyle()

# Generate eccentricities limited by the habitable zone for Earthlike planets
def EarthlikeECC(sma, HZ_inner, HZ_outer, num_planets):
    """
    Generate eccentricities that keep the planet within the habitable zone.

    Parameters:
    semi_major_axes (float or np.array): Semi-major axis (or axes) in AU.
    HZ_inner (float or np.array): Inner edge of the habitable zone in AU.
    HZ_outer (float or np.array): Outer edge of the habitable zone in AU.

    Returns:
    eccentricities (float or np.array): Eccentricities that ensure the orbit remains within the habitable zone.
                                        If semi_major_axes is an array, returns an array of eccentricities.
    """
    # Calculate e_inner to avoid the planet going too close to the star
    # Ensure the periastron is still inside the HZ; HZ_inner = a(1 - e_inner)
    e_inner = 1 - HZ_inner / sma

    # Calculate e_outer to avoid the planet getting too far from the star
    # Ensure the periastron is still inside the HZ; HZ_outer = a(1 + e_outer)
    e_outer = HZ_outer / sma - 1

    # Pick the smaller of the two to be your max eccentricity
    e_max = np.minimum(e_inner, e_outer)

    # Generate random eccentricities
    alpha = 0.867  # Shape parameter
    Beta = 3.03    # Shape parameter
    
    return beta.rvs(alpha, Beta, size=num_planets) * e_max

# Generate eccentricities limited by the habitable zone for Earthlike planets
def BetaECC(num_planets, e_max = 0.99):
    """
    Generate eccentricities that keep the planet within the habitable zone.

    Parameters:
    semi_major_axes (float or np.array): Semi-major axis (or axes) in AU.
    HZ_inner (float or np.array): Inner edge of the habitable zone in AU.
    HZ_outer (float or np.array): Outer edge of the habitable zone in AU.

    Returns:
    eccentricities (float or np.array): Eccentricities that ensure the orbit remains within the habitable zone.
                                        If semi_major_axes is an array, returns an array of eccentricities.
    """

    # Generate random eccentricities
    alpha = 0.867  # Shape parameter
    Beta = 3.03    # Shape parameter
    
    return beta.rvs(alpha, Beta, size=num_planets) * e_max

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
                                 IWA=0.06, OWA=1, coronagraph_contrast=1e-11):
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

def generate_single_star_completeness(star_row, rad_centers, per_centers, ecc_centers, num_planets_per_cell=10000):
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
    # Create meshgrid of all mass-sma combinations
    rad_grid, per_grid, ecc_grid = np.meshgrid(rad_centers, per_centers, ecc_centers, indexing='ij')
    rad_flat = rad_grid.flatten()
    per_flat = per_grid.flatten()
    ecc_flat = ecc_grid.flatten()

    # Calculate total number of planets
    n_grid_points = len(rad_flat)
    total_planets = n_grid_points * num_planets_per_cell

    radii = np.repeat(rad_flat, num_planets_per_cell)
    P = np.repeat(per_flat, num_planets_per_cell)
    eccs = np.repeat(ecc_flat, num_planets_per_cell)

    # Generate orbital parameters for all planets at once
    inc = Inclinations(total_planets)
    aop, lan = OrbitalAngles(total_planets)
    
    # Compute sma (vectorized)
    smas = ((P**2) * star_row['M']) ** (1/3)

    # Generate epp
    epp = EpochOfPeriastronPassage(P, missionStart=2035, num_planets=total_planets)

    # Generate albedos
    albedos = EarthLikeAlbedo(total_planets)

    # Initialize completeness array
    completeness_cube = np.zeros((len(rad_centers), len(per_centers), len(ecc_centers)))

    # Define the two observation epochs (initial and 2 months later)
    mission_start = 2035.0
    n_epochs = 8
    cadence_months = 3

    epochs = mission_start + np.arange(n_epochs) * (cadence_months / 12)
    
    # Process in manageable batches to avoid memory issues
    batch_size = 100
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
        detection = np.zeros(end_idx - start_idx, dtype=bool)

        # Calculate orbital positions for both epochs
        for i, epoch in enumerate(epochs):
            sep, _, r3d, phase = solve_orbit_vectorized_full(
                params_batch,
                epoch,
                np.full(end_idx - start_idx, star_row['Dist'])
            )
            # Make sep, r3d, phase 2D arrays with shape (batch_size, 1)
            detection_epoch = check_detectability_vectorized(
                sep[:, None],
                r3d[:, None],
                phase[:, None],
                albedos[start_idx:end_idx],
                radii[start_idx:end_idx]
            )

            # Combine detections over all epochs (logical OR)
            detection |= detection_epoch

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
            rad_idx = grid_index // (len(per_centers) * len(ecc_centers))
            # Step 3
            remaining = grid_index % (len(per_centers) * len(ecc_centers))
            # Step 4
            per_idx = remaining // len(ecc_centers)
            # Step 5
            ecc_idx = remaining % len(ecc_centers)
            
            if (rad_idx < len(rad_centers) and 
                per_idx < len(per_centers) and 
                ecc_idx < len(ecc_centers)):
                completeness_cube[rad_idx, per_idx, ecc_idx] += detected

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
    print(f"\nProcessing star {i+1}: {star_dict['HDName']}", flush = True)  # Changed from star.HDName to star_dict['HDName']
    return generate_single_star_completeness(
        star_dict,
        rad_centers,
        sma_centers,
        ecc_centers,
        num_planets_per_cell
    )
    
def generate_survey_completeness(stars_df, rad_centers, per_centers, ecc_centers, num_planets_per_cell=10000):
    """
    Generate a survey-wide completeness map by parallel processing stars.
    """
    
    # Determine number of cores to use
    n_cores = 15
    print(f"Using {n_cores} CPU cores")
    
    # Create args list with just the necessary data from each star
    args_list = [(i, 
                  {
                      'HDName': star['HDName'],
                      'M': float(star['M']),  # Convert to native Python float
                      'Dist': float(star['Dist'])  # Convert to native Python float
                  }, 
                  rad_centers, 
                  per_centers, 
                  ecc_centers,
                  num_planets_per_cell) 
                 for i, star in stars_df.iterrows()]
    
    # Process stars in parallel
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        completeness_maps = list(executor.map(process_star, args_list))
    
    # Sum all the completeness maps
    survey_completeness = np.sum(completeness_maps, axis=0)

    # 4D completeness map
    survey_completeness_4d = np.stack(completeness_maps, axis=3)
    
    return survey_completeness, survey_completeness_4d

if __name__ == '__main__':

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    multiprocessing.set_start_method("spawn", force=True)

    # Directory Management
    curr_dir = Path(__file__).resolve().parent
    parent_dir = curr_dir.parent
    data_dir = parent_dir / "Data"

    stars = pd.read_csv(data_dir / 'HWO Stars.csv')
    print(len(stars))
    #stars = stars.sort_values('Dist', ascending=False).head(20)
    # Closest 20 stars
    #stars = stars.head(20)

    # -----------------------------  TONGUE PLOT SETUP  ---------------------------- #

    # Define the ranges (kept the same as original)
    radius_min = 0.01     # Minimum planet mass in Earth masses (M⊕)
    radius_max = 3.4      # Maximum planet mass in Earth masses (M⊕)
    period_min = 0.01     # Minimum semi-major axis in AU
    period_max = 40.0     # Maximum semi-major axis in AU
    ecc_min = 1e-4
    ecc_max = 0.99

    # Number of points in the grid 
    n_rad_points = 61  
    n_per_points = 101  
    n_ecc_points = 21

    # Generate logarithmic grids 
    rad_grid = np.logspace(np.log10(radius_min), np.log10(radius_max), n_rad_points)
    per_grid = np.logspace(np.log10(period_min), np.log10(period_max), n_per_points)
    ecc_grid = np.linspace(ecc_min, ecc_max, n_ecc_points)

    # Calculate cell centers
    rad_centers = np.sqrt(rad_grid[:-1] * rad_grid[1:])
    per_centers = np.sqrt(per_grid[:-1] * per_grid[1:])
    ecc_centers = 0.5 * (ecc_grid[:-1] + ecc_grid[1:])
    # ----------------------------- ------------------------------------------------ #

    # Generate the survey-wide completeness (new approach)
    survey_completeness, survey_completeness_4D = generate_survey_completeness(
        stars,
        rad_centers,
        per_centers,
        ecc_centers,
        num_planets_per_cell=10
    )

    np.savez(
    curr_dir / '8b. 4D Tongue Plot.npz',
    completeness=survey_completeness_4D,
    rad_centers=rad_centers,
    per_centers=per_centers,
    ecc_centers=ecc_centers,
    rad_edges=rad_grid,
    per_edges=per_grid,
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
                    extent=[np.log10(period_min), np.log10(period_max), 
                            np.log10(radius_min), np.log10(radius_max)],
                    cmap=ylrdblu_cmap)
    
    # Apply Gaussian smoothing to the completeness data
    smoothing_sigma = 1.0  # Adjust this value to control smoothing amount
    smoothed_completeness = gaussian_filter(marginalized_completeness, sigma=smoothing_sigma)
    
    # Add contours using the smoothed data
    levels = [20, 40, 60, 80, 90, 95]  # Contour levels
    CS = ax.contour(np.linspace(np.log10(period_min), np.log10(period_max), survey_completeness.shape[1]),
                    np.linspace(np.log10(radius_min), np.log10(radius_max), survey_completeness.shape[0]),
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
    ax.set_xlabel('Period (yr)')
    ax.set_ylabel('Planet Radius (R$_\oplus$)')

    # Set custom tick positions (kept the same)
    per_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40]
    rad_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3]

    # Convert to log10 for plotting (kept the same)
    per_tick_positions = np.log10(per_ticks)
    rad_tick_positions = np.log10(rad_ticks)

    ax.set_xticks(per_tick_positions)
    ax.set_yticks(rad_tick_positions)

    # Set tick labels (kept the same)
    ax.set_xticklabels(per_ticks)
    ax.set_yticklabels(rad_ticks)

    plt.tight_layout()
    plt.savefig(curr_dir / '8b. Tongue Plot.png', dpi=300, bbox_inches='tight')
    plt.show()

        




