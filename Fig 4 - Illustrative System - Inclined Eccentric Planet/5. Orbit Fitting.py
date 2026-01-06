"""
Orbit fitting and plotting the results (Fig 3 in Abbas et al. 2026) (Part 5 of 5)
---------------------------------------------------
Orbit fits all the "detected" astrometric data from the previous observing logs,
and recreates Fig 2a and 2b.

Inputs:
    3. Observing Log.csv              # From Part 3 (Same directory)

    OR

    4. Observing Log.csv              # From Part 4 (Same directory)

Outputs:
    fig4a_OrbitalFit_SC.png           # If "3. Observing Log.csv" is used

    OR

    fig4b_OrbitalFit_AC.png           # If "4. Observing Log.csv" is used
"""

# ------------------------ Scientific Assumptions --------------------------- #
# - Orbits are fit only to detections from the observing logs (for the uniform and adaptive cadence)
# - Planet is considered detected if:
# -     (a) 2D star-planet sep is within the working angles
# -     (b) star-planet contrast is above the defined contrast floor
# - Planet is considered to be in the HZ if its orbit is fully within HZ limits from Kopparappu et. al. 2013 
# - HZ Confidence = Fraction of orbital posterior samples fully in the HZ / Total number of posterior samples * 100
# --------------------------------------------------------------------------- #


import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

# orbituary and PlotStyle are local libraries that will need to be installed
# Installation intructions can be found in REQUIREMENTS.md in the root directory
from orbituary.orbituary_interface import fit_orbit
from orbituary.solve_orbit import solve_orbit_XY
from PlotStyle import plotStyle

plotStyle()
mpl.rcParams.update({
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# >>>>>>>>>>>>>>>>>>>>>>> USER-TUNABLE VARIABLES <<<<<<<<<<<<<<<<<<<<<<<< #
OBS_LOG_FILE  = "3. Observing Log.csv"   # 3 = Uniform Cadence, 4 = Adaptive Cadence

PLANET_ID     = 'HD 72905_0'             # Can be found in the corresponding Observation Log
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #

if OBS_LOG_FILE[0] == '3':
    FIG_TITLE = "Uniform Cadence"
    SAVE_PATH = "fig4a_OrbitalFit_SC"

elif OBS_LOG_FILE[0] == '4':
    FIG_TITLE = "Adaptive Cadence"
    SAVE_PATH = "fig4b_OrbitalFit_AC"



def plot_progressive_fits_with_HZ(orbit_dfs, observation_times, sep_obs, pa_obs, dStar, L, visibility_flags, n_orbits=100):
    """
    Plot the progression of orbital fits as epochs are added.
    
    Parameters
    ----------
    orbit_dfs : list of pd.DataFrame
        List of DataFrames containing orbital parameters for each epoch
    observation_times : array-like
        Array of observation times
    sep_obs : array-like
        Array of observed separations
    pa_obs : array-like
        Array of observed position angles
    dStar : float
        Distance to the system in parsecs
    L : float
        Luminosity of the star in Solar luminosity
    visibility_flags : array-like
        Array of booleans with size = n_epochs that stores detection status (True if detected, and False otherwise)
    n_orbits : int, optional
        Number of orbits to plot per subplot (default: 100)

        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the subplots
    """

    # HZ for the given star from Kopparappu 2013
    hz_inner = np.sqrt(L / 1.78)
    hz_outer = np.sqrt(L / 0.32)
    
    # ----------------  Figure Set-up  ----------------- # 
    # Rows and columns in the subplots based on the number of observations
    n_epochs = len(orbit_dfs)
    n_rows   = 2
    n_cols   = (n_epochs + 1) // 2  # Rounds up for odd numbers

    # Size and spacing of the subplot panels
    panel_size = 3.2  
    fig, axs   = plt.subplots(n_rows, n_cols, figsize=(panel_size*n_cols, panel_size*n_rows),sharex=True, sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0}, constrained_layout = False)
    
    # Tick and label placement for each subplot
    axs = axs.flatten()
    for a in axs:
        # Only show labels for the outer subplots
        a.label_outer()   
        # Make the ticks point inward instead of the default out
        a.tick_params(which='both', direction='in', top=True, right=True, bottom=True, left=True)
    # -------------------------------------------------- #
    

    # Set up time array for plotting orbits
    time_start = min(observation_times) 
    time_end   = max(observation_times) 
    time       = np.linspace(time_start, time_end, 500)

    # ----------------  Subplot for each epoch  ----------------- # 
    for i in range(n_epochs):

        ax = axs[i]

        # Variable to count how many of the fitted orbits at an epoch are in the HZ
        hz_count = 0
        
        orbits = orbit_dfs[i].iloc[:n_orbits, :7].values
        total_count = orbits.shape[0]

        # Loop through each orbit, classify as HZ/non-HZ, and plot in the according colour
        for orbit in orbits:

            # ------ HZ Classification ------ #
            sma, ecc  = orbit[0], orbit[1]
            peri, apo = sma * (1 - ecc), sma * (1 + ecc)

            # HZ Check
            in_hz = (peri > hz_inner) and (apo < hz_outer)

            # Increment HZ counter if it is in the HZ (for HZ confidence calculations. See Sec 3.3, Abbas et al. 2026)
            if in_hz:
                hz_count += 1

            # Use the time array defined earlier to trace the orbit out in the sky plane
            x, y  = solve_orbit_XY(*orbit, time)

            # Colour HZ orbits in green; non-HZ in grey
            color = '#67C700' if in_hz else 'grey'
            alpha = 0.3
            ax.plot(x, y, color=color, lw=0.5, alpha=alpha, zorder=1)
            # ------------------------------- #
        
        # Compute HZ confidence 
        hz_conf = (100.0 * hz_count / total_count) 
        hz_text = f"HZ: {hz_conf:.1f}%"

        # - Plot the current and all previous obervations of the planet (detections AND non-detections)
        # - Detection = Solid, Non-Detection = Faded
        # - Current epoch = Red, Past epochs = Blue

        for j in range(i + 1):
            # Convert sep and PA into a point on the Cartesian sky plane
            x_obs = -sep_obs[j] * np.sin(np.radians(pa_obs[j])) * dStar
            y_obs =  sep_obs[j] * np.cos(np.radians(pa_obs[j])) * dStar

            # Current epoch: Red (alpha = 1 if visible, alpha = 0.35 otherwise)
            if j == i:
                ax.scatter(x_obs, y_obs, color = 'red', alpha=1.0 if visibility_flags[j] else 0.35, edgecolor = 'black', zorder = 5)
            
            # Past epochs: blue if visible, faded blue if not
            else:
                ax.scatter(x_obs, y_obs, color = 'blue', alpha=1.0 if visibility_flags[j] else 0.35, edgecolor = 'black', zorder = 4)


        # ----- Subplot formatting ----- #
        ax.set_xlim([-2.2, 2.2])
        ax.set_ylim([-2.2, 2.2])
        ax.set_xticks(np.arange(-2, 2.1, 1.0))
        ax.set_yticks(np.arange(-2, 2.1, 1.0))
        ax.text(0.03, 0.97, f"Epoch {i+1}: {'Detection' if visibility_flags[i] else 'Non-detection'} ({hz_text})",
            transform=ax.transAxes, ha='left', va='top', # transform = ax.transAxes --> (0.03, 0.97) refers to units scaled to the subplot NOT absolute units
            fontsize=14, weight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=2), zorder=10
        )
        # ------------------------------ #
    # ----------------------------------------------------------- #
    
    # Shared X and Y axis labels
    fig.text(0.5, 0.02, r"$x_{\rm sky}$ (AU)", ha='center', va='center', fontsize=18)
    fig.text(0.02, 0.5, r"$y_{\rm sky}$ (AU)", ha='center', va='center', rotation=90, fontsize=18)
    
    fig.suptitle(FIG_TITLE, fontsize=22, y=0.995)

    # Tight mosaic: no gaps; keep a hair of margin for shared labels
    fig.subplots_adjust(left=0.06, right=0.995, top=0.94, bottom=0.08, wspace=0, hspace=0)

    return fig

# ---------------------------------- I/O ------------------------------------ #
curr_dir = Path(__file__).resolve().parent

# Load the observing log
obs_df = pd.read_csv(curr_dir / OBS_LOG_FILE)
# --------------------------------------------------------------------------- #

# Pick a planet to fit
target_planet_id = PLANET_ID

# Find the observing details for the planets (detections and non-detections)
planet_obs     = obs_df[(obs_df["PlanetID"] == target_planet_id)]
# Subset for detections only
det_planet_obs = obs_df[(obs_df["PlanetID"] == target_planet_id) & (obs_df["DetStatus"] == 1)]

# Sort by observation time
det_planet_obs = det_planet_obs.sort_values("LastObs")

# Astrometric data of detections for orbit fitting
det_times           = np.array(det_planet_obs["LastObs"])
det_separations     = np.array(det_planet_obs["Sep"])
det_position_angles = np.array(det_planet_obs["PA"])

# All astrometric data (detections and non) for plotting
times           = np.array(planet_obs["LastObs"])
separations     = np.array(planet_obs["Sep"])
position_angles = np.array(planet_obs["PA"])

# Host star properties for orbit fitting / HZ limits
# Read off the first row since they are repeated across rows
stellar_mass = planet_obs["M_sol"].iloc[0]
distance_pc  = planet_obs["d_pc"].iloc[0]
L            = planet_obs["L_sol"].iloc[0]

# Orbit fit the detections
results = fit_orbit(
    det_times, det_separations, det_position_angles, stellar_mass, distance_pc, 
    mcmc_steps=30000,
    mcmc_walkers=500,
    mcmc_burnin=2000,
    all_epochs=True, 
    progress_bar=False
)

# ----------------  Assign orbit fits to each epoch for plotting  --------------- #
# - We just orbit fit detections (to simulate real-world operations)
# - But we plot epochs of both detections and non-detections
# - If the current epoch is a detection, use the latest orbit fit
# - Otherwise, use the most recent orbit fit (since a "non-detection" cannot orbital posteriors)

# List of orbital posteriors at each each (both detection and non-detection epochs)
orbit_dfs = []

# List of booleans to track if the planet was detected at each epoch (size = n_epochs)
# Useful for plotting
visibility_flags = []

det_idx = 0            # Variabe to iterate through the detected epochs, since they are the only ones with orbit fits
last_valid_fit = None  # Orbital posteriors for the last detection. Initialised to None

# Loop through all epochs the planet was observed (both detections and non-detections)
for t in times:
    # If the epoch is detected
    if det_idx < len(det_times) and np.isclose(t, det_times[det_idx]):
        # Use the new fit
        orbit_dfs.append(results[det_idx])
        last_valid_fit = results[det_idx]
        visibility_flags.append(True)    # Set the visibility flag for this epoch as true 
        det_idx += 1                     
    else:
        # No new detection: reuse previous orbits
        orbit_dfs.append(last_valid_fit)
        visibility_flags.append(False)   # Set the visibility flag for this epoch as false
# ------------------------------------------------------------------------------- #

# Plotting the astrometric data and orbits at each epoch
fig = plot_progressive_fits_with_HZ(
    orbit_dfs,
    times,
    separations,
    position_angles,
    distance_pc,
    L,
    visibility_flags,
    n_orbits=2000  # Or however many you want to show
)

fig.savefig(curr_dir / SAVE_PATH, dpi=300, bbox_inches='tight')