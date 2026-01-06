import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import os

from orbituary.ofti_algorithms import fit_single_epoch, fit_two_epochs
from orbituary.solve_orbit import solve_orbit_XY, solve_orbit
from orbituary.mcmc_solver import MCMC

def plotStyle():

    plt.rcParams['xtick.color'] = "323034"
    plt.rcParams['ytick.color'] = "323034"
    plt.rcParams['text.color'] = "323034"
    plt.rcParams['lines.markeredgecolor'] = "black"
    plt.rcParams['patch.facecolor'] = "bc80bd"
    plt.rcParams['patch.force_edgecolor'] = True
    plt.rcParams['patch.linewidth'] = 0.8
    plt.rcParams['scatter.edgecolors'] = "black"
    plt.rcParams['grid.color'] = "b1afb5"
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.title_fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.prop_cycle'] = "(cycler('color', ['1f77b4', 'fdb462', 'b3de69', 'fb8072', 'bc80bd', 'fccde5', '8dd3c7', 'ffed6f', 'bebada', '80b1d3', 'ccebc5', 'd9d9d9']))"
    plt.rcParams['mathtext.fontset'] = "stix"
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = ['Calibri']
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.fontsize'] = 13
    plt.rcParams['legend.edgecolor'] = "black"
    plt.rcParams['legend.borderpad'] = 0.2
    plt.rcParams['legend.columnspacing'] = 1.5
    plt.rcParams['legend.labelspacing'] = 0.4
    plt.rcParams['text.usetex'] = False
    plt.rcParams['axes.labelsize'] = 17
    plt.rcParams['axes.titlelocation'] = "center"
    plt.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams['axes.autolimit_mode'] = "round_numbers"
    plt.rcParams['axes.labelpad'] = 3
    plt.rcParams['axes.formatter.limits'] = (-4, 4)
    plt.rcParams['axes.labelcolor'] = "black"
    plt.rcParams['axes.edgecolor'] = "black"
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.spines.right'] = True
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.top'] = True
    plt.rcParams['figure.titlesize'] = 18
    #plt.rcParams['figure.autolayout'] = True
    plt.rcParams['figure.dpi'] = 300
    plt.gca().tick_params(axis='both', which='major', width=2)
    plt.gca().tick_params(axis='both', which='minor', width=2)

plotStyle()

def analyze_fit_quality(accepted_orbits, observation_times, sep_obs, pa_obs, dStar, epoch_idx=None):
    """
    Analyze the quality of orbital fits across all epochs.
    
    Parameters
    ----------
    accepted_orbits : array-like
        Array of accepted orbital parameters [sma, e, inc, aop, pan, epp, P]
    observation_times : array-like
        Array of observation times
    sep_obs : array-like
        Array of observed separations
    pa_obs : array-like
        Array of observed position angles
    dStar : float
        Distance to system in parsecs
    epoch_idx : int, optional
        If provided, analyze only this specific epoch
        
    Returns
    -------
    dict
        Dictionary containing fit statistics per epoch
    """
    
    stats = {}
    
    # If epoch_idx is provided, only analyze that epoch
    epochs_to_analyze = [epoch_idx] if epoch_idx is not None else range(len(observation_times))
    
    for epoch in epochs_to_analyze:
        t = observation_times[epoch]
        sep_true = sep_obs[epoch]
        pa_true = pa_obs[epoch]
        
        # Calculate predictions for all orbits at this epoch
        sep_pred = []
        pa_pred = []
        for orbit in accepted_orbits:
            s, p, _, _ = solve_orbit(*orbit, t, dStar)
            sep_pred.append(s)
            pa_pred.append(p)
            
        sep_pred = np.array(sep_pred)
        pa_pred = np.array(pa_pred)
        
        # Handle periodic boundary for PA
        pa_diff = np.abs((pa_pred - pa_true + 180) % 360 - 180)
        
        # Calculate statistics
        stats[epoch] = {
            'sep_mean_error': np.mean(np.abs(sep_pred - sep_true)),
            'sep_std_error': np.std(np.abs(sep_pred - sep_true)),
            'pa_mean_error': np.mean(pa_diff),
            'pa_std_error': np.std(pa_diff),
            'max_sep_error': np.max(np.abs(sep_pred - sep_true)),
            'max_pa_error': np.max(pa_diff)
        }
    
    return stats

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

# Get path relative to current file
current_dir = os.path.dirname(os.path.abspath(__file__))
obs_log_path = os.path.join(current_dir, 'data', 'observing_log.csv')
stars_data_path = os.path.join(current_dir, 'data', 'stars_with_planets.csv')

dfObs = pd.read_csv(obs_log_path)
observation_times = np.array(dfObs['LastObs'])  # Example observation times (years)
sep_obs = np.array(dfObs['Sep'])  # Example separations (arcsec)
pa_obs = np.array(dfObs['PA'])  # Example position angles (degrees)

#observation_times = [2035, 2035.5]
#sep_obs = [1, 1]
#pa_obs = [270, 90]

# Circular edge-on orbit
#observation_times = np.array([2035.0, 2035.25, 2035.5, 2035.75, 2036, 2036.25])  # Example observation times (years)
#sep_obs = np.array([1, 0.5, -1, 1, ])  # Oscillating separation
#pa_obs = np.array([90, 270, 90])  # PA flips between East and West

dfData = pd.read_csv(stars_data_path)
Mstar = np.array(dfData['M(sol)'])[0]  # Stellar mass (solar masses)
dStar = np.array(dfData['d(pc)'])[0]  # Distance to the system (parsecs)
num_trials = 1000  # Number of trial orbits per batch

switch_mcmc = 0 

# Create a figure and define the grid with 2 rows and 5 columns
fig = plt.figure(figsize = (28,12))
gs = gridspec.GridSpec(2, 5, width_ratios=[3,1,1,1,1], top = 1, bottom = 0.55)
gs_bottom = gridspec.GridSpec(2, 4, top = 0.45, bottom = 0.05)

HZ_inner, HZ_outer = HZ(1)
ax_left = fig.add_subplot(gs[:2,0])

ax_right = []
for j in range(6):
    if j < 3:
        ax = fig.add_subplot(gs[0,j+1])
    else:
        ax = fig.add_subplot(gs[1,j-2])
    ax_right.append(ax)


x_values = []
y_hz_values = []
y_nonhz_values = []

for i in range(len(observation_times)):

    print(i)

    HZ_orbits = []
    non_HZ_orbits = []

    t_obs_curr = observation_times[:i+1]
    sep_obs_curr = sep_obs[:i+1]
    pa_obs_curr = pa_obs[:i+1]

    if i < 2:
        if len(t_obs_curr) == 1:
            accepted_orbits = fit_single_epoch(50000, t_obs_curr[0], sep_obs_curr[0], pa_obs_curr[0], Mstar, dStar)

        else:
            accepted_orbits, switch_mcmc = fit_two_epochs(1000, num_trials, t_obs_curr, sep_obs_curr, pa_obs_curr, Mstar, dStar, old_orbits)
    if i >= 2:
        print("Switching to MCMC")

        accepted_orbits = MCMC(1000, sep_obs_curr, pa_obs_curr, t_obs_curr, dStar, Mstar, accepted_orbits)

    accepted_orbits = np.array(accepted_orbits)

    print(f"\nDiagnostic Analysis for Epoch {i}:")
    stats = analyze_fit_quality(accepted_orbits, t_obs_curr, sep_obs_curr, pa_obs_curr, dStar)
    for epoch, epoch_stats in stats.items():
        print(f"\nEpoch {epoch}:")
        print(f"Separation errors (arcsec) - Mean: {epoch_stats['sep_mean_error']:.3f}, "
              f"Std: {epoch_stats['sep_std_error']:.3f}, Max: {epoch_stats['max_sep_error']:.3f}")
        print(f"Position Angle errors (deg) - Mean: {epoch_stats['pa_mean_error']:.3f}, "
              f"Std: {epoch_stats['pa_std_error']:.3f}, Max: {epoch_stats['max_pa_error']:.3f}")


    old_orbits = accepted_orbits

    for orbit in accepted_orbits:
        if orbit[0] * (1 - orbit[1]) > HZ_inner and orbit[0] * (1 + orbit[1]) < HZ_outer:
            HZ_orbits.append(orbit)
        else:
            non_HZ_orbits.append(orbit)

    x_values.append(i+1)
    hz_fraction = len(HZ_orbits)/(len(HZ_orbits) + len(non_HZ_orbits))
    nonhz_fraction = len(non_HZ_orbits)/(len(HZ_orbits) + len(non_HZ_orbits))
    y_hz_values.append(hz_fraction)
    y_nonhz_values.append(nonhz_fraction)

    ax_left.scatter(i+1, hz_fraction, color='green', edgecolor = 'black', zorder = 3)
    ax_left.scatter(i+1, nonhz_fraction, color='red', edgecolor = 'black', zorder = 3)
            

    # Plot the accepted orbits in the sky plane
    time = np.linspace(2034,2038,1000)
    for orbit in accepted_orbits[:100]:
        x_sky, y_sky = solve_orbit_XY(*orbit, time)
        if orbit[0] * (1 - orbit[1]) > HZ_inner and orbit[0] * (1 + orbit[1]) < HZ_outer:
            ax_right[i].plot(x_sky, y_sky, color = "green", lw = 0.5)
        else:
            ax_right[i].plot(x_sky, y_sky, color = "grey", lw = 0.5)
        ax_right[i].set_xlim([-2,2])
        ax_right[i].set_ylim([-2,2])
        ax_right[i].set_xlabel("$x_{sky}$ (AU)", fontsize = 16)
        ax_right[i].set_ylabel("$y_{sky}$ (AU)", fontsize = 16)

    # Add observed positions for comparison
    for j, (sep, pa) in enumerate(zip(sep_obs_curr, pa_obs_curr)):
        x_obs = -sep * np.sin(np.radians(pa)) * dStar
        y_obs = sep * np.cos(np.radians(pa)) * dStar
        ax_right[i].scatter(x_obs, y_obs, color="red", zorder = 3)

ax_left.axhline(1, color = "grey", ls = "--", zorder = -1)

ax_left.set_xlabel("Epoch", fontsize = 16)
ax_left.set_ylabel("Fraction of orbits", fontsize = 16)

ax_left.plot(x_values, y_hz_values, color='green', ls="-")
ax_left.plot(x_values, y_nonhz_values, color='red', ls="-")

ax_left.set_xlim([0.5,6.5])
ax_left.set_ylim([0,1.25])
ax_left.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax_left.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax_left.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax_left.tick_params(axis='both', which='major', length=10, width = 2)
ax_left.tick_params(axis='both', which='minor', length=5)

ax_left.legend(loc = "upper left")

fig_path = os.path.join(current_dir, 'TestFit.png')
plt.savefig(fig_path, bbox_inches = 'tight', dpi = 300)

# Run OFTI
#accepted_orbits, min_chisq = ofti_serial(num_orbits, num_trials, observation_times, sep_obs, pa_obs, Mstar, dStar)


