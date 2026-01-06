import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker

from PlotStyle import plotStyle
plotStyle()

def compute_median_cdf(df, column, simulation_col='simulation_id', n_points=100):
    """
    Compute a 'median' CDF by finding, for each quantile p in [0,1],
    the median of the per-simulation x-values that achieve that quantile.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing simulation data.
    column : str
        Column name (e.g. 'planet_rad') for which to compute.
    simulation_col : str
        Name of column identifying which simulation each row belongs to.
    n_points : int
        How many quantile points to sample in [0,1].
    
    Returns:
    --------
    median_x : np.ndarray of shape (n_points,)
        Median x-value across simulations for each quantile p.
    p_values : np.ndarray of shape (n_points,)
        The quantile (CDF) values in [0,1] at which we computed median_x.
    """
    # Define a set of quantiles, e.g. 0, 0.01, 0.02, ..., 1
    p_values = np.linspace(0, 1, n_points)
    
    # Collect the x-values for these quantiles from each simulation
    x_arrays = []
    
    for sim_id, group in df.groupby(simulation_col):
        # Sort the data in this simulation
        values = np.sort(group[column].values)
        if len(values) == 0:
            continue
        
        # Get x-values for each quantile p
        x_for_this_sim = np.quantile(values, p_values)
        x_arrays.append(x_for_this_sim)
    
    # Convert to array of shape (num_sims, n_points)
    x_arrays = np.array(x_arrays)
    
    # Take the median across the "simulation" axis
    median_x = np.median(x_arrays, axis = 0)

    lower_bound = np.percentile(x_arrays, 16, axis = 0)
    upper_bound = np.percentile(x_arrays, 84, axis = 0)
    
    return median_x, p_values, lower_bound, upper_bound

# Get the path to the current directory
curr_dir = Path(__file__).resolve().parent

planets_df = pd.read_csv(curr_dir / "6d. Detected Planets.csv")

# Actual data
observed_per = np.sort(planets_df['P_yr'])
observed_per_cdf = np.arange(1, len(observed_per)+1) / len(observed_per)
observed_per =  np.concatenate(([0], observed_per))
observed_per_cdf = np.concatenate(([0], observed_per_cdf))

observed_rad = np.sort(planets_df["Rp_REarth"])
observed_rad_cdf = np.arange(1, len(observed_rad)+1) / len(observed_rad)
observed_rad =  np.concatenate(([0], observed_rad))
observed_rad_cdf = np.concatenate(([0], observed_rad_cdf))

observed_star_mass = np.sort(planets_df["M_sol"])
observed_star_mass_cdf = np.arange(1, len(observed_star_mass)+1) / len(observed_star_mass)
observed_star_mass =  np.concatenate(([0], observed_star_mass))
observed_star_mass_cdf = np.concatenate(([0], observed_star_mass_cdf))

observed_ecc = np.sort(planets_df["ecc"])
observed_ecc_cdf = np.arange(1, len(observed_ecc)+1) / len(observed_ecc)
observed_ecc =  np.concatenate(([0], observed_ecc))
observed_ecc_cdf = np.concatenate(([0], observed_ecc_cdf))

# Load the mock survey data
mock_survey = pd.read_csv(os.path.join(curr_dir, '6g. Mock Survey.csv'))

# Group the data by simulation_id
grouped = mock_survey.groupby('simulation_id')

# Initialize the figure and subplots
fig, axs = plt.subplots(2, 3, figsize=(16.5, 10), gridspec_kw={'wspace': 0.24, 'hspace': 0.24})

# Flatten the axs array for easier indexing
axs = axs.flatten()

# Initialize lists to store the cumulative distributions
planet_rad_cumulative = []
per_cumulative = []
star_mass_cumulative = []
ecc_cumulative = []
num_planets = []
num_HZ_planets = []

rad_grid, median_cdf_rad, rad_lower, rad_upper = compute_median_cdf(mock_survey, 'planet_rad')
star_mass_grid, median_cdf_star, star_lower, star_upper = compute_median_cdf(mock_survey, 'star_mass')
per_grid, median_cdf_per, per_lower, per_upper = compute_median_cdf(mock_survey, 'per')
ecc_grid, median_cdf_ecc, ecc_lower, ecc_upper = compute_median_cdf(mock_survey, 'ecc')

# Calculate cumulative distributions for each group
for name, group in grouped:
    planet_rad_cumulative.append(np.sort(group['planet_rad']))
    per_cumulative.append(np.sort(group['per']))
    star_mass_cumulative.append(np.sort(group['star_mass']))
    ecc_cumulative.append(np.sort(group['ecc']))
    num_planets.append(len(group))
    num_HZ_planets.append(len(group[(group['HZ'] == True) & (group['planet_rad']>=0.8) & (group['planet_rad'] <= 1.5)]))

# Plot cumulative distribution of planet rad
for i, data in enumerate(planet_rad_cumulative[:1000]):
    x_vals = np.concatenate(([0], data))
    y_vals = np.concatenate(([0], np.arange(1, len(data) + 1) / len(data)))
    axs[0].step(x_vals, y_vals, where='post', alpha=0.1, color='#7FC5FF', lw=0.3)

rad_grid = np.concatenate(([0], rad_grid))
rad_lower = np.concatenate(([0], rad_lower))
rad_upper = np.concatenate(([0], rad_upper))
median_cdf_rad = np.concatenate(([0], median_cdf_rad))
axs[0].step(rad_grid, median_cdf_rad, where='post', color = 'blue', lw = 3)
axs[0].step(rad_lower, median_cdf_rad, where='post', color = 'blue', lw = 2)
axs[0].step(rad_upper, median_cdf_rad, where='post', color = 'blue', lw = 2)
axs[0].step(observed_rad, observed_rad_cdf, where = 'post', color = 'red', lw = 3)

axs[0].set_xlabel('Radius ($R_\oplus$)')
axs[0].set_ylabel('Cumulative Probability')
axs[0].set_xlim([0.5, 5])
axs[0].set_ylim([0,1])
axs[0].set_xscale('log')

# Major ticks at each power of 10
axs[0].xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=10))
# Minor ticks at 2, 3, ..., 9 within each decade
axs[0].xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=100))
axs[0].xaxis.set_minor_formatter(ticker.NullFormatter())

axs[0].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
axs[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))


# Plot cumulative distribution of separation
for i, data in enumerate(per_cumulative[:1000]):
    x_vals = np.concatenate(([0], data))
    y_vals = np.concatenate(([0], np.arange(1, len(data) + 1) / len(data)))
    axs[1].step(x_vals, y_vals, where='post', alpha=0.1, color='#7FC5FF', lw=0.3)

per_grid = np.concatenate(([0], per_grid))
median_cdf_per = np.concatenate(([0], median_cdf_per))
per_lower = np.concatenate(([0], per_lower))
per_upper = np.concatenate(([0], per_upper))
axs[1].step(per_grid, median_cdf_per, where='post', color = 'blue', lw = 3)
axs[1].step(per_lower, median_cdf_per, where='post', color = 'blue', lw = 2)
axs[1].step(per_upper, median_cdf_per, where='post', color = 'blue', lw = 2)
axs[1].step(observed_per, observed_per_cdf, where = 'post', color = 'red', lw = 3)
axs[1].set_xlabel('Period (yr)')
axs[1].set_ylabel('Cumulative Probability')
axs[1].set_xlim([0.1,50])
axs[1].set_ylim([0,1])
axs[1].set_xscale('log')

# Major ticks at each power of 10
axs[1].xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=10))
# Minor ticks at 2, 3, ..., 9 within each decade
axs[1].xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=100))
axs[1].xaxis.set_minor_formatter(ticker.NullFormatter())

axs[1].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
axs[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

# Plot cumulative distribution of star mass
for i, data in enumerate(star_mass_cumulative[:1000]):
    x_vals = np.concatenate(([0], data))
    y_vals = np.concatenate(([0], np.arange(1, len(data) + 1) / len(data)))
    axs[2].step(x_vals, y_vals, where='post', alpha=0.1, color='#7FC5FF', lw=0.3)

star_mass_grid = np.concatenate(([0], star_mass_grid))
median_cdf_star = np.concatenate(([0], median_cdf_star))
star_lower = np.concatenate(([0], star_lower))
star_upper = np.concatenate(([0], star_upper))

axs[2].step(star_mass_grid, median_cdf_star, where = 'post', color = 'blue', lw = 3)
axs[2].step(star_lower, median_cdf_star, where = 'post', color = 'blue', lw = 2)
axs[2].step(star_upper, median_cdf_star, where = 'post', color = 'blue', lw = 2)
axs[2].step(observed_star_mass, observed_star_mass_cdf, where = 'post', color = 'red', lw = 3)
axs[2].set_xlabel('Stellar Host Mass ($M_{\odot}$)')
axs[2].set_ylabel('Cumulative Probability')
axs[2].set_xlim([0.2, 1.5])
axs[2].set_ylim([0,1])

axs[2].xaxis.set_major_locator(ticker.MultipleLocator(0.2))
axs[2].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

axs[2].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
axs[2].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

# Plot cumulative distribution of eccentricity
for i, data in enumerate(ecc_cumulative[:1000]):
    x_vals = np.concatenate(([0], data))
    y_vals = np.concatenate(([0], np.arange(1, len(data) + 1) / len(data)))
    axs[3].step(x_vals, y_vals, where='post', alpha=0.1, color='#7FC5FF', lw=0.3)

ecc_grid = np.concatenate(([0], ecc_grid))
median_cdf_ecc = np.concatenate(([0], median_cdf_ecc))
ecc_lower = np.concatenate(([0], ecc_lower))
ecc_upper = np.concatenate(([0], ecc_upper))
axs[3].step(ecc_grid, median_cdf_ecc, where='post', color = 'blue', lw = 3)
axs[3].step(ecc_lower, median_cdf_ecc, where='post', color = 'blue', lw = 2)
axs[3].step(ecc_upper, median_cdf_ecc, where='post', color = 'blue', lw = 2)
axs[3].step(observed_ecc, observed_ecc_cdf, where = 'post', color = 'red', lw = 3)
axs[3].set_xlabel('Eccentricity')
axs[3].set_ylabel('Cumulative Probability')
axs[3].set_xlim([0,1])
axs[3].set_ylim([0,1])

axs[3].xaxis.set_major_locator(ticker.MultipleLocator(0.2))
axs[3].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

axs[3].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
axs[3].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

# Plot frequency distribution of the number of planets per run
median = np.median(num_planets)
lower, upper = np.percentile(num_planets, [16, 84])

bins = np.arange(0, max(num_planets) + 2, 5)
axs[4].hist(num_planets, bins=bins, edgecolor='#7FC5FF', histtype='step', lw = 3)
axs[4].axvline(x=median, color='blue', lw=6)
axs[4].axvline(x=lower, color='blue', lw=2)
axs[4].axvline(x=upper, color='blue', lw=2)
axs[4].axvline(x=len(planets_df), color='red', lw=3)
axs[4].set_xlabel('Number of Detected Companions')
axs[4].set_ylabel('Frequency')
axs[4].set_xlim([80,200])

axs[4].xaxis.set_major_locator(ticker.MultipleLocator(20))
axs[4].xaxis.set_minor_locator(ticker.MultipleLocator(10))

axs[4].yaxis.set_major_locator(ticker.MultipleLocator(50))
axs[4].yaxis.set_minor_locator(ticker.MultipleLocator(25))

# Plot frequency distribution of the number of HZ planets per run
median = np.median(num_HZ_planets)
lower, upper = np.percentile(num_HZ_planets, [16, 84])

true_HZ_planets = planets_df[(planets_df['HZ'] == True) & (planets_df['Rp_REarth'] >= 0.8) & (planets_df['Rp_REarth'] <= 1.5)]

bins = np.arange(0, max(num_HZ_planets) + 2, 1)
axs[5].hist(num_HZ_planets, bins=bins, edgecolor='#7FC5FF', histtype='step', lw = 3)
axs[5].axvline(x=median, color='blue', lw=6)
axs[5].axvline(x=lower, color='blue', lw=2)
axs[5].axvline(x=upper, color='blue', lw=2)
axs[5].axvline(x=len(true_HZ_planets), color='red', lw=3)
axs[5].set_xlabel('Number of Exo-Earth Companions')
axs[5].set_ylabel('Frequency')
axs[5].set_xlim([0,15])

axs[5].xaxis.set_major_locator(ticker.MultipleLocator(2))
axs[5].xaxis.set_minor_locator(ticker.MultipleLocator(1))

axs[5].yaxis.set_major_locator(ticker.MultipleLocator(50))
axs[5].yaxis.set_minor_locator(ticker.MultipleLocator(25))

plt.savefig(curr_dir / "6h. Survey Results.png", bbox_inches = 'tight', dpi = 300)
