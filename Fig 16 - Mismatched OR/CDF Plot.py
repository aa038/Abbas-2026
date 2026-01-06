"""
Mock Survey CDF Plots
-------------------------------------------
Script to plot the mock survey CDFs in planet mass and sma instead of radius and period

Input:
    Data/Part II - Demographics/Mismatched OR/5d. Detected Planets.csv      # List of detected planets (From Part 4)
    Data/Part II - Demographics/Mismatched OR/5g. Mock Survey.csv           # The full list of mock surveys (From Part 7)


Output:
    fig16_MismatchedModel.png        # Plot of the CDFs
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from PlotStyle import plotStyle
plotStyle()

def compute_median_cdf(df, column, n_points=100):
    """
    Compute a 'median' CDF by finding for a given set of mock surveys

    For each simulation, the values in 'column' are converted into an empirical
    quantile function x(p), i.e. the inverse CDF evaluated at a fixed set of
    quantile levels p ∈ [0, 1]. These quantile functions are then compared across
    simulations at the same p, which would not have been possible with the raw columns. 
    At each quantile level, the median (and 16th/84th percentiles) of x(p) across 
    simulations is computed, yielding a pointwise median inverse-CDF and uncertainty 
    envelope.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing mock-survey or simulation output. Each row
        corresponds to one object (e.g. a detected planet).
    column : str
        Column name containing the variable of interest (e.g. planet radius,
        mass, semi-major axis).
    simulation_col : str
        Column identifying which simulation / mock survey each row belongs to.
    n_points : int
        Number of quantile (CDF) levels p ∈ [0, 1] at which to evaluate the
        inverse CDF for each simulation.
    
    Returns:
    --------
    median_x : np.ndarray of shape (n_points,)
        Median x-value across simulations for each quantile p.
    p_values : np.ndarray of shape (n_points,)
        The quantile (CDF) values in [0,1] at which we computed median_x.
    """
    # Define a set of quantiles, e.g. 0, 0.01, 0.02, ..., 1
    p_values = np.linspace(0, 1, n_points)
    
    # List to store values corresponding to each quantile
    x_arrays = []
    
    for sim_id, group in df.groupby('simulation_id'):
        # Sort the data in this simulation
        values = np.sort(group[column].values)

        if len(values) == 0:
            continue
        
        # Get the corresponding quantities for each quantile p
        x_for_this_sim = np.quantile(values, p_values)
        x_arrays.append(x_for_this_sim)
    
    # Convert to NumPy array. Shape: (num_sims, n_points)
    x_arrays = np.array(x_arrays)
    
    # Take the median across the "simulation" axis
    median_x = np.median(x_arrays, axis = 0)

    lower_bound = np.percentile(x_arrays, 16, axis = 0)
    upper_bound = np.percentile(x_arrays, 84, axis = 0)
    
    return median_x, p_values, lower_bound, upper_bound

# ------------------------------------ I/O ------------------------------------- #
# Get the path to the current directory
curr_dir = Path(__file__).resolve().parent

data_dir = curr_dir.parent / "Data" / "Part II - Demographics" / "Mismatched OR" 

# Detected planets
planets_df = pd.read_csv(data_dir / "5d. Detected Planets.csv")

# Load the mock survey data
mock_survey = pd.read_csv(data_dir / '5g. Mock Survey.csv')
# ------------------------------------------------------------------------------ #


# --------------------------------  Observed Data  ----------------------------- #
# Read in the observed planet population and compute cdfs for the planet properties

# SMA
observed_sma     = np.sort(planets_df['SMA_AU'])
observed_sma_cdf = np.arange(1, len(observed_sma)+1) / len(observed_sma)
# Force the cdf to start at 0
observed_sma     = np.concatenate(([0], observed_sma))
observed_sma_cdf = np.concatenate(([0], observed_sma_cdf))

# Mass
observed_mass     = np.sort(planets_df["Mp_MEarth"])
observed_mass_cdf = np.arange(1, len(observed_mass)+1) / len(observed_mass)
observed_mass     = np.concatenate(([0], observed_mass))
observed_mass_cdf = np.concatenate(([0], observed_mass_cdf))

# Host star mass
observed_star_mass     = np.sort(planets_df["M_sol"])
observed_star_mass_cdf = np.arange(1, len(observed_star_mass)+1) / len(observed_star_mass)
observed_star_mass     = np.concatenate(([0], observed_star_mass))
observed_star_mass_cdf = np.concatenate(([0], observed_star_mass_cdf))

# Eccentricity
observed_ecc     = np.sort(planets_df["ecc"])
observed_ecc_cdf = np.arange(1, len(observed_ecc)+1) / len(observed_ecc)
observed_ecc     = np.concatenate(([0], observed_ecc))
observed_ecc_cdf = np.concatenate(([0], observed_ecc_cdf))
# ------------------------------------------------------------------------------ #


# Group the planets in the mock surveys by simulation_id (Treat all planets observed in one mock survey as one group)
grouped = mock_survey.groupby('simulation_id')

# Initialize the figure and subplots
fig, axs = plt.subplots(2, 3, figsize=(16.5, 10), gridspec_kw={'wspace': 0.24, 'hspace': 0.24})

# Flatten the axs array for easier indexing
axs = axs.flatten()

# Initialize lists to store the cumulative distributions
planet_mass_cumulative = []
sma_cumulative         = []
star_mass_cumulative   = []
ecc_cumulative         = []
num_planets            = []
num_HZ_planets         = []

# Calculate cumulative distribution across parameters for each group
for sim_id in range(0, 10_000):
    # If there were detected planets in the mock survey
    if sim_id in grouped.groups:
        group = grouped.get_group(sim_id)
        planet_mass_cumulative.append(np.sort(group['planet_mass']))
        sma_cumulative.append(np.sort(group['sma']))
        ecc_cumulative.append(np.sort(group['ecc']))
        star_mass_cumulative.append(np.sort(group['star_mass']))
        num_planets.append(len(grouped.get_group(sim_id)))
        num_HZ_planets.append(len(group[(group['HZ'] == True) & (group['planet_mass']>=0.5) & (group['planet_mass'] <= 2.8)]))
    else:
        # No planets detected in this simulation
        planet_mass_cumulative.append([0])
        sma_cumulative.append([0])
        ecc_cumulative.append(0)
        star_mass_cumulative.append([0])
        num_planets.append(0)
        num_HZ_planets.append(0)

# ------------------------------  Radius  ------------------------------------ #
for i, data in enumerate(planet_mass_cumulative[:1000]):
    x_vals = np.concatenate(([0], data))
    y_vals = np.concatenate(([0], np.arange(1, len(data) + 1) / len(data)))
    axs[0].step(x_vals, y_vals, where='post', alpha=0.1, color='#7FC5FF', lw=0.3)

# Compute the 'median' CDF with the 16th and 84th percentiles
mass_grid, median_cdf_mass, mass_lower, mass_upper  = compute_median_cdf(mock_survey, 'planet_mass')

# Force the cdf to start at zero for plotting
mass_grid       = np.concatenate(([0], mass_grid))
mass_lower      = np.concatenate(([0], mass_lower))
mass_upper      = np.concatenate(([0], mass_upper))
median_cdf_mass = np.concatenate(([0], median_cdf_mass))

# Plot the median, 16th and 84th percentile mock surveys
axs[0].step(mass_grid, median_cdf_mass, where='post', color = 'blue', lw = 3)
axs[0].step(mass_lower, median_cdf_mass, where='post', color = 'blue', lw = 2)
axs[0].step(mass_upper, median_cdf_mass, where='post', color = 'blue', lw = 2)

# Plot the actual data
axs[0].step(observed_mass, observed_mass_cdf, where = 'post', color = 'red', lw = 3)

axs[0].set_xlabel('Mass ($M_\oplus$)')
axs[0].set_ylabel('Cumulative Probability')
axs[0].set_xlim([0.01, 40])
axs[0].set_ylim([0,1])
axs[0].set_xscale('log')


axs[0].xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=10))
axs[0].xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=100))
axs[0].xaxis.set_minor_formatter(ticker.NullFormatter())

axs[0].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
axs[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
# ---------------------------------------------------------------------------- #


# ------------------------------  Period  ------------------------------------ #
for i, data in enumerate(sma_cumulative[:1000]):
    x_vals = np.concatenate(([0], data))
    y_vals = np.concatenate(([0], np.arange(1, len(data) + 1) / len(data)))
    axs[1].step(x_vals, y_vals, where='post', alpha=0.1, color='#7FC5FF', lw=0.3)

# Compute the 'median' CDF with the 16th and 84th percentiles
sma_grid, median_cdf_sma, sma_lower, sma_upper = compute_median_cdf(mock_survey, 'sma')

# Force the cdf to start at zero for plotting
sma_grid       = np.concatenate(([0], sma_grid))
sma_lower      = np.concatenate(([0], sma_lower))
sma_upper      = np.concatenate(([0], sma_upper))
median_cdf_sma = np.concatenate(([0], median_cdf_sma))

axs[1].step(sma_grid, median_cdf_sma, where='post', color = 'blue', lw = 3)
axs[1].step(sma_lower, median_cdf_sma, where='post', color = 'blue', lw = 2)
axs[1].step(sma_upper, median_cdf_sma, where='post', color = 'blue', lw = 2)
axs[1].step(observed_sma, observed_sma_cdf, where = 'post', color = 'red', lw = 3)
axs[1].set_xlabel('SMA (AU)')
axs[1].set_ylabel('Cumulative Probability')
axs[1].set_xlim([0.1,10])
axs[1].set_ylim([0,1])
axs[1].set_xscale('log')

axs[1].xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=10))
axs[1].xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=100))
axs[1].xaxis.set_minor_formatter(ticker.NullFormatter())

axs[1].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
axs[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
# ---------------------------------------------------------------------------- #


# ----------------------------  Host-star Mass  ------------------------------ #
# Plot cumulative distribution of star mass
for i, data in enumerate(star_mass_cumulative[:1000]):
    x_vals = np.concatenate(([0], data))
    y_vals = np.concatenate(([0], np.arange(1, len(data) + 1) / len(data)))
    axs[2].step(x_vals, y_vals, where='post', alpha=0.1, color='#7FC5FF', lw=0.3)

# Compute the 'median' CDF with the 16th and 84th percentiles
star_mass_grid, median_cdf_star, star_lower, star_upper = compute_median_cdf(mock_survey, 'star_mass')

star_mass_grid  = np.concatenate(([0], star_mass_grid))
star_lower      = np.concatenate(([0], star_lower))
star_upper      = np.concatenate(([0], star_upper))
median_cdf_star = np.concatenate(([0], median_cdf_star))

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
# ---------------------------------------------------------------------------- #


# ------------------------------  Eccentricity  ------------------------------ #
# Plot cumulative distribution of eccentricity
for i, data in enumerate(ecc_cumulative[:1000]):
    x_vals = np.concatenate(([0], data))
    y_vals = np.concatenate(([0], np.arange(1, len(data) + 1) / len(data)))
    axs[3].step(x_vals, y_vals, where='post', alpha=0.1, color='#7FC5FF', lw=0.3)

ecc_grid, median_cdf_ecc, ecc_lower, ecc_upper  = compute_median_cdf(mock_survey, 'ecc')

ecc_grid       = np.concatenate(([0], ecc_grid))
ecc_lower      = np.concatenate(([0], ecc_lower))
ecc_upper      = np.concatenate(([0], ecc_upper))
median_cdf_ecc = np.concatenate(([0], median_cdf_ecc))

axs[3].step(ecc_grid, median_cdf_ecc, where='post', color = 'blue', lw = 3)
axs[3].step(ecc_lower, median_cdf_ecc, where='post', color = 'blue', lw = 2)
axs[3].step(ecc_upper, median_cdf_ecc, where='post', color = 'blue', lw = 2)
axs[3].step(observed_ecc, observed_ecc_cdf, where = 'post', color = 'red', lw = 3)
axs[3].set_xlabel('Eccentricity')
axs[3].set_ylabel('Cumulative Probability')
axs[3].set_xlim([0, 0.25])
axs[3].set_ylim([0,1])

axs[3].xaxis.set_major_locator(ticker.MultipleLocator(0.04))
axs[3].xaxis.set_minor_locator(ticker.MultipleLocator(0.02))

axs[3].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
axs[3].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
# ---------------------------------------------------------------------------- #


# -----------------------------  No. of Planets  ----------------------------- #
# Plot frequency distribution of the number of planets per run
median       = np.median(num_planets)
lower, upper = np.percentile(num_planets, [16, 84])

bins = np.arange(0, max(num_planets) + 2, 5)
axs[4].hist(num_planets, bins=bins, edgecolor='#7FC5FF', histtype='step', lw = 3)
axs[4].axvline(x=median, color='blue', lw=6)
axs[4].axvline(x=lower, color='blue', lw=2)
axs[4].axvline(x=upper, color='blue', lw=2)
axs[4].axvline(x=len(planets_df), color='red', lw=3)
axs[4].set_xlabel('Number of Detected Companions')
axs[4].set_ylabel('Frequency')
axs[4].set_xlim([100, 230])

axs[4].xaxis.set_major_locator(ticker.MultipleLocator(20))
axs[4].xaxis.set_minor_locator(ticker.MultipleLocator(10))

axs[4].yaxis.set_major_locator(ticker.MultipleLocator(100))
axs[4].yaxis.set_minor_locator(ticker.MultipleLocator(50))
# ---------------------------------------------------------------------------- #

# ----------------------------  No. of exo-Earths  --------------------------- #
# Plot frequency distribution of the number of HZ planets per run
median       = np.median(num_HZ_planets)
lower, upper = np.percentile(num_HZ_planets, [16, 84])

true_HZ_planets = planets_df[(planets_df['HZ'] == True) & (planets_df['Rp_REarth'] >= 0.8) & (planets_df['Rp_REarth'] <= 1.4)]

bins = np.arange(0, max(num_HZ_planets) + 2, 1)
axs[5].hist(num_HZ_planets, bins=bins, edgecolor='#7FC5FF', histtype='step', lw = 3)
axs[5].axvline(x=median, color='blue', lw=6)
axs[5].axvline(x=lower, color='blue', lw=2)
axs[5].axvline(x=upper, color='blue', lw=2)
axs[5].axvline(x=len(true_HZ_planets), color='red', lw=3)
axs[5].set_xlabel('Number of Exo-Earth Companions')
axs[5].set_ylabel('Frequency')
axs[5].set_xlim([2, 32])

axs[5].xaxis.set_major_locator(ticker.MultipleLocator(4))
axs[5].xaxis.set_minor_locator(ticker.MultipleLocator(2))

axs[5].yaxis.set_major_locator(ticker.MultipleLocator(100))
axs[5].yaxis.set_minor_locator(ticker.MultipleLocator(50))
# ---------------------------------------------------------------------------- #

plt.savefig(curr_dir / "fig16_MismatchedModel.png", bbox_inches = 'tight', dpi = 300)
