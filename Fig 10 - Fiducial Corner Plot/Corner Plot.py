"""
Corner Plot for the OR parameters
-------------------------------------------
This script generates a corner plot for the results of the MCMC OR fit.

Input:
    Data/Part II - Demographics/5. Fiducial Case - IWA - 0.06, Contrast = 1e-10/5e. Fit, N = 1e4.csv        # The full list of MCMC posteriors for each parameter in the OR model (From Part 5)


Output:
    fig10_CornerPlot.png         # Corner plot for the posteriors
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

# PlotStyle is a local library that will need to be installed
# Installation intructions can be found in REQUIREMENTS.md in the root directory
from PlotStyle import plotStyle
plotStyle()

# Plotting parameters
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['xtick.minor.size'] = 2
plt.rcParams['ytick.minor.size'] = 2

def get_contour_levels(z):
    """
    Function to compute the 68%, 95% and 99.7% contour levels for a given parameter array z
    """
    z = z.flatten()

    # Compute the cumulative distribution
    sorted_z = np.sort(z)[::-1]  
    cumsum = np.cumsum(sorted_z)
    cumsum /= cumsum[-1]  # Normalize

    # Find thresholds for the 68%, 95% and 99.7% percentiles
    levels = []
    for level in [0.997, 0.95, 0.68]:  # Note: reversed order for plotting
        idx = np.searchsorted(cumsum, level)
        levels.append(sorted_z[idx])

    return levels

def sigma_uncerainty(quantity):
    """
    Function to compute the +/- 1 sigma uncertainties for a given parameter array
    """

    median = np.percentile(quantity, 50)

    lower = np.percentile(quantity, 16)
    upper = np.percentile(quantity, 84)

    return median, median - lower, upper - median

# ------------------------------------ I/O ------------------------------------- #
curr_dir = Path(__file__).resolve().parent
data_dir = curr_dir.parent / "Data" / "Part II - Demographics" / "5. Fiducial Case - IWA - 0.06, Contrast = 1e-10"

# Load the MCMC posteriors as a dataframe
df = pd.read_csv(data_dir / '5e. Fit, N = 1e4.csv')
# ------------------------------------------------------------------------------ #

# ---------------------  Exo-Earth OR for a 1M_sol star  ----------------------- #
alpha_plus_one = df['alpha'] + 1
beta_plus_one  = df['beta'] + 1

# Radius and period range over which the OR is defined
# Should match the numbers in 5e. Fitter.jl (Variable Name: direct_imaging_analysis_regions)
total_radius_range = [0.5, 3.4]
total_period_range = [0.03, 10]

# Compute the exo-Earth OR for a 1 M_sol star
# Scaling the total OR to exo-Earth radii (0.8 - 1.4 R_E)
df['eta_Earth'] = df['freq'] * (1.4 ** alpha_plus_one - 0.8 ** alpha_plus_one) / (total_radius_range[1] ** alpha_plus_one - total_radius_range[0] ** alpha_plus_one)
# Scaling the total OR to match the solar HZ period (0.92 - 2.15 yr)
df['eta_Earth'] = df['eta_Earth'] * (2.15 ** beta_plus_one - 0.92 ** beta_plus_one) / (total_period_range[1] ** beta_plus_one - total_period_range[0] ** beta_plus_one)
df['eta_Earth'] = df['eta_Earth'] * 100
# ------------------------------------------------------------------------------ #

# -----------------------------  Plotting Setup  ------------------------------- #
# List of parameters to be plotted in the corner plot
parameters = df[['alpha', 'beta', 'gamma', 'e_alpha', 'e_beta', 'freq', 'eta_Earth']]

# Define parameter labels with LaTeX formatting
param_labels = ["$\\alpha$", "$\\beta$", "$\gamma$", "$e_{\\alpha}$", "$e_{\\beta}$", "$f$", "$\\eta_{Earth}$"]

# Define axis labels
x_labels = ["Radius Index", "Period Index", "Stellar Mass Index", "$e_{\\alpha}$", "$e_{\\beta}$", "Occurrence Rate", "Exo-Earth OR (%)"]
y_labels = ["Radius Index", "Period Index", "Stellar Mass Index", "$e_{\\alpha}$", "$e_{\\beta}$", "Occurrence Rate", "Exo-Earth OR (%)"]

# Define colors for the contours and corresponding percentile levels for the correlation plots
contour_colors = ['green', 'blue', 'red']
contour_levels = [68, 95, 99.7]  

# Axis limits for each parameter
param_limits = [
    (-2, -0.25),   # alpha
    (-1.5, 0.25),   # beta
    (-2.5, 2.5),   # gamma
    (2.5,  6.5),   # e_alpha
    (32.5, 70.5),  # e_beta
    (2.5,  5.5),   # freq
    (10.0, 30.0),  # eta_Earth (%)
]


# Input SAG13 values for comparison
alpha     = -1.19
beta      = -0.84
gamma     = 0
e_alpha   = 3.7
e_beta    = 46.2
freq      = 3.9
eta_Earth = freq * (1.4 ** (alpha+1) - 0.8 ** (alpha+1)) / (3.4 ** (alpha+1) - 0.5 ** (alpha+1))
eta_Earth = eta_Earth * (2.15 ** (beta+1) - 0.92 ** (beta+1)) / (10 ** (beta+1) - 0.03 ** (beta+1))
eta_Earth = eta_Earth * 100
true_vals = [alpha, beta, gamma, e_alpha, e_beta, freq, eta_Earth]


# Create the figure and axes for the triangle plot
n_params = len(parameters.columns)
fig, axes = plt.subplots(n_params, n_params, figsize=(14, 12))
plt.subplots_adjust(wspace=0, hspace=0)

for i in range(n_params):
    for j in range(n_params):
        ax = axes[i, j]
        
        # Remove upper triangle subplots
        if i < j:
            ax.axis('off')
        
        # Diagonal: Plot 1D histograms
        elif i == j:
            param_data = parameters.iloc[:, i]

            ax.hist(param_data, bins=30, color='black', histtype='step', density=True, edgecolor='black')

            # Plot the true input SAG13 value
            ax.axvline(x = true_vals[i], ls = "--", color = 'red')

            # Set fixed x-limits based on the parameter index
            ax.set_xlim(param_limits[i])

            # Remove y-ticks and y-labels for the diagonal histograms
            ax.set_yticks([])
            ax.set_yticklabels([])
            
            # Set tick locators depending on the parameter:
            if i < 2:           # α, β
                ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
            elif i == 2:        # γ
                ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            elif i == 3:        # e_alpha
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            elif i == 4:        # e_beta
                ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            elif i == 5:        # freq
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            else:               # eta_Earth
                ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            
            # Make ticks appear on all four sides
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=8)
            
            # Add parameter label at the top right of the subplot
            ax.text(0.95, 0.95, param_labels[i], transform=ax.transAxes, ha='right', va='top', fontsize=12)
        
        # Lower triangle: Plot 2D contours
        else:
            # For the x-axis use parameter j; for the y-axis use parameter i
            x_min, x_max = param_limits[j]
            y_min, y_max = param_limits[i]
            
            # Get the data for the two parameters
            x_data = parameters.iloc[:, j]
            y_data = parameters.iloc[:, i]
            
            # 2D joint raw posterior cloud
            xy  = np.vstack([x_data, y_data])

            # Fit a 2D Gaussian kernel on every sample point, add and normalise
            # This gives a smooth approximation to the joint posterior
            kde = gaussian_kde(xy)
            
            #  EVenly spaced grid to evaluate the gaussian kde on
            x_grid, y_grid = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

            # Evaluate the Gaussian kernel across the full axis range
            z = kde(np.vstack([x_grid.flatten(), y_grid.flatten()]))
            z = z.reshape(x_grid.shape)
            
            # Determine contour levels using the specified percentiles
            levels = get_contour_levels(z)
        
            ax.contour(x_grid, y_grid, z, levels=levels, colors=contour_colors, linewidths=1)

            # Plot the true input SAG13 values for comparison
            ax.scatter(true_vals[j], true_vals[i], s = 20, color = 'red', edgecolors= "black", zorder = 5)
            
            # Set the fixed axis limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # Set custom tick locators for the x-axis based on parameter j
            if j < 2:           
                ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
            elif j == 2:           
                ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            elif j == 3:
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            elif j == 4:
                ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            elif j == 5:
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            else:
                ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            
            # Set custom tick locators for the y-axis based on parameter i
            if i < 2:           
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
            elif i == 2:           
                ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            elif i == 3:
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            elif i == 4:
                ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
            elif i == 5:
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            else:
                ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
            
            # Make ticks appear on all four sides
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=8)
            
            # Hide tick labels unless on the bottom row (for x-axis) or left column (for y-axis)
            if i != n_params - 1:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])

# Add x-axis labels to the bottom row
for j in range(n_params):
    axes[-1, j].set_xlabel(x_labels[j], fontsize=12, labelpad=10)

# Add y-axis labels to the left column
for i in range(n_params):
    axes[i, 0].set_ylabel(y_labels[i], fontsize=12, labelpad=10)

# Add a legend for the contour levels
legend_elements = [
    Line2D([0], [0], color='red', lw=1, label=r'$1\sigma$'),
    Line2D([0], [0], color='blue', lw=1, label=r'$2\sigma$'),
    Line2D([0], [0], color='green', lw=1, label=r'$3\sigma$')
]
fig.legend(handles=legend_elements, bbox_to_anchor=(0.85, 0.85), fontsize=12)
# ------------------------------------------------------------------------------ #

plt.savefig(curr_dir / 'fig10_CornerPlot.png', dpi=300, bbox_inches='tight')



