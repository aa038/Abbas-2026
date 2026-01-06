import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from pathlib import Path

from PlotStyle import plotStyle

plotStyle()

plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['xtick.minor.size'] = 2
plt.rcParams['ytick.minor.size'] = 2

def get_levels(z):
    z = z.flatten()
    sorted_z = np.sort(z)[::-1]  # Sort in descending order
    cumsum = np.cumsum(sorted_z)
    cumsum /= cumsum[-1]  # Normalize
    # Find thresholds for 0.25, 0.50, 0.75 probability mass
    levels = []
    for level in [0.997, 0.95, 0.68]:  # Note: reversed order for plotting
        idx = np.searchsorted(cumsum, level)
        levels.append(sorted_z[idx])
    return levels

curr_dir = Path(__file__).resolve().parent

# Load your CSV file into a DataFrame
df = pd.read_csv(os.path.join(curr_dir, '5e. Fit, N = 1e4.csv'))

alpha_plus_one = df['alpha'] + 1
beta_plus_one = df['beta'] + 1

# Compute the exo-Earth OR
df['eta_Earth'] = df['freq'] * (1.5 ** alpha_plus_one - 0.8 ** alpha_plus_one) / (3.4 ** alpha_plus_one - 0.5 ** alpha_plus_one)
df['eta_Earth'] = df['eta_Earth'] * (2.15 ** beta_plus_one - 0.92 ** beta_plus_one) / (40 ** beta_plus_one - 0.02 ** beta_plus_one)
df['eta_Earth'] = df['eta_Earth'] 

# Extract the columns corresponding to the fitted parameters
parameters = df[['alpha', 'beta', 'gamma', 'e_alpha', 'e_beta', 'freq', 'eta_Earth']]


# Define parameter labels with LaTeX formatting
param_labels = ["$\\alpha$", "$\\beta$", "$\gamma$", "$e_{\\alpha}$", "$e_{\\beta}$", "$f$", "$\\eta_{Earth}$"]

# Define axis labels
x_labels = ["Mass Index", "SMA Index", "Stellar Mass Index", "$e_{\\alpha}$", "$e_{\\beta}$", "Occurrence Rate", "Exo-Earth OR"]
y_labels = ["Mass Index", "SMA Index", "Stellar Mass Index", "$e_{\\alpha}$", "$e_{\\beta}$", "Occurrence Rate", "Exo-Earth OR"]

# Define colors for the contours and corresponding percentile levels
contour_colors = ['green', 'blue', 'red']
contour_levels = [68, 95, 99.7]  

# Define fixed axis limits for each parameter:
# For indices 0,1,2: range is (-5, 5); for index 3 (frequency): range is (0, 35)
param_limits = [(-4, 2), (-2, 0), (-2, 2), (0.5, 5.5), (2.5, 8.5), (65.5, 105.5), (2.25, 3.75)]

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
            # Set the fixed x-limits based on the parameter index
            ax.set_xlim(param_limits[i])

            # Remove y-ticks and y-labels for the diagonal histograms
            ax.set_yticks([])
            ax.set_yticklabels([])
            
            # Set tick locators depending on the parameter:
            # For Mass, SMA, and Stellar Mass indices (indices 0,1,2)
            if i < 5:
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            # For frequency (index 3)
            elif i == 5:
                ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
            else:
                ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
            
            # Make ticks appear on all four sides
            ax.tick_params(axis='both', which='both', direction='in',
                           top=True, right=True, labelsize=8)
            
            # Add parameter label at the top right of the subplot
            ax.text(0.95, 0.95, param_labels[i], transform=ax.transAxes,
                    ha='right', va='top', fontsize=12)
        
        # Lower triangle: Plot 2D contours
        else:
            # For the x-axis use parameter j; for the y-axis use parameter i
            x_min, x_max = param_limits[j]
            y_min, y_max = param_limits[i]
            
            # Get the data for the two parameters
            x_data = parameters.iloc[:, j]
            y_data = parameters.iloc[:, i]
            
            # Compute kernel density estimate
            xy = np.vstack([x_data, y_data])
            kde = gaussian_kde(xy)
            
            # Create a grid for the contour plot using the fixed limits
            x_grid, y_grid = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            z = kde(np.vstack([x_grid.flatten(), y_grid.flatten()]))
            z = z.reshape(x_grid.shape)
            
            # Determine contour levels using the specified percentiles
            levels = get_levels(z)
            #levels = np.sort(levels)  # Ensure levels are in increasing order
        
            # Only plot if levels are strictly increasing
            if np.all(np.diff(levels) > 0):
                ax.contour(x_grid, y_grid, z, levels=levels, colors=contour_colors, linewidths=1)
            
            # Set the fixed axis limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # Set custom tick locators for the x-axis based on parameter j
            if j < 5:
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            # For frequency (index 3)
            elif j == 5:
                ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
            else:
                ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
            
            # Set custom tick locators for the y-axis based on parameter i
            if i < 5:
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            # For frequency (index 3)
            elif i == 5:
                ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
            else:
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
            
            # Make ticks appear on all four sides
            ax.tick_params(axis='both', which='both', direction='in',
                           top=True, right=True, labelsize=8)
            
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
    Line2D([0], [0], color='green', lw=1, label=r'$2\sigma$'),
    Line2D([0], [0], color='blue', lw=1, label=r'$3\sigma$')
]
fig.legend(handles=legend_elements, bbox_to_anchor=(0.85, 0.85), fontsize=12)

# Save the plot
plt.savefig(os.path.join(curr_dir, '5f. Corner Plot.png'), dpi=300, bbox_inches='tight')



