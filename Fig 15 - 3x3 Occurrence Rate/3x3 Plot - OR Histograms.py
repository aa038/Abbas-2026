"""
Tongue Plot with planet catalog overlaid 
-------------------------------------------
This script reads in the MCMC fits for all 9 combinations of IWA/contrast floor and plots the recovered 
OR distributions as a 3x3 panel plot.

Input:
    The MCMC posteriors for all 9 IWA/contrast floor combinations in Data/Part II - Demographics/
    
Output:
    fig15_3x3_ORPlot.png      # Matches Fig 15
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from PlotStyle import plotStyle
plotStyle()

# ------------------------------------ I/O ------------------------------------- #
curr_dir   = Path(__file__).resolve().parent
parent_dir = curr_dir.parent
data_dir   = parent_dir / "Data"
fit_dir    = data_dir / "Part II - Demographics"
# ------------------------------------------------------------------------------ #

# Directory names containing each individual fit
run_dirs = [
    '1. IWA - 0.04, OWA - 1, Contrast = 1e-9',
    '2. IWA - 0.06, OWA - 1, Contrast = 1e-9',
    '3. IWA - 0.08, OWA - 1, Contrast = 1e-9',
    '4. IWA - 0.04, OWA - 1, Contrast = 1e-10',
    '5. Fiducial Case - IWA - 0.06, Contrast = 1e-10',
    '6. IWA - 0.08, OWA - 1, Contrast = 1e-10',
    '7. IWA - 0.04, OWA - 1, Contrast = 1e-11',
    '8. IWA - 0.06, OWA - 1, Contrast = 1e-11',
    '9. IWA - 0.08, OWA - 1, Contrast = 1e-11'
]

text = [
    'a) $0.04^{\prime\prime}, 10^{-9}$',
    'b) $0.06^{\prime\prime}, 10^{-9}$',
    'c) $0.08^{\prime\prime}, 10^{-9}$',
    'd) $0.04^{\prime\prime}, 10^{-10}$',
    'e) $0.06^{\prime\prime}, 10^{-10}$',
    'f) $0.08^{\prime\prime}, 10^{-10}$',
    'g) $0.04^{\prime\prime}, 10^{-11}$',
    'h) $0.06^{\prime\prime}, 10^{-11}$',
    'i) $0.08^{\prime\prime}, 10^{-11}$'
]

# -----------------------------  Plotting Setup  --------------------------------- #
fig, axs = plt.subplots(3, 3, figsize=(15, 12))
axs = axs.flatten()

# Loop through all the directories and plot the OR distributions individually
for idx, run_dir_name in enumerate(run_dirs):

    run_dir   = fit_dir / run_dir_name
    data_file = run_dir / f'{idx+1}e. Fit, N = 1e4.csv'

    # Load the fit data
    df = pd.read_csv(data_file)

    # Extract the OR column from the fit
    freq_data = df['freq']

    ax = axs[idx]

    # Plot histogram for 'freq'
    ax.hist(freq_data, bins=30, histtype='step', density=True, lw = 2)

    # Plot dotted vertial line at actual value
    ax.axvline(x = 3.9, ls = "--", color = 'red')

    # Set axis labels and limits
    ax.set_xlim(0.5, 13.5)  
    ax.set_yticks([])  # Hide y-axis ticks for clean look

    if idx < 3:
        ax.set_ylim(0, 0.3)
        ax.text(9.5, 0.27, text[idx])
    elif idx < 6:
        ax.set_ylim(0, 1.5)
        ax.text(9.5, 1.35, text[idx])
    else:
        ax.set_ylim(0, 1.53)
        ax.text(9.5, 1.35, text[idx])        

    if idx >= 6:
        ax.set_xlabel('Occurrence Rate')
    else:
        ax.set_xticklabels([])

    # Customize x-axis ticks
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=14)
# -------------------------------------------------------------------------------- #

# Tight layout
plt.tight_layout(rect=[0, 0.05, 1, 1])

# Save the figure
plt.savefig(curr_dir / 'fig15_3x3_ORPlot.png', dpi=300, bbox_inches='tight')
plt.show()