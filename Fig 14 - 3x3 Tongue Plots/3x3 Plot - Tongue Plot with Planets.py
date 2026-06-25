"""
Tongue Plot with planet catalog overlaid 
-------------------------------------------
This script marginalises over the 4D tongue plot for all 9 IWA/contrast floor combinations,
and plots it as a heatmap with the detected and non-detected planets overlaid, and colour-coded.

Input:
    The tongue plots for all 9 IWA/contrast floor combinations in Data/Part II - Demographics/
    
Output:
    fig14_3x3_tplot.png      # Matches Fig 14
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import beta
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

from PlotStyle import plotStyle
plotStyle()

def HZ(L):
    """
    Find the inner and outer edges of the HZ

    Parameters:
    L (float/np.array): 

    Returns:
    HZ_inner (float/np.array): Distance to the inner HZ limit (in AU)
    HZ_outer (float/np.array): Distance to the outer HZ limit (in AU)
    """
    hz_inner = np.sqrt(L / 1.78)
    hz_outer = np.sqrt(L / 0.32)

    return hz_inner, hz_outer

# ------------------------------------ I/O ------------------------------------- #
curr_dir   = Path(__file__).resolve().parent
parent_dir = curr_dir.parent
data_dir   = parent_dir / "Data"
fit_dir    = data_dir / "Part II - Demographics"
# ------------------------------------------------------------------------------ #

# Directory names containing each individual tongue plot
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

# IWA/contrast pair, to display as text in the plot
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

colors = ["white", "yellow", "red", "blue"]
ylrdblu_cmap = LinearSegmentedColormap.from_list("YlRdBlu", colors, N=1024)

# Loop through all the directories and plot the tongue plots individually
for idx, run_dir_name in enumerate(run_dirs):
    run_dir = fit_dir / run_dir_name

    tplot_file = run_dir / f'{idx+1}b. 4D Tongue Plot.npz'
    log_file = run_dir / f'{idx+1}a. Observing Log.csv'

    # Load the tongue plot
    data = np.load(tplot_file, allow_pickle=True)

    # Store the 4D tongue plot, bin centers, and bin edges as separate arrays
    completeness_4d = data['completeness']      # Shape: (n_mass, n_sma, n_ecc, n_stars)

    # Bin centers
    rad_centers = data['rad_centers']
    per_centers = data['per_centers']
    ecc_centers = data['ecc_centers']

    #Bin edges
    rad_edges = data['rad_edges']
    per_edges = data['per_edges']
    ecc_edges = data['ecc_edges']

    # -----------------------  Marginalising to a 2D array  ------------------------ #
    # For plotting, we marginalise the tongue plot over the eccentricity and star dimensions

    # Marginalising over the ecc dimension using the Kipping 2013 Beta distrbution
    beta_weights     = beta.pdf(ecc_centers, a=0.867, b=3.03)
    beta_weights    /= np.sum(beta_weights)
    completeness_3d  = np.average(completeness_4d, axis=2, weights=beta_weights)  # over ecc

    # Marginalising over the stellar dimension by taking the average
    completeness_2d = np.sum(completeness_3d , axis=2)  # Now shape (n_rad, n_per)
    # ------------------------------------------------------------------------------- #

    ax = axs[idx]

    ax.set_xlim([np.log10(0.02), np.log10(40)])
    ax.set_ylim([np.log10(0.5), np.log10(3)])

    # Plot heatmap
    im = ax.imshow(completeness_2d,
                   origin='lower',
                   aspect='auto',
                   extent=[np.log10(per_edges[0]), np.log10(per_edges[-1]),
                           np.log10(rad_edges[0]), np.log10(rad_edges[-1])],
                   cmap=ylrdblu_cmap)
    
    # Smooth and draw contours
    smoothed      = gaussian_filter(completeness_2d, sigma=3.0)
    star_contours = levels = [1, 40, 80, 95]      # Number of stars for which contour lines are drawn
    colors = ['#8A2BE2', '#48D1CC', '#00FFFF', 'deeppink']
    labels = ['1 star', '40 stars', '80 stars', '95 stars']
    linstyles = ['dotted', 'dashdot', 'dashed', 'dashdot']

    # Store contour handles for legend only for the middle plot (idx == 4)
    contour_handles = []
    contour_labels = []

    for i, level in enumerate(levels):
        cs = ax.contour(np.log10(per_centers), np.log10(rad_centers), smoothed,
                        levels=[level], colors=colors[i], linewidths=2, linestyles = linstyles[i])

        if idx == 4:
            legend_elements = cs.legend_elements()[0]  # List of Line2D handles
            contour_handles.append(legend_elements[0])
            contour_labels.append(labels[i])

    # Add legend to the middle subplot
    if idx == 4:
        ax.legend(contour_handles, contour_labels, loc='upper left')

    # -----------------------  Group planets for plotting  -------------------------- #
    obs_df = pd.read_csv(log_file) 

    # Group by PlanetID and aggregate over all epochs
    grouped = obs_df.groupby('PlanetID').agg({
        'Rp_REarth': 'first',
        'SMA_AU': 'first',
        'ecc': 'first',
        'M_sol': 'first',
        'L_sol': 'first',
        'NDet': 'sum'  # Total number of detections over all epochs
    }).reset_index()

    # Planet period through Kepler's Third Law
    grouped['P'] = np.sqrt(grouped['SMA_AU']**3 / grouped['M_sol'])

    # Classify detections
    detected   = grouped[grouped['NDet'] >= 1]
    undetected = grouped[grouped['NDet'] == 0]

    # --------  Exo-Earth check  -------- #
    # Check radius and HZ conditions
    hz_inner, hz_outer = HZ(grouped['L_sol'])
    # HZ check
    grouped['hz'] = (grouped['SMA_AU'] * (1 - grouped['ecc']) >= hz_inner) & (grouped['SMA_AU'] * (1 + grouped['ecc']) <= hz_outer)
    # Radius Check
    exo_radius = (grouped['Rp_REarth'] > 0.8) & (grouped['Rp_REarth'] < 1.4)
    is_exo = grouped['hz'] & exo_radius
    # ----------------------------------- #
    # ------------------------------------------------------------------------------- #


    # ----------------------  Planet Plotting (scatter points) ---------------------- #
    # - Plotting choices:
    # - Detected --> Green (larger), Undetected --> Red (smaller)
    # - Exo-Earth --> Large diamond, Others --> Small circle

    # Detected and undetected exo-Earths
    det_exo = detected[is_exo.loc[detected.index]]
    und_exo = undetected[is_exo.loc[undetected.index]]

    # Detected and undetected non-exo-Earths
    det_non = detected[~is_exo.loc[detected.index]]
    und_non = undetected[~is_exo.loc[undetected.index]]

    # --- Non exo-Earths as circles ---
    ax.scatter(np.log10(det_non['P']), np.log10(det_non['Rp_REarth']),
            s=26, color='green', edgecolor='white', linewidth=0.6, alpha=0.95, zorder=3, label='Detected')
    ax.scatter(np.log10(und_non['P']), np.log10(und_non['Rp_REarth']),
            s=18, color='red', edgecolor='white', linewidth=0.6, alpha=0.6,  zorder=3, label='Undetected')

    # --- Exo-Earth overlay: diamond + bolder edge + subtle white halo ---
    halo = [pe.withStroke(linewidth=2.0, foreground='white')]

    ax.scatter(np.log10(det_exo['P']), np.log10(det_exo['Rp_REarth']),
            s=60, marker='D', facecolors='green', edgecolors='black', linewidths=1.2,
            alpha=1.0, zorder=5, path_effects=halo, label='Detected exo-Earths')

    ax.scatter(np.log10(und_exo['P']), np.log10(und_exo['Rp_REarth']),
            s=42, marker='D', facecolors='red', edgecolors='black', linewidths=1.2,
            alpha=0.9, zorder=5, path_effects=halo, label='Undetected exo-Earths')

    # --- Soft band to guide the eye for exo-Earth radii ---
    ax.axhspan(np.log10(0.8), np.log10(1.4), color='black', alpha=0.1, zorder=0)

    ax.axhline(y = np.log10(0.8), ls = "--", color = "grey")
    ax.axhline(y = np.log10(1.4), ls = "--", color = "grey")
    # ------------------------------------------------------------------------------- #

    # Axis settings
    per_ticks = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40]
    rad_ticks = [0.5, 1, 2, 3]
    ax.set_xticks(np.log10(per_ticks))
    ax.set_xticklabels(per_ticks)
    ax.set_yticks(np.log10(rad_ticks))
    ax.set_yticklabels(rad_ticks)

    if idx % 3 == 0:
        ax.set_ylabel('Radius (R$_\oplus$)')
    else:
        ax.set_yticklabels([])

    if idx >= 6:
        ax.set_xlabel('Period (yr)')
    else:
        ax.set_xticklabels([])

    ax.text(
    np.log10(0.021), np.log10(0.52), text[idx],
    fontsize=20, fontweight="bold", color="black",
    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.3")
    )
# -------------------------------------------------------------------------------- #

fig.tight_layout()
cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label('Number of Stars', rotation=270, labelpad=15)

plt.savefig(curr_dir / 'fig14_3x3_tplot.png', dpi=300, bbox_inches='tight')
plt.show()