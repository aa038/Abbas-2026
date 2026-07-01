"""
Tongue Plot with planet catalog overlaid 
-------------------------------------------
This script marginalises over the 4D tongue plot,
and plots it as a heatmap with the detected and non-detected planets overlaid, and colour-coded.

Input:
    Data/Part II - Demographics/5. Fiducial Case - IWA - 0.06, Contrast = 1e-10/4D Tongue Plot.npz     # The 4D tongue plot data file for the fiducial IWA/CF
    Data/Part II - Demographics/5. Fiducial Case - IWA - 0.06, Contrast = 1e-10/5a. Observing Log.csv  # Observing log for the fiducial IWA/CF

Output:
    fig9_TonguePlot-w-Planets      # Matches Fig 9
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import beta
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap


# PlotStyle is a local library that will need to be installed
# Installation intructions can be found in REQUIREMENTS.md in the root directory
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
curr_dir = Path(__file__).resolve().parent

# Load the saved tongue plot
data = np.load(curr_dir / '2. Uniform Tongue Plot.npz', allow_pickle=True)

# Load planet simulation results
obs_df = pd.read_csv(curr_dir / '1. UC Observing Log - 1e-10.csv') 
# ------------------------------------------------------------------------------ #

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

# -------------------------  Plotting the Tongue Plot  -------------------------- #
fig, ax = plt.subplots(figsize=(10, 8))

# Defining a custom colour map to match the GPIES standard
colors       = ["white", "yellow", "red", "blue"]
ylrdblu_cmap = LinearSegmentedColormap.from_list("YlRdBlu", colors, N=1024)

# Plot the image
im = ax.imshow(completeness_2d,
               origin='lower',
               aspect='auto',
               extent=[np.log10(per_edges[0]), np.log10(per_edges[-1]),
                       np.log10(rad_edges[0]), np.log10(rad_edges[-1])],
               cmap=ylrdblu_cmap)

# Smooth and draw contours
smoothed      = gaussian_filter(completeness_2d, sigma=2)
star_contours = [1, 5, 10, 20, 40, 60, 80, 90, 95]      # Number of stars for which contour lines are drawn

# Contour definition
CS = ax.contour(np.log10(per_centers),
                np.log10(rad_centers),
                smoothed,
                levels=star_contours,
                colors='#cacfd2')

# Define ontour labels
contour_label = {1: '1 star'}    # Going the extra mile to write "1 star" instead of "1 stars". My mum taught English and Comp Litt. I wouldn't hear the end of it
contour_label.update({lvl: f'{lvl} stars' for lvl in star_contours[1:]})

# Distplay contour labels on the contours
ax.clabel(CS, CS.levels, inline=True, fmt=contour_label, fontsize=12)

# Labels
cbar = plt.colorbar(im)
cbar.set_label('Number of Stars', rotation=270, labelpad=15)
ax.set_xlabel('Period (yr)')
ax.set_ylabel('Planet Radius (R$_\oplus$)')

# Log ticks
per_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40]
rad_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3]
ax.set_xticks(np.log10(per_ticks))
ax.set_xticklabels(per_ticks)
ax.set_yticks(np.log10(rad_ticks))
ax.set_yticklabels(rad_ticks)

ax.set_ylim([np.log10(0.5), np.log10(3.4)])
# ------------------------------------------------------------------------------- #

# -----------------------  Group planets for plotting  -------------------------- #
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
grouped = grouped[grouped['P'] < 10]

# Classify detections
detected   = grouped[grouped['NDet'] >= 1]
undetected = grouped[grouped['NDet'] == 0]

print(len(detected))
print(len(detected) + len(undetected))

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
plt.axhline(y = np.log10(0.8), ls = "--", color = "grey")
plt.axhline(y = np.log10(1.4), ls = "--", color = "grey")

ax.legend(loc = 'upper left')
# ------------------------------------------------------------------------------- #

plt.savefig(curr_dir / 'fig19a_uc_tongue_plot.png', dpi=300, bbox_inches='tight')