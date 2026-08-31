"""
Tongue Plot with planet catalog overlaid 
-------------------------------------------
This script marginalises over the 4D tongue plot, and plots it as a heatmap with the 
detected and non-detected exo-Earths overlaid, and colour-coded

Input:
    Data/Part II - Demographics/Exo-Earth Centric ORs/5. IWA - 0.06, OWA - 1, Contrast = 1e-10/4D Tongue Plot.npz      # The 4D tongue plot data file for the fiducial IWA/CF
    Data/Part II - Demographics/Exo-Earth Centric ORs/5. IWA - 0.06, OWA - 1, Contrast = 1e-10/5a. Observing Log.csv   # Observing log for the exaggerated IWA/CF

Output:
    fig14_ExoEarth_tplot.png      # Matches Fig 14
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
data_dir = curr_dir.parent / "Data" / "Part II - Demographics" / "Exo-Earth Centric ORs" / "5. Fiducial Case - IWA - 0.06, Contrast = 1e-10"

# Load the saved tongue plot
data = np.load(data_dir / '5b. 4D Tongue Plot.npz', allow_pickle=True)

# Load planet simulation results
obs_df = pd.read_csv(data_dir / '5a. Observing Log.csv') 
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
smoothed      = gaussian_filter(completeness_2d, sigma=1.0)
star_contours = levels = [1, 10, 20, 25]      # Number of stars for which contour lines are drawn
colors = ['#8A2BE2', '#48D1CC', '#00FFFF', 'deeppink']
labels = ['1 star', '10 stars', '20 stars', '25 stars']
linstyles = ['dotted', 'dashdot', 'dashed', 'dashdot']

# Store contour handles for legend only for the middle plot (idx == 4)
contour_handles = []
contour_labels = []

for i, level in enumerate(levels):
    cs = ax.contour(np.log10(per_centers), np.log10(rad_centers), smoothed,
                    levels=[level], colors=colors[i], linewidths=2, linestyles = linstyles[i])


    legend_elements = cs.legend_elements()[0]  # List of Line2D handles
    contour_handles.append(legend_elements[0])
    contour_labels.append(labels[i])

# Contour legend: top right
contour_legend = ax.legend(contour_handles,contour_labels,loc='upper right')

# Keep this legend when another one is added later
ax.add_artist(contour_legend)

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
ax.set_xlim([np.log10(0.02), np.log10(40)])
# ------------------------------------------------------------------------------- #

# -----------------------  Group planets for plotting  -------------------------- #
# Group by PlanetID and aggregate over all epochs
grouped = obs_df.groupby('PlanetID').agg({
    'Rp_REarth': 'first',
    'SMA_AU': 'first',
    'ecc': 'first',
    'M_sol': 'first',
    'L_sol': 'first',
    'NDet': 'max'  # Total number of detections over all epochs
}).reset_index()

# Planet period through Kepler's Third Law
grouped['P'] = np.sqrt(grouped['SMA_AU']**3 / grouped['M_sol'])

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

ax.legend(loc = 'upper left')
# ------------------------------------------------------------------------------- #

plt.savefig(curr_dir / 'fig14_ExoEarth_tplot.png', dpi=300, bbox_inches='tight')