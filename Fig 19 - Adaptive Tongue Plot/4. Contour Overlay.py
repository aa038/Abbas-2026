"""
Overlay the tongue plot contours
-------------------------------------------
This script overlays the contours for the sensitivity plot from the uniform and adapative cadences on top of each other

Input:
    Data/Planet Generation/SAG13 Planet Catalog.csv       # The full planet catalog from Fig 1


Output:
    5a. Observing Log.csv                                 # The observing log for the chosen telescope configuration
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import beta
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter


# PlotStyle is a local library that will need to be installed
# Installation intructions can be found in REQUIREMENTS.md in the root directory
from PlotStyle import plotStyle
plotStyle()


def extract_data(data, smoothing_sigma = 1):

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

    smoothed_2d = gaussian_filter(completeness_2d, sigma=smoothing_sigma)

    return smoothed_2d, completeness_2d, rad_centers, per_centers, rad_edges, per_edges

# ---------------------------------- I/O ------------------------------------ #
curr_dir = Path(__file__).resolve().parent

adaptive = np.load(curr_dir / '2. Adaptive Tongue Plot.npz', allow_pickle=True)
uniform  = np.load(curr_dir / '2. Uniform Tongue Plot.npz', allow_pickle=True)
# --------------------------------------------------------------------------- #

C_ac_smooth, C_ac, rad_centers_ac, per_centers_ac, rad_edges_ac, per_edges_ac = extract_data(adaptive, smoothing_sigma=1)
C_uc_smooth, C_uc, rad_centers_uc, per_centers_uc, rad_edges_uc, per_edges_uc = extract_data(uniform, smoothing_sigma=2)

rad_centers = rad_centers_uc
per_centers = per_centers_uc
rad_edges = rad_edges_uc
per_edges = per_edges_uc

# ------------------------------- Plot ------------------------------------- #
fig, ax = plt.subplots(figsize=(10, 8))

star_contours = [1, 5, 10, 20, 40, 60, 80, 90, 95]

# Only use contour levels available in both maps
max_common = min(np.nanmax(C_uc_smooth), np.nanmax(C_ac_smooth))
levels = [lvl for lvl in star_contours if lvl <= max_common]

if len(levels) == 0:
    raise ValueError("No contour levels are within the range of both maps.")

# Uniform cadence: solid black
CS_uc = ax.contour(
    np.log10(per_centers),
    np.log10(rad_centers),
    C_uc_smooth,
    levels=levels,
    colors="black",
    linewidths=1.5,
    linestyles="solid"
)

# Adaptive cadence: dashed blue
CS_ac = ax.contour(
    np.log10(per_centers),
    np.log10(rad_centers),
    C_ac_smooth,
    levels=levels,
    colors="tab:blue",
    linewidths=1.5,
    linestyles="dashed"
)

# Label only the uniform contours to avoid clutter
contour_label = {1: "1 star"}
contour_label.update({lvl: f"{lvl} stars" for lvl in levels if lvl != 1})

ax.clabel(
    CS_uc,
    CS_uc.levels,
    inline=True,
    fmt=contour_label,
    fontsize=11
)

# Labels
ax.set_xlabel("Period (yr)")
ax.set_ylabel("Planet Radius (R$_\oplus$)")

# Log ticks
per_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40]
rad_ticks = [0.5, 0.8, 1, 1.4, 2, 3]

ax.set_xticks(np.log10(per_ticks))
ax.set_xticklabels(per_ticks)

ax.set_yticks(np.log10(rad_ticks))
ax.set_yticklabels(rad_ticks)

ax.set_xlim(np.log10(0.01), np.log10(40))
ax.set_ylim(np.log10(0.5), np.log10(3.4))

# Exo-Earth radius band
ax.axhspan(np.log10(0.8), np.log10(1.4), color="grey", alpha=0.15, zorder=0)
ax.axhline(y = np.log10(0.8), ls = "--", color = "grey")
ax.axhline(y = np.log10(1.4), ls = "--", color = "grey")

# Legend
legend_handles = [
    Line2D([0], [0], color="black", lw=1.8, linestyle="-", label="Uniform cadence"),
    Line2D([0], [0], color="tab:blue", lw=1.8, linestyle="--", label="Adaptive cadence")
]

ax.legend(handles=legend_handles, loc="upper left", frameon=True)

plt.savefig(curr_dir / "fig19c_contour_overlay.png", dpi=300, bbox_inches="tight")
plt.show()
# --------------------------------------------------------------------------- #
