"""
Tongue Plot with planet catalog overlaid (Part 3 of 9)
-------------------------------------------
This script marginalises over the 4D tongue plot,
and plots it as a heatmap with the detected and non-detected planets overlaid, and colour-coded.

Input:
    5b. 4D Tongue Plot.npz                # The 4D tongue plot data file


Output:
    5c. Tongue Plot with Planets.png      # The marginalised 2D tongue plot with the planets overlaid
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
curr_dir   = Path(__file__).resolve().parent

# Load the saved tongue plot
data = np.load(curr_dir / '5b. 4D Tongue Plot.npz', allow_pickle=True)

# Load planet simulation results
obs_df = pd.read_csv(curr_dir / '5a. Observing Log.csv') 
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
star_contours = [1, 5, 10, 20, 40, 60, 80, 90, 95]      # Number of stars for which contour lines are drawn

# Contour definition
CS = ax.contour(np.log10(per_centers),
                np.log10(rad_centers),
                smoothed,
                levels=star_contours,
                colors='#cacfd2')

# Define contour labels
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

# Classify detections
detected   = grouped[grouped['NDet'] >= 1]
undetected = grouped[grouped['NDet'] == 0]

print(len(detected))
print(len(detected) + len(undetected))

# ---------------- Completeness-weighted variance diagnostic ---------------- #

# Use the 3D completeness grid relevant to the demographic likelihood.
# This is summed over stars but NOT marginalized over eccentricity.
C_grid = np.sum(completeness_4d, axis=3)  # shape: (n_rad, n_per, n_ecc)


def get_bin_index(values, edges):
    """
    Return the bin index for each value given bin edges.
    Values outside the grid are clipped to the nearest valid bin.
    """
    idx = np.searchsorted(edges, values, side='right') - 1
    return np.clip(idx, 0, len(edges) - 2)


def local_invC_rms(ir, ip, ie, C_grid):
    """
    RMS local difference in inverse completeness between cell p and
    its 26 neighboring cells in (Rp, P, e).
    """
    Cp = C_grid[ir, ip, ie]

    if not np.isfinite(Cp) or Cp <= 0:
        return np.nan, np.nan

    diffs = []

    for dr in [-1, 0, 1]:
        for dp in [-1, 0, 1]:
            for de in [-1, 0, 1]:

                # Skip central cell
                if dr == 0 and dp == 0 and de == 0:
                    continue

                rr = ir + dr
                pp = ip + dp
                ee = ie + de

                # Skip out-of-bounds neighbors
                if (
                    rr < 0 or rr >= C_grid.shape[0] or
                    pp < 0 or pp >= C_grid.shape[1] or
                    ee < 0 or ee >= C_grid.shape[2]
                ):
                    continue

                Cq = C_grid[rr, pp, ee]

                if np.isfinite(Cq) and Cq > 0:
                    diffs.append((1.0 / Cp - 1.0 / Cq)**2)

    if len(diffs) == 0:
        return np.nan, np.nan

    D_rms = np.sqrt(np.mean(diffs))
    g_frac = D_rms / (1.0 / Cp)

    return D_rms, g_frac


# Work on a copy so diagnostics do not mess with plotting
det_diag = detected.copy()

# Bin each detected planet into the completeness grid
det_diag['rad_bin'] = get_bin_index(det_diag['Rp_REarth'].values, rad_edges)
det_diag['per_bin'] = get_bin_index(det_diag['P'].values, per_edges)
det_diag['ecc_bin'] = get_bin_index(det_diag['ecc'].values, ecc_edges)

# Extract completeness for each detected planet's true cell
det_diag['C_cell'] = C_grid[
    det_diag['rad_bin'].values,
    det_diag['per_bin'].values,
    det_diag['ecc_bin'].values
]

# Remove pathological cells if any
det_diag = det_diag[np.isfinite(det_diag['C_cell']) & (det_diag['C_cell'] > 0)].copy()

# Completeness weight per detected planet
det_diag['invC'] = 1.0 / det_diag['C_cell']

# Completeness-weighted detected count and Poisson variance
N_weighted = det_diag['invC'].sum()
sigma2_pois_C = np.sum(det_diag['invC']**2)
sigma_pois_C = np.sqrt(sigma2_pois_C)

# Local inverse-completeness roughness around each detected planet
local_vals = [
    local_invC_rms(ir, ip, ie, C_grid)
    for ir, ip, ie in zip(det_diag['rad_bin'], det_diag['per_bin'], det_diag['ecc_bin'])
]

det_diag['D_invC_rms'] = [v[0] for v in local_vals]
det_diag['g_frac']     = [v[1] for v in local_vals]

# A conservative scale for the possible effect of one-cell leakage,
# evaluated for all detected planets.
sigma_local_all = np.sqrt(np.nansum(det_diag['D_invC_rms']**2))

print("\n--- Completeness-weighted diagnostic ---")
print(f"Detected planets used: {len(det_diag)}")
print(f"Completeness-weighted detected count, sum(1/C): {N_weighted:.3f}")
print(f"Completeness-weighted Poisson sigma, sqrt(sum(1/C^2)): {sigma_pois_C:.3f}")
print(f"Median local fractional inverse-C roughness: {np.nanmedian(det_diag['g_frac']):.3f}")
print(f"84th percentile local fractional inverse-C roughness: {np.nanpercentile(det_diag['g_frac'], 84):.3f}")
print(f"95th percentile local fractional inverse-C roughness: {np.nanpercentile(det_diag['g_frac'], 95):.3f}")
print(f"Conservative all-detected local leakage scale: {sigma_local_all:.3f}")
print(f"Leakage scale / weighted Poisson sigma: {sigma_local_all / sigma_pois_C:.3f}")
# -------------------------------------------------------------------------- #

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
ax.axhspan(np.log10(0.8), np.log10(1.4), color='#e6e6e6', alpha=0.3, zorder=0)

ax.legend(loc = 'upper left')
# ------------------------------------------------------------------------------- #

plt.savefig(curr_dir / '5c. Tongue Plot with Planets.png', dpi=300, bbox_inches='tight')