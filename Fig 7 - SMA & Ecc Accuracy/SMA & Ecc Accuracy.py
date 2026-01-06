"""
Accuracy of SMA and Ecc after 8 epochs
-------------------------------------------
This script reproduces Fig. 7 from Abbas et al. 2026.
Accuracy of the fitted sma and ecc after 8 epochs 

Input:
Takes in the orbit fits for the planets observed through the adaptive cadence 

    4. AC Fits - 2.5e-11.pkl

    OR

    4. AC Fits - 4e-11.pkl

Output:
    fig7_recovered_sma_2.5e-11.png

    OR

    fig7_recovered_sma_4e-11.png

    depending on the user's choice of input orbit fits

Notes:
- 2 contrast floors: 2.5e-11 and 4e-11 for the adaptive cadence
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# PlotStyle is a local library that needs to be installed.
# Installation intructions can be found in REQUIREMENTS.md in the root directory
from PlotStyle import plotStyle
plotStyle()

def HZ(L):
   """
   Calculate inner and outer edges of the habitable zone of a star,
   based off Kopparappu 2013.
   """
   return np.sqrt(L / 1.78), np.sqrt(L / 0.32)

def compute_hz_probability(orbit_df, L):
   """
   Compute the fraction of orbital posterior samples within the HZ of a star
   """
   # HZ boundaries
   hz_in, hz_out = HZ(L)

   # Periastron and apastron for all the orbital posteriors
   peri = orbit_df["sma"] * (1 - orbit_df["ecc"])
   ap   = orbit_df["sma"] * (1 + orbit_df["ecc"])

   # Boolean mask to check if an orbit is in the HZ
   inside = (peri > hz_in) & (ap < hz_out)

   # Fraction of HZ orbits
   return inside.sum() / len(orbit_df)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> USER-TUNABLE VALUES <<<<<<<<<<<<<<<<<<<<<<<<<<< #
# Figures 7 and 8 only consider the results from the adaptive cadences
# That leaves two sets of orbit fits, for the contrast floors 2.5e-11 and 4e-11
input_orbit_fits = "4. AC Fits - 4e-11.pkl"                  
#input_orbit_fits = "4. AC Fits - 2.5e-11.pkl"

fig_title         = "Contrast Floor = $4\\times10^{-11}$"
#fig_title         = "Contrast Floor = $2.5\\times10^{-11}$"

output_file_name  = "fig7_recovered_sma_4e-11.png"
#output_file_name  = "fig7_recovered_sma_2.5e-11.png"
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #

# ---------------------------------- I/O ------------------------------------ #
curr_dir = Path(__file__).resolve().parent
parent_dir = curr_dir.parent
data_dir = parent_dir / "Data" / "Part I - Observing Cadence"

# Planet catalog
planets_df_dir = data_dir / "1. Planet Catalog.csv"
planets_df     = pd.read_csv(planets_df_dir)
planets_df.set_index("PlanetID", inplace=True)

# Fits to the planets in the observing log
fits_path = data_dir / input_orbit_fits

# This is a MultiIndex Dataframe containing the orbit fits for every observed planet at every epoch
# The indices are ["PlanetID", "EpochNum"]
# For more details on this file, see the docstring in Data/Part I - Observing Cadence/4. Orbit Fits.py
df_fits = pd.read_pickle(fits_path)
# --------------------------------------------------------------------------- #

# Figure 7 shows the accuracy of the sma and ecc after **8 EPOCHS**
# Therefore, choose only the 8th epoch of observation for all these planets
idx = pd.IndexSlice
df_fits = df_fits.loc[idx[:, 8], :]

# Grab the planet IDs from the MultiIndex
planet_ids = df_fits.index.get_level_values("PlanetID")

# Extract the true values for sma and ecc from the Planet Catalog
true_sma = planets_df.loc[planet_ids, "sma_AU"].values 
true_ecc = planets_df.loc[planet_ids, "ecc"].values 

# ----------------  Empty arrays to store:  ---------------------- #
# -   1. Median sma and ecc from the fits after 8 epochs
# -   2. 16th and 84th percentiles for sma and ecc (for errorbars)
# -   3. Confidence the planet is in the HZ from orbital posteriors (for color-coding the scatter points)

fit_sma = np.empty(len(df_fits))
fit_ecc = np.empty(len(df_fits))

sma_p16 = np.empty(len(df_fits)); sma_p84 = np.empty(len(df_fits))
ecc_p16 = np.empty(len(df_fits)); ecc_p84 = np.empty(len(df_fits))

hz_prob = np.empty(len(df_fits))
# ---------------------------------------------------------------- #

for i, ((pid, epochNum), row) in enumerate(df_fits.iterrows()):
   # The "orbit_df" column contains a dataframe with the full set of orbital posteriors at that epoch
   # This is a datafram inside the parent multiIndex dataframe
   orbit_df = row["orbit_df"]
   L_star   = row["L_sol"]

   hz_prob[i] = compute_hz_probability(orbit_df, L_star)

   # Median and 16th/84th percentiles for sma and ecc for planet i
   # Median sma and ecc
   fit_sma[i] = orbit_df['sma'].median()
   fit_ecc[i] = orbit_df['ecc'].median()

   # 16/84 percentiles (for asymmetric errorbars)
   sma_p16[i], sma_p84[i] = np.percentile(orbit_df['sma'].values, [16, 84])
   ecc_p16[i], ecc_p84[i] = np.percentile(orbit_df['ecc'].values, [16, 84])

# Boolean mask for planets classified at 95% confidence vs not
mask_hz95 = hz_prob >= 0.95
mask_non  = ~mask_hz95

# -----------------------  Error bars in SMA/Ecc  ------------------------ #
# Shape (2, n_planets)
sma_yerr = np.vstack([fit_sma - sma_p16, sma_p84 - fit_sma])
ecc_yerr = np.vstack([fit_ecc - ecc_p16, ecc_p84 - fit_ecc])

# Ensure the lower limit for the error bar is >= 0 (since negative SMA and ecc are unphysical)
sma_yerr[0] = np.clip(sma_yerr[0], 1e-6, None)  # Clipped at 1e-6 (i.e. >0) since the y-axis is in logspace

ecc_yerr[0] = np.clip(ecc_yerr[0], 0, None)
ecc_yerr[1] = np.clip(ecc_yerr[1], 0, None)
# ----------------------------------------------------------------------- #

# -------------  Plot Median SMA and Ecc with error bars  --------------- #
def eb(ax, x, y, yerr, color, label):
   """
   Plot x vs y with vertical error bars, with fixed plot formatting options.
   """
   ax.errorbar(x, y, yerr=yerr, mfc=color, label=label, 
               fmt='o', ms=5, elinewidth=1.5, ecolor=color, mec='k', mew=1, capsize=2, alpha=0.95, zorder=2)


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Colours from the Tableau 10 palette (muted, visually balanced colours)
# Default green and red are aggressively bright
# HZ with 95% confidence --> Green, else --> Red
c_in  = 'tab:green'
c_out = 'tab:red'

# ------------  SMA (log-log)  ----------- #
ax = axes[0]
ax.plot([0.1, 10], [0.1, 10], 'k--', zorder=1)


# Planets detected with >= 95% confidence
eb(ax, true_sma[mask_hz95], fit_sma[mask_hz95],                 # Axis, x (True SMA), y (Fitted SMA)
   np.vstack([sma_yerr[0, mask_hz95], sma_yerr[1, mask_hz95]]), # (Lower bound, Upper bound)
   c_in, f'$HZ \geq 95\% ({mask_hz95.sum()})$')                 # Colour (Green), Label

# Planets detected with < 95% confidence
eb(ax, true_sma[mask_non], fit_sma[mask_non],                   # Axis, x (True SMA), y (Fitted SMA)
   np.vstack([sma_yerr[0, mask_non], sma_yerr[1, mask_non]]),   # (Lower bound, Upper bound)
   c_out, f'$HZ < 95\% ({mask_non.sum()})$')                    # Colour (Red), Label

ax.legend(frameon=False)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.1, 10)
ax.set_ylim(0.1, 10)
ax.set_xlabel("True SMA (AU)")
ax.set_ylabel("Recovered SMA (AU)")
# ---------------------------------------- #

# ------------  Ecc (Linear)  ------------ #
ax = axes[1]
ax.plot([0, 0.4], [0, 0.4], 'k--', zorder=1)

# Planets detected with >= 95% confidence
eb(ax, true_ecc[mask_hz95], fit_ecc[mask_hz95],
   np.vstack([ecc_yerr[0, mask_hz95], ecc_yerr[1, mask_hz95]]),
   c_in, None)

# Planets detected with < 95% confidence
eb(ax, true_ecc[mask_non], fit_ecc[mask_non],
   np.vstack([ecc_yerr[0, mask_non], ecc_yerr[1, mask_non]]),
   c_out, None)

ax.set_xlim(0, 0.4)
ax.set_ylim(0, 0.4)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.set_xlabel("True ecc")
ax.set_ylabel("Recovered ecc")
# ---------------------------------------- #

fig.suptitle(fig_title)
# ----------------------------------------------------------------------- #

plt.savefig(curr_dir / output_file_name, dpi=300, bbox_inches = 'tight')
plt.show()


