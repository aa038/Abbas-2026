"""
Accuracy of SMA and Ecc after 8 epochs
-------------------------------------------
This script reproduces Fig. 7b from Abbas et al. 2026.
Accuracy of the fitted sma and ecc after 8 epochs 

Input:
Takes in the orbit fits for the planets observed in the fiducial configuration

    5. Data / Part II - Demographics / 5. Fiducial Case - IWA - 0.06, Contrast = 1e-10 / 5a. Orbit Fits.pkl


Output:
    fig7b_recovered_sma.png

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

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> USER-TUNABLE VALUES <<<<<<<<<<<<<<<<<<<<<<<<<<< #
# Figures 7 and 8 consider the orbit fits to the 8 epochs spaced 3 months apart case,
# for the fiducial IWA = 0.06" and contrast floor = 1e-10 case.
            
input_orbit_fits = "5a. Orbit Fits.pkl"

output_file_name  = "fig7b_recovered_sma.png"
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #

# ---------------------------------- I/O ------------------------------------ #
curr_dir = Path(__file__).resolve().parent
parent_dir = curr_dir.parent
data_dir = parent_dir / "Data" / "Part II - Demographics" / "5. Fiducial Case - IWA - 0.06, Contrast = 1e-10"

# Planet catalog
planets_df_dir = parent_dir / "Data" / "Planet Generation" / "SAG13 Planet Catalog.csv"
planets_df     = pd.read_csv(planets_df_dir)
planets_df     = planets_df[planets_df['P_yr'] < 10]
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

# Keep only rows with actual orbit posteriors
df_fits = df_fits[df_fits["orbit_df"].notna()]
df_fits = df_fits[df_fits["orbit_df"].apply(lambda x: isinstance(x, pd.DataFrame) and len(x) > 0)]

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
# ---------------------------------------------------------------- #

for i, ((pid, epochNum), row) in enumerate(df_fits.iterrows()):
   # The "orbit_df" column contains a dataframe with the full set of orbital posteriors at that epoch
   # This is a datafram inside the parent multiIndex dataframe
   orbit_df = row["orbit_df"]

   # Median and 16th/84th percentiles for sma and ecc for planet i
   # Median sma and ecc
   fit_sma[i] = orbit_df['sma'].median()
   fit_ecc[i] = orbit_df['ecc'].median()

   # 16/84 percentiles (for asymmetric errorbars)
   sma_p16[i], sma_p84[i] = np.percentile(orbit_df['sma'].values, [16, 84])
   ecc_p16[i], ecc_p84[i] = np.percentile(orbit_df['ecc'].values, [16, 84])

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
def eb(ax, x, y, yerr):
   """
   Plot x vs y with vertical error bars, with fixed plot formatting options.
   """
   ax.errorbar(x, y, yerr=yerr, 
               fmt='o', ms=5, elinewidth=1.5, mec='k', mew=1, capsize=2, alpha=0.95, zorder=2)


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ------------  SMA (log-log)  ----------- #
ax = axes[0]

amin = min(np.nanmin(true_sma), np.nanmin(sma_p16))
amax = max(np.nanmax(true_sma), np.nanmax(sma_p84))

ax.set_xlim(0.8 * amin, 1.2 * amax)
ax.set_ylim(0.8 * amin, 1.2 * amax)
ax.plot([0.8 * amin, 1.2 * amax], [0.8 * amin, 1.2 * amax], 'k--', zorder=1)

eb(ax, true_sma, fit_sma,                      # Axis, x (True SMA), y (Fitted SMA)
   np.vstack([sma_yerr[0,:], sma_yerr[1, :]])) # (Lower bound, Upper bound)

ax.legend(frameon=False)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Injected $a$ (AU)")
ax.set_ylabel("UC recovered $a$ (AU)")
# ---------------------------------------- #

# ------------  Ecc (Linear)  ------------ #
ax = axes[1]

emax = 0.6
ax.plot([0, emax], [0, emax], 'k--', zorder=1)
ax.set_xlim(0, emax)
ax.set_ylim(0, emax)

eb(ax, true_ecc, fit_ecc, np.vstack([ecc_yerr[0, :], ecc_yerr[1, :]]))

ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.set_xlabel("Injected $e$")
ax.set_ylabel("UC recovered $e$")
# ---------------------------------------- #

# ----------------------------------------------------------------------- #

plt.savefig(curr_dir / output_file_name, dpi=300, bbox_inches = 'tight')
plt.show()


