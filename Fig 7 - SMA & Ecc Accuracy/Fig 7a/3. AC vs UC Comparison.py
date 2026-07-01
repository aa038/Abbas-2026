"""
Accuracy of SMA and Ecc after 8 epochs
-------------------------------------------
This script reproduces Fig. 7a from Abbas et al. 2026.
Consistency of the adaptive and uniform cadences after 8 epochs 

Input:
Takes in the orbit fits for the planets observed in the fiducial configuration

    5. Data / Part II - Demographics / 5. Fiducial Case - IWA - 0.06, Contrast = 1e-10 / 5a. Orbit Fits.pkl


Output:
    fig_7a_adaptive_vs_uniform.png

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

def extract_orbit_summary(df_fits, epoch_num=8):
   """
   Extract recovered median a/e and 16th/84th percentiles
   from a MultiIndex orbit-fit dataframe with indices:
   ["PlanetID", "EpochNum"].

   Returns a dataframe indexed by PlanetID.
   """

   idx = pd.IndexSlice

   # Keep only the requested epoch
   df = df_fits.loc[idx[:, epoch_num], :].copy()

   # Keep only rows with valid orbit posterior dataframes
   df = df[df["orbit_df"].notna()]
   df = df[df["orbit_df"].apply(lambda x: isinstance(x, pd.DataFrame) and len(x) > 0)]

   rows = []

   for (pid, epoch), row in df.iterrows():
      orbit_df = row["orbit_df"]

      rows.append({
         "PlanetID": pid,

         "a_med": orbit_df["sma"].median(),
         "a_p16": np.percentile(orbit_df["sma"].values, 16),
         "a_p84": np.percentile(orbit_df["sma"].values, 84),

         "e_med": orbit_df["ecc"].median(),
         "e_p16": np.percentile(orbit_df["ecc"].values, 16),
         "e_p84": np.percentile(orbit_df["ecc"].values, 84),
      })

   out = pd.DataFrame(rows)

   if out.empty:
      return out

   return out.set_index("PlanetID")

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> USER-TUNABLE VALUES <<<<<<<<<<<<<<<<<<<<<<<<<<< #
# Figures 7 and 8 consider the orbit fits to the 8 epochs spaced 3 months apart case,
# for the fiducial IWA = 0.06" and contrast floor = 1e-10 case.
            
uniform_fits_file  = "2. UC Orbit Fits.pkl"
adaptive_fits_file = "2. AC Orbit Fits.pkl"

output_file_name = "fig_7a_adaptive_vs_uniform.png"

epoch_to_compare = 8
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #

# ---------------------------------- I/O ------------------------------------ #
curr_dir = Path(__file__).resolve().parent

uniform_path  = curr_dir / uniform_fits_file
adaptive_path = curr_dir / adaptive_fits_file

df_uc = pd.read_pickle(uniform_path)
df_ac = pd.read_pickle(adaptive_path)
# --------------------------------------------------------------------------- #

uc = extract_orbit_summary(df_uc, epoch_to_compare)
ac = extract_orbit_summary(df_ac, epoch_to_compare)

common_ids = uc.index.intersection(ac.index)

uc = uc.loc[common_ids].sort_index()
ac = ac.loc[common_ids].sort_index()

print(f"Common planets with valid orbit fits at epoch {epoch_to_compare}: {len(common_ids)}")


def asymmetric_err(med, p16, p84, lower_clip=0.0):
    """
    Return asymmetric error array suitable for matplotlib errorbar.
    """
    lo = med - p16
    hi = p84 - med

    lo = np.clip(lo, lower_clip, None)
    hi = np.clip(hi, lower_clip, None)

    return np.vstack([lo, hi])


def eb_xy(ax, x, y, xerr, yerr):
    """
    Plot x vs y with both x and y asymmetric error bars.
    """
    ax.errorbar(x, y,
        xerr=xerr,
        yerr=yerr,
        fmt="o",
        ms=5,
        elinewidth=1.2,
        mec="k",
        mew=1,
        capsize=2,
        alpha=0.95,
        zorder=2
    )


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ---------------- SMA comparison ---------------- #
ax = axes[0]

x_a = uc["a_med"].values
y_a = ac["a_med"].values

xerr_a = asymmetric_err(
    uc["a_med"].values,
    uc["a_p16"].values,
    uc["a_p84"].values,
    lower_clip=1e-6
)

yerr_a = asymmetric_err(
    ac["a_med"].values,
    ac["a_p16"].values,
    ac["a_p84"].values,
    lower_clip=1e-6
)

ax.set_xlim(0.1, 9)
ax.set_ylim(0.1, 9)

ax.plot(
    [0.1, 9],
    [0.1, 9],
    "k--",
    zorder=1
)

eb_xy(ax, x_a, y_a, xerr_a, yerr_a)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("UC recovered $a$ (AU)")
ax.set_ylabel("AC recovered $a$ (AU)")

# ---------------- Eccentricity comparison ---------------- #
ax = axes[1]

x_e = uc["e_med"].values
y_e = ac["e_med"].values

xerr_e = asymmetric_err(
    uc["e_med"].values,
    uc["e_p16"].values,
    uc["e_p84"].values,
    lower_clip=0.0
)

yerr_e = asymmetric_err(
    ac["e_med"].values,
    ac["e_p16"].values,
    ac["e_p84"].values,
    lower_clip=0.0
)

emax = 0.6

emax = min(1.0, 1.05 * emax)

ax.plot([0, emax], [0, emax], "k--", zorder=1)
ax.set_xlim(0, emax)
ax.set_ylim(0, emax)

eb_xy(ax, x_e, y_e, xerr_e, yerr_e)

ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))

ax.set_xlabel("UC recovered $e$")
ax.set_ylabel("AC recovered $e$")

plt.savefig(curr_dir / output_file_name, dpi=300, bbox_inches="tight")
plt.show()


