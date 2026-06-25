"""
Median SMA and ecc precision as a function of epoch for the exo-Earths
-------------------------------------------
This script reproduces Fig. 8 from Abbas et al. 2026.
Median SMA and ecc for all exo-Earths with 16-84th percentile spreads as a function of epoch

Output:
    fig8_sma_ecc_precision_vs_epoch
"""

# ------------------------ Scientific Assumptions --------------------------- #
# - The fractional uncertainties for sma and ecc are computed following Eqn 7 in Abbas et al. 2026
# --------------------------------------------------------------------------- #

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from pathlib import Path

# PlotStyle is a local library that needs to be installed.
# Installation intructions can be found in REQUIREMENTS.md in the root directory
from PlotStyle import plotStyle
plotStyle()

def compute_uncertainties(row):
    """
    Function to compute the 16-84 percentile spread in the sma and ecc posterior for each planet at each epoch
    """
    orbit_df = row["orbit_df"]

    if not isinstance(orbit_df, pd.DataFrame) or orbit_df.empty:
        return pd.Series({"frac_sma_unc": np.nan, "ecc_unc": np.nan})
    
    sma = orbit_df["sma"].to_numpy()
    ecc = orbit_df["ecc"].to_numpy()

    # 16-84 percentile spread of the sma and ecc posteriors
    sma_p16, sma_p84 = np.percentile(sma, [16, 84])
    ecc_p16, ecc_p84 = np.percentile(ecc, [16, 84])

    sma_med = np.median(sma)
    
    # Fractional uncertainty in sma and ecc
    frac_sma_unc = 0.5 * (sma_p84 - sma_p16) / sma_med
    ecc_unc      = 0.5 * (ecc_p84 - ecc_p16)
    
    return pd.Series({"frac_sma_unc": frac_sma_unc, "ecc_unc": ecc_unc})

# ---------------------------------- I/O ------------------------------------ #
curr_dir = Path(__file__).resolve().parent
parent_dir = curr_dir.parent
data_dir = parent_dir / "Data" / "Part II - Demographics" / "5. Fiducial Case - IWA - 0.06, Contrast = 1e-10"

fit_files = {
    "Uniform: $1\\times10^{-10}$": data_dir / "5a. Orbit Fits.pkl"
}
# --------------------------------------------------------------------------- #

# Figure Setup
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
plt.subplots_adjust(wspace=0.25)

for label, path in fit_files.items():
    # This is a MultiIndex Dataframe containing the orbit fits for every observed planet at every epoch
    # The indices are ["PlanetID", "EpochNum"]
    # For more details on this file, see the docstring in Data/Part I - Observing Cadence/4. Orbit Fits.py
    df = pd.read_pickle(path)

    # Identify planets with valid posteriors at epoch 8
    idx = pd.IndexSlice
    df_epoch8 = df.loc[idx[:, 8], :]

    valid_epoch8 = df_epoch8["orbit_df"].apply(lambda x: isinstance(x, pd.DataFrame) and len(x) > 0)

    valid_planet_ids = df_epoch8[valid_epoch8].index.get_level_values("PlanetID").unique()

    # Restrict full time series to the same final fitted planet sample
    df = df.loc[df.index.get_level_values("PlanetID").isin(valid_planet_ids)].copy()

    print(f"{label}: {len(valid_planet_ids)} planets with valid epoch-8 posteriors")

    # Fractional uncertainty in sma and ecc for all planets at all epochs, enforced using the apply() function
    df[["frac_sma_unc", "ecc_unc"]] = df.apply(compute_uncertainties, axis=1)

    # Group by epoch number (1-8)
    grouped = df.groupby(level="EpochNum")[["frac_sma_unc", "ecc_unc"]]

    # Compute the median fractional uncertainty in sma and ecc at each epoch,
    # and the 16-84th percentile spread in the fractional uncertainties
    median = grouped.median()
    p16    = grouped.quantile(0.16)
    p84    = grouped.quantile(0.84)

    epochs = median.index.values

    # Fractional uncertainty in SMA
    ax = axes[0]
    ax.plot(epochs, median["frac_sma_unc"], marker="o", label=label)
    ax.fill_between(epochs, p16["frac_sma_unc"], p84["frac_sma_unc"], alpha=0.2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    # Fractional uncertainty in Eccentricity
    ax = axes[1]
    ax.plot(epochs, median["ecc_unc"], marker="o", label=label)
    ax.fill_between(epochs, p16["ecc_unc"], p84["ecc_unc"], alpha=0.2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

axes[0].set_xlabel("Epoch number")
axes[0].set_ylabel("$\\Delta a\,/\,a$")
axes[0].set_ylim([1e-3, 1])
axes[0].set_xlim([1, 8])
axes[1].set_xlabel("Epoch number")
axes[1].set_ylabel("$\\Delta e$")
axes[1].set_ylim([1e-3, 1])
axes[1].set_xlim([1, 8])
axes[0].set_yscale('log')
axes[1].set_yscale('log')

plt.savefig(curr_dir / "fig8_sma_ecc_precision_vs_epoch", dpi=300, bbox_inches = 'tight')
plt.show()
