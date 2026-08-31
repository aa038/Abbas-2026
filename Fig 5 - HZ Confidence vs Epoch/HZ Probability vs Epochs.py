"""
HZ Confidence vs Epoch Number
-------------------------------------------
This script reproduces Fig. 5 from Abbas et al. 2026.

the number of planets classified as HZ at each observational epoch 
across cadences and contrast floors.

Output:
    fig5_HZ_Confidence_Epoch.png

Notes:
- 2 cadence strategies: Uniform and Adaptive (See Sec 3.3, Abbas et al.)
- 2 contrast floors: 2.5e-11 and 4e-11
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

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


# ---------------------------------- I/O ------------------------------------ #
curr_dir = Path(__file__).resolve().parent
parent_dir = curr_dir.parent
data_dir = parent_dir / "Data" / "Part I - Observing Cadence"

# - 4 sets of orbit fits for:
# -     2x cadence strategies (uniform vs adaptive; See Abbas et al. 2026 Sec 3.3) 
# -     2x contrast floors (2.5e-11 and 4e-11; based on Mamajek & Stapelfeldt 2024)
# - Each .pkl file contains all the orbit fits to all planets across all observed epochs (i.e. 8 total)
cadences = {
    "Adaptive Cadence: C = $2.5\\times10^{-11}$": data_dir / "4. AC Fits - 2.5e-11.pkl",
    "Adaptive Cadence: C = $4\\times10^{-11}$"  : data_dir / "4. AC Fits - 4e-11.pkl",
    "Uniform Cadence: C = $2.5\\times10^{-11}$" : data_dir / "4. UC Fits - 2.5e-11.pkl",
    "Uniform Cadence: C = $4\\times10^{-11}$"   : data_dir / "4. UC Fits - 4e-11.pkl"   
}

# Plotting choice for each contrast floor x cadence combination
ls = ["-", "--", "-.", ":"]

# Planet catalog
planets_df_dir = data_dir / "1. Planet Catalog.csv"
planets_df     = pd.read_csv(planets_df_dir)
planets_df.set_index("PlanetID", inplace=True)
# --------------------------------------------------------------------------- #


# ----------------------------- Figure Set-up ------------------------------- #
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
# --------------------------------------------------------------------------- #


# --------------- Process all the observations across cadences -------------- #
# Fig 5 compares the number of confirmed exo-Earths across cadences at each epoch (1-8).
# To plot this we need, we:
#   - Iterate over the results of each cadence strategy
#   - The results file for each cadence strategy contains the orbital posteriors to each planet at every observational epoch
#   - Extract sma and ecc for each posterior and compute what fraction is consistent with being in the HZ (confidence)
#   - Count the number of planets at each epoch that are classified as HZ with 68% and 95% confidence
#   - The script also computes classifications that are false positives (not shown in the paper)

epochs = range(1, 9)    # We have observed each planet a max of 8 times. Change this if you alter the number of observations

for i, (label, path) in enumerate(cadences.items()):

    # Read the pickle file 
    # This is a MultiIndex Dataframe containing the orbit fits for every observed planet at every epoch
    # The indices are ["PlanetID", "EpochNum"]
    # For more details on this file, see the docstring in Data/Part I - Observing Cadence/4. Orbit Fits.py
    df_fits = pd.read_pickle(path)

    # HZProb computes our confidence that a planet is in the HZ at each epoch,
    # by computing how many of the orbits at each epoch are consisent with being fully in the HZ
    df_fits["HZProb"] = np.nan

    # Loop through each row of the dataframe
    # i.e. Loop through each planet at each detection epoch
    for idx, row in df_fits.iterrows():

        # The "orbit_df" column contains a dataframe with the full set of orbital posteriors at that epoch
        # This is a datafram inside the parent multiIndex dataframe
        orbit_df = row["orbit_df"]
        L_star   = row["L_sol"]

        # Compute the fraction of orbital posteriors at this epoch consistent with being fully in the HZ
        if orbit_df is not None:
            df_fits.at[idx, "HZProb"] = compute_hz_probability(orbit_df, L_star)


    # Empty lists to store:
    #   - Number of planets classified as HZ with 68% and 95% confidence (TP + FP)
    #   - Number of planets correctly classified as HZ with 68% and 95% confidence (TP only)
    counts_68, counts_95           = [], []
    counts_68_true, counts_95_true = [], []

    # Loop through observational epochs 1-8
    for epoch in epochs:

        # Get all the planets observed at this epoch numner
        # For epoch = 1, get all the first observations of every planet
        rows = df_fits[df_fits.index.get_level_values("EpochNum") == epoch]

        # Get PlanetIDs for these rows
        planet_ids = rows.index.get_level_values("PlanetID")

        # Look up their true HZ status
        true_hz_mask = planets_df.loc[planet_ids, "HZ"].values.astype(bool)

        # Count TPs
        n_68_true = ((rows["HZProb"] >= 0.68) & true_hz_mask).sum()
        n_95_true = ((rows["HZProb"] >= 0.95) & true_hz_mask).sum()
        counts_68_true.append(n_68_true)
        counts_95_true.append(n_95_true)

        # Count TPs + FPs
        n_68 = (rows["HZProb"] >= 0.68).sum()
        n_95 = (rows["HZProb"] >= 0.95).sum()
        counts_68.append(n_68)
        counts_95.append(n_95)

        # Identify the false positives that still remain by epoch 8
        if epoch == 8:
            # HZProb ≥ threshold AND not truly in the HZ → False Positives
            fp_68_mask = (rows["HZProb"] >= 0.68) & (~true_hz_mask)
            fp_95_mask = (rows["HZProb"] >= 0.95) & (~true_hz_mask)

            fp_68_ids = planet_ids[fp_68_mask]
            fp_95_ids = planet_ids[fp_95_mask]

            print(f"\nFalse Positives at epoch 8 for {label}:")
            print(f"  68% confidence: {list(fp_68_ids)}")
            print(f"  95% confidence: {list(fp_95_ids)}")

    # Testing
    #if label == "Uniform Cadence: C = $2.5\\times10^{-11}$":
    #    counts_68_true[-1] += 1
    #    counts_95_true[-1] += 1

    axs[0].plot(epochs, counts_68_true, marker='o', label=f"{label}", ls = ls[i])
    axs[1].plot(epochs, counts_95_true, marker='s', label=f"{label}", ls = ls[i])

    axs[0].text(0.7, 22, "(a)")
    axs[1].text(0.7, 22, "(b)")
# --------------------------------------------------------------------------- #


# --------------------------- Final Formatting ------------------------------ #
axs[0].set_title("≥ 68% HZ Confidence")
axs[0].set_xlabel("Epoch Number")
axs[0].set_ylabel("Number of Planets")
axs[0].set_xticks(epochs)
axs[0].set_ylim([0, 25])
axs[0].set_xlim([0.5, 8.5])
axs[0].yaxis.set_major_locator(ticker.MultipleLocator(5))
axs[0].yaxis.set_minor_locator(ticker.MultipleLocator(1))
axs[0].legend()

axs[1].set_title("≥ 95% HZ Confidence")
axs[1].set_xlabel("Epoch Number")
axs[1].set_xticks(epochs)
axs[1].set_ylim([0, 25])
axs[1].set_xlim([0.5, 8.5])
axs[1].yaxis.set_major_locator(ticker.MultipleLocator(4))
axs[1].yaxis.set_minor_locator(ticker.MultipleLocator(2))

axs[0].axhline(y = 24, ls = "--", color = 'red')
axs[1].axhline(y = 24, ls = "--", color = 'red')

fig.tight_layout(rect=[0, 0, 1, 0.95])
# --------------------------------------------------------------------------- #

plt.savefig(curr_dir / "fig5_HZ_Confidence_Epoch.png", dpi=300, bbox_inches = 'tight')