"""
HZ Confidence vs Calendar Time
-------------------------------------------
This script reproduces Fig. 6 from Abbas et al. 2026.

The number of planets classified as HZ at each observational epoch 
across cadences and contrast floors, as a function of calendar time.

Output:
    fig6_HZ_Confidence_Time.png

Notes:
- 2 cadence strategies: Uniform and Adaptive (See Sec 3.3, Abbas et al.)
- 2 contrast floors: 2.5e-11 and 4e-11
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
# Fig 6 compares the number of confirmed exo-Earths across cadences at each epoch (1-8), as a function of clock time.
# To plot this we need, we:
#   - Iterate over the results of each cadence strategy
#   - The results file for each cadence strategy contains the orbital posteriors to each planet at every observational epoch
#   - Extract sma and ecc for each posterior and compute what fraction is consistent with being in the HZ (confidence)
#   - Count the number of planets at each epoch that are classified as HZ with 68% and 95% confidence
#   - The script also computes classifications that are false positives (not shown in the paper)

# Sets to store data for the inset plots
series_68 = {}
series_95 = {}

for i, (label, path) in enumerate(cadences.items()):

    # Read the pickle file
    # This is a MultiIndex Dataframe containing the orbit fits for every observed planet at every epoch
    # The indices are ["PlanetID", "EpochNum"]
    # For more details on this file, see the docstring in Data/Part I - Observing Cadence/4. Orbit Fits.py
    df_fits = pd.read_pickle(path)

    # Sort the rows by the time of observation
    epoch_times = sorted(df_fits["EpochTime"].unique())
    
    # HZProb computes our confidence that a planet is in the HZ at each epoch,
    # by computing how many of the orbits at each epoch are consisent with being fully in the HZ
    df_fits["HZProb"] = np.nan

    # Loop through each row of the dataframe and assign a HZ confidence to each planet at each epoch using the orbit fits
    for idx, row in df_fits.iterrows():
        # The "orbit_df" column contains a dataframe with the full set of orbital posteriors at that epoch
        # This is a datafram inside the parent multiIndex dataframe
        orbit_df = row["orbit_df"]
        L_star   = row["L_sol"]

        # Compute the fraction of orbital posteriors at this epoch consistent with being fully in the HZ
        if orbit_df is not None:
            df_fits.at[idx, "HZProb"] = compute_hz_probability(orbit_df, L_star)

    # Unique sorted times
    epoch_times = sorted(df_fits["EpochTime"].unique())

    # Initialize lists to:
    #   1. Store the times of observations
    #   2. HZ TPs + FPs at this time
    #   3. HZ TPs only
    times_68          = []
    counts_68         = []
    true_positives_68 = []

    times_95          = []
    counts_95         = []
    true_positives_95 = []

    # Track which planets are computed to be in the HZ at each observation time
    true_positive_68  = set()
    false_positive_68 = set()
    true_positive_95  = set()
    false_positive_95 = set()

    # Group data by time for efficiency
    grouped = df_fits.groupby("EpochTime")

    # --------------------  Loop through ALL observation times  ------------------- #
    # - This is a two-step process:
    # -     1. Identify which of the planets are classified as HZ with 68% and 95% confidence (Both TPs and FPs)
    # -     2. Store this list of planets and their times of observations to a list for plotting
    for t in epoch_times:
        # All the planets observed at this time
        rows       = grouped.get_group(t)
        # Using get_level_values() to extract the planet IDs since PlanetID is an index
        planet_ids = rows.index.get_level_values("PlanetID")

        for pid, hz_prob in zip(planet_ids, rows["HZProb"]):

            # Check if the planet is truly in the HZ (True Positive)
            is_true_hz = planets_df.loc[pid, "HZ"]

            # ----------- HZ confidence check ----------- #
            # - There are three possibilities:
            # -     1. >= X% HZ confidence and the planet is a TP
            # -        Then, add it to the set of TPs and remove it from the set of FPs
            # -     2. >= X% HZ confidence and the planet is NOT a TP
            # -        Then, add it to the set of FPs and remove it from the set of TPs
            # -     3. < X% confidence
            # -        Then the planet is not a positive of either kind. Remove from the set of both TPs and FPs
            # -     Sets will not throw an error if a planetID is not present in that set, meaning discard() can be used safely

            # ------- 68% threshold ------- #
            if hz_prob >= 0.68:
                # True Positive
                if is_true_hz:
                    true_positive_68.add(pid)
                    false_positive_68.discard(pid)
                # False Positive
                else:
                    false_positive_68.add(pid)
                    true_positive_68.discard(pid)
            # Negative
            else:
                true_positive_68.discard(pid)
                false_positive_68.discard(pid)
            # ----------------------------- #

            # ------- 95% threshold ------- #
            if hz_prob >= 0.95:
                # True positive
                if is_true_hz:
                    true_positive_95.add(pid)
                    false_positive_95.discard(pid)
                # False positive
                else:
                    false_positive_95.add(pid)
                    true_positive_95.discard(pid)
            # Negative
            else:
                true_positive_95.discard(pid)
                false_positive_95.discard(pid)
            # ----------------------------- #
            # ------------------------------------------- #
            
        # Save this time to the list of times (for plotting)
        times_68.append(t)
        times_95.append(t)


        # Save the list of TPs and FPs at this time t
        # 68% confidence
        counts_68.append(len(true_positive_68 | false_positive_68))  # TP + FP
        true_positives_68.append(len(true_positive_68))              # TP only
        # 95% confidence
        counts_95.append(len(true_positive_95 | false_positive_95))  # TP + FP
        true_positives_95.append(len(true_positive_95))              # TP only

    # Store the final results for this cadence for the inset plot
    series_68[label] = (np.array(times_68), np.array(true_positives_68))
    series_95[label] = (np.array(times_95), np.array(true_positives_95))

    # ------------  Plot the final results  -------------------- #
    #axs[0].step(times_68, counts_68, label=f"{label} (TP + FP)")
    axs[0].step(times_68, true_positives_68, label=f"{label}", ls = ls[i])

    #axs[1].step(times_95, counts_95, label=f"{label} (TP + FP)")
    axs[1].step(times_95, true_positives_95, label=f"{label}", ls = ls[i])
    # ---------------------------------------------------------- #

    # ---------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #

# ------------------------------
# Final plot formatting
# ------------------------------
axs[0].set_title("≥ 68% HZ Confidence")
axs[0].set_xlabel("Time (yr)")
axs[0].set_ylabel("Number of Planets")
axs[0].set_ylim([0, 25])
axs[0].set_xlim([2035, 2045])
axs[0].yaxis.set_major_locator(ticker.MultipleLocator(5))
axs[0].yaxis.set_minor_locator(ticker.MultipleLocator(1))
axs[0].xaxis.set_major_locator(ticker.MultipleLocator(5))
axs[0].xaxis.set_minor_locator(ticker.MultipleLocator(1))
#axs[0].legend(loc='upper right')

axs[1].set_title("≥ 95% HZ Confidence")
axs[1].set_xlabel("Time (yr)")
axs[1].set_ylim([0, 25])
axs[1].set_xlim([2035, 2045])
axs[1].yaxis.set_major_locator(ticker.MultipleLocator(5))
axs[1].yaxis.set_minor_locator(ticker.MultipleLocator(1))
axs[1].xaxis.set_major_locator(ticker.MultipleLocator(5))
axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(1))
#axs[1].legend()

axs[0].text(2035.1, 22, "(a)")
axs[1].text(2035.1, 22, "(b)")

axs[0].axhline(y = 24, ls = "--", color = 'red')
axs[1].axhline(y = 24, ls = "--", color = 'red')

# ----------------------  Inset Plot  ------------------------- #
# x- and y-axis limits for the inset 
x1, x2 = 2035.0, 2037.0
y1, y2 = 0, 25

# Inset for ≥68%
axins0 = inset_axes(axs[0], width="45%", height="45%", loc="lower right", borderpad=2)

# Inset for ≥95%
axins1 = inset_axes(axs[1], width="45%", height="45%", loc="lower right", borderpad=2)

# Loop through all the cadences to extract the times and the corresponding number of HZ planets (TPs)
for i, label in enumerate(cadences.keys()):
    t68, tp68 = series_68[label]
    t95, tp95 = series_95[label]
    axins0.step(t68, tp68, ls=ls[i])
    axins1.step(t95, tp95, ls=ls[i])

# Plot formatting
for axins in [axins0, axins1]:
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axins.yaxis.set_major_locator(ticker.MultipleLocator(5))
    axins.tick_params(labelsize=8)
# ------------------------------------------------------------- #

fig.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(curr_dir / "fig6_HZ_Confidence_Time.png", dpi=300, bbox_inches = "tight")
plt.show()