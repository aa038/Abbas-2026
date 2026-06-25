"""
Plot the ETC Comparison 
-------------------------------------------
This script reads the observing logs for the noise model limited and baseline working angle + contrast floor catalogs generated
in the previous script and compares the number of unique planet detections as a function of epoch.

Input:
    1. Observing Log - 3 hr.csv       
    1. Observing Log - 12 hr.csv
    1. Observing Log - 24 hr.csv                         # Each of these contain the baseline catalog within them

Output:
    2. fig10_ETC_Comparison                              # Matches Fig 10 in the paper
"""

import numpy as np
import pandas as pd              
import matplotlib.pyplot as plt
from pathlib import Path

from PlotStyle import plotStyle
plotStyle()

# ------------------------------------ I/O ------------------------------------- #
# Get the path to the current directory
curr_dir = Path(__file__).resolve().parent

# Observing logs generated for different ETs
log_paths = {
    "$t_{\\rm exp} = 3\,hr$": curr_dir / "1. Observing Log - 3 hr.csv",
    "$t_{\\rm exp} = 12\,hr$": curr_dir / "1. Observing Log - 12 hr.csv",
    "$t_{\\rm exp} = 24\,hr$": curr_dir / "1. Observing Log - 24 hr.csv"
}

ideal_label = "Working angle + contrast"
# ------------------------------------------------------------------------------- #


def cumulative_unique_detections(log, det_col, epoch_col="NObs", planet_col="PlanetID", max_epoch=8):
    """
    Count cumulative unique planets detected as a function of epoch.

    A planet contributes once, at its first detection epoch.
    Later detections of the same planet do not increase the cumulative count.
    Later non-detections do not remove the planet.
    """
    # Remove NaNs
    tmp = log[[planet_col, epoch_col, det_col, 'NDet']].copy()
    tmp = tmp.dropna(subset=[planet_col, epoch_col, det_col, 'NDet'])

    # Extract both the epochs 1-8 and the detection status at each of them
    tmp[epoch_col] = tmp[epoch_col].astype(int)
    tmp[det_col]   = tmp[det_col].astype(int)

    # Find the first detected epoch for each planet in the log
    first_detection_epoch = (tmp[tmp[det_col] == 1].groupby(planet_col)[epoch_col].min())

    epochs = np.arange(1, max_epoch + 1)

    cumulative = [int((first_detection_epoch <= epoch).sum()) for epoch in epochs]
    
    return cumulative
# ------------------------------------------------------------------------------- #

# ----------------------- Load logs and compute curves -------------------------- #

all_curves = {}


# Compute the cumulative no. of unique detections for the noise-limited logs
for label, path in log_paths.items():
    log = pd.read_csv(path)
    print(path)
    etc_curve = cumulative_unique_detections(log, det_col="DetStatus_ETC")
    print(etc_curve)
    all_curves[label] = etc_curve


# Use the first log to compute the ideal curve.
# This should be identical for all exposure-time cases.
first_log = pd.read_csv(next(iter(log_paths.values())))
ideal_curve = cumulative_unique_detections(first_log, det_col="DetStatus_Ideal")
print(ideal_curve)
# ------------------------------------------------------------------------------- #

# --------------------- Plot cumulative unique detections ----------------------- #
fig, ax = plt.subplots(figsize=(8, 5))

epochs = np.arange(1, 9)

ax.plot(epochs, ideal_curve, marker="o", linewidth=2.5, label=ideal_label)

for label, curve in all_curves.items():
    ax.plot(epochs, all_curves[label], marker="o", linewidth=2, label=label)

ax.set_xlabel("Epoch")
ax.set_ylabel("Cumulative unique planets detected")
ax.set_xticks(np.arange(1, 9))
ax.legend()
fig.savefig("fig17_ETC_Comparison.png", dpi=300, bbox_inches = 'tight')
# ------------------------------------------------------------------------------- #