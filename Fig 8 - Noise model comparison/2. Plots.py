"""
Plot the ETC Comparison 
-------------------------------------------
This script reads the observing logs for the noise model limited and baseline working angle + contrast floor catalogs generated
in the previous script and compares the number of unique planet detections as a function of epoch.

Input:                    
    1. Observing Log 

Output:
    2. fig8_ETC_Comparison                              # Matches Fig 8 in the paper
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
log_path = curr_dir / "1. Observing Log.csv"

det_columns = {
    "$t_{\\rm exp} = 3\\,hr$": "DetStatus_ETC_3hr",
    "$t_{\\rm exp} = 12\\,hr$": "DetStatus_ETC_12hr",
    "$t_{\\rm exp} = 24\\,hr$": "DetStatus_ETC_24hr"
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
log = pd.read_csv(log_path)

# ------------------- Identify true exo-Earths -------------------- #
hz_inner = np.sqrt(log["L_sol"] / 1.78)
hz_outer = np.sqrt(log["L_sol"] / 0.32)

periastron = log["SMA_AU"] * (1.0 - log["ecc"])
apastron   = log["SMA_AU"] * (1.0 + log["ecc"])
in_hz      = (periastron >= hz_inner) & (apastron <= hz_outer)


earth_radius = (log["Rp_REarth"] > 0.8) & (log["Rp_REarth"] < 1.4)

log["IsExoEarth"] = in_hz & earth_radius
exoearth_log      = log[log["IsExoEarth"]]
# ----------------------------------------------------------------- #

all_curves = {}
exoEarths  = {}

for label, det_col in det_columns.items():

    # All planets
    curve = cumulative_unique_detections(log, det_col=det_col)
    print(label, curve)
    all_curves[label] = curve

    # Detected true exo-Earths only
    exo_curve = cumulative_unique_detections(exoearth_log, det_col=det_col)
    exoEarths[label] = exo_curve

ideal_curve = cumulative_unique_detections(log, det_col="DetStatus_Ideal")
print("Ideal:", ideal_curve)

ideal_exoearth_curve = cumulative_unique_detections(exoearth_log, det_col="DetStatus_Ideal")
print(ideal_exoearth_curve)
# ------------------------------------------------------------------------------- #



# --------------------- Plot cumulative unique detections ----------------------- #
fig, ax = plt.subplots(figsize=(8, 5))

epochs = np.arange(1, 9)

ideal_line, = ax.plot(epochs, ideal_curve, marker="o", linewidth=2.5, label=ideal_label)
ax.plot(epochs, ideal_exoearth_curve, marker="o", linewidth=2.5, ls="--", color=ideal_line.get_color())

for label, curve in all_curves.items():
    line, = ax.plot(epochs, all_curves[label], marker="o", linewidth=2, label=label)
    ax.plot(epochs, exoEarths[label], marker="o", linewidth=2.5, ls="--", color=line.get_color())

ax.set_xlabel("Epoch")
ax.set_ylabel("Cumulative unique planets detected")
ax.set_xticks(np.arange(1, 9))
ax.legend()
fig.savefig(curr_dir / "fig8_ETC_Comparison.png", dpi=300, bbox_inches = 'tight')
# ------------------------------------------------------------------------------- #