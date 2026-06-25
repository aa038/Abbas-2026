"""
Plot Adaptive Metric Diagnostics
--------------------------------
Reads the saved .npz diagnostic files from the adaptive scheduler and plots
Delta(t), p_det(t), and J(t) for each scheduling decision.

Input:
    2. Diagnostics - Epoch *.npz

Output:
    fig18_AC_Diagnostics.png
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from PlotStyle import plotStyle
plotStyle()

# ---------------------------------- I/O ------------------------------------ #
curr_dir = Path(__file__).resolve().parent
files    = sorted(curr_dir.glob("*.npz"))
files    = sorted(files)

outpath  = curr_dir / "fig18_AC_Diagnostics.png"
# --------------------------------------------------------------------------- #

# ------------------------------ Figure layout ------------------------------ #
fig, axs = plt.subplots(2, 4, figsize=(14, 6.5), sharey=True, constrained_layout=False)

axs = axs.flatten()

for ax in axs:
    ax.tick_params(which="both", direction="in", top=True, right=True)
# --------------------------------------------------------------------------- #


# ---------------------------- Plot each epoch ------------------------------ #
legend_handles = None
legend_labels = None

for i, file in enumerate(files):
    ax = axs[i]

    # Load the data
    data   = np.load(file, allow_pickle=True)

    # Time, Delta(t) (or internal spread if classified), p_geom
    t_grid = data["t_grid"]
    Delta  = data["Delta"]
    p_geom = data["p_geom"]

    # Normalise Delta to [0,1]
    Delta = Delta / np.nanmax(Delta)

    # Was the planet fully classified as HZ or non-HZ
    classified = data["classified"]

    J = Delta * p_geom

    # Plot time relative to the previous observation
    last_obs = float(data["last_obs"])
    x_days = (t_grid - last_obs) * 365.25

    # Best revisit time
    idx_best = int(np.argmax(J))
    t_selected = t_grid[idx_best]
    x_selected = x_days[idx_best]


    metric_label = "Internal Spread" if classified else r"$\Delta(t)$"
    l1, = ax.plot(x_days, Delta, lw=2.0, label=metric_label)

    l2, = ax.plot(x_days, p_geom, lw=2.0, label=r"$p_{\rm geom}(t)$")
    l3, = ax.plot(x_days, J, lw=2.0, label=r"$J(t)$", color = 'red', zorder = 3)

    # Plot the chosen epoch
    ax.axvline(x_selected, ls="--", lw=1.5, color="k", alpha=0.8)

    ax.set_title(f"After epoch {i+1}", fontsize=13)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    if i >= 4:
        ax.set_xlabel("Time since last epoch (d)")

    if i % 4 == 0:
        ax.set_ylabel("Normalized value")

    legend_handles = [l1, l2, l3]
    legend_labels = [h.get_label() for h in legend_handles]
# ---------------------------------------------------------------------------- #

# ----------------------------- Legend panel --------------------------------- #
legend_ax = axs[7]
legend_ax.axis("off")

legend_ax.legend(legend_handles, legend_labels, loc="center", frameon=False, fontsize=14)

legend_ax.text(0.5, 0.25, "\nDashed line:\nBest revisit time", ha="center", va="center", fontsize=12, transform=legend_ax.transAxes)
# --------------------------------------------------------------------------- #

# ---------------------------- Final formatting ----------------------------- #
fig.subplots_adjust(left=0.07, right=0.98, bottom=0.11, top=0.90, wspace=0.18, hspace=0.30)
ax.set_ylim(-0.1, 1.1)
fig.savefig(outpath, dpi=300, bbox_inches="tight")
plt.close(fig)
# --------------------------------------------------------------------------- #