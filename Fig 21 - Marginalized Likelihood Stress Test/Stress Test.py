"""
Plot the marginalized-likelihood stress test

Top two rows:
    Posterior median and 16-84% interval for each demographic
    parameter as a function of the imposed orbital uncertainty.

Bottom row:
    Eccentricity distributions implied by the posterior-median
    Beta parameters.

Output:
    fig21_Marginalized_Likelihood_Stress_Test.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import beta as beta_distribution

# ------------------------------------ I/O ------------------------------------- #
curr_dir = Path(__file__).resolve().parent
data_dir = curr_dir.parent / "Data" / "Poisson vs Fractional Likelihood Comparison"

poisson_file = data_dir / "5e. Fit, N = 1e4.csv"

marginalized_cases = [
    {
        "width": 0.0,
        "label": "Zero uncertainty",
        "file": data_dir / "5e. Fit, N = 1e4 - zero.csv",
        "color": "#0072B2"
    },
    {
        "width": 0.5,
        "label": "Half-bin uncertainty",
        "file": data_dir / "5e. Fit, N = 1e4 - half-bin.csv",
        "color": "#56B4E9"
    },
    {
        "width": 1.0,
        "label": "One-bin uncertainty",
        "file": data_dir / "5e. Fit, N = 1e4 - one-bin.csv",
        "color": "#E69F00"
    },
    {
        "width": 2.0,
        "label": "Two-bin uncertainty",
        "file": data_dir / "5e. Fit, N = 1e4 - two-bins.csv",
        "color": "#D55E00"
    }
]

# Read the Poisson and marginalized posteriors
poisson = pd.read_csv(poisson_file)
for case in marginalized_cases:
    case["posterior"] = pd.read_csv(case["file"])
# ------------------------------------------------------------------------------ #

# ------------------------  Helper Functions  ---------------------------------- #
def median_beta_distribution(posterior, eccentricity_grid):
    """
    Evaluate the Beta distribution at the posterior-median values
    of e_alpha and e_beta
    """
    e_alpha = np.median(posterior["e_alpha"])
    e_beta  = np.median(posterior["e_beta"])

    return beta_distribution.pdf(eccentricity_grid, e_alpha, e_beta)

def median_beta_eccentricity(posterior):
    """
    Median eccentricity of the Beta distribution plotted for a posterior
    """
    e_alpha = np.median(posterior["e_alpha"])
    e_beta  = np.median(posterior["e_beta"])

    return beta_distribution.median(e_alpha, e_beta)
# ------------------------------------------------------------------------------ #



# ---------------------------  Figure Layout ----------------------------------- #
parameters = [
    ("alpha", r"$\alpha$"),
    ("beta", r"$\beta$"),
    ("gamma", r"$\gamma$"),
    ("freq", r"$f$"),
    ("e_alpha", r"$e_{\alpha}$"),
    ("e_beta", r"$e_{\beta}$")
]

fig = plt.figure(figsize=(11, 9))

grid = fig.add_gridspec(nrows=3, ncols=3, height_ratios=[1.0, 1.0, 1.35], hspace=0.3, wspace=0.2)

parameter_axes = [
    fig.add_subplot(grid[0, 0]),
    fig.add_subplot(grid[0, 1]),
    fig.add_subplot(grid[0, 2]),
    fig.add_subplot(grid[1, 0]),
    fig.add_subplot(grid[1, 1]),
    fig.add_subplot(grid[1, 2])
]

eccentricity_axis = fig.add_subplot(grid[2, :])
# ------------------------------------------------------------------------------ #




# ------------------------------  Top two rows  ------------------------------- #
poisson_color     = "black"
poisson_linestyle = "-"

case_linestyles = {0.0: "--", 0.5: "-", 1.0: "-", 2.0: "-"}

for ax, (parameter, label) in zip(parameter_axes, parameters):

    # Construct common histogram edges for all five posterior samples
    all_samples = np.concatenate([poisson[parameter].to_numpy()] + [case["posterior"][parameter].to_numpy() for case in marginalized_cases])

    # Exclude only extreme sampling outliers when defining the displayed range
    lower_limit, upper_limit = np.percentile(all_samples, [0.2, 99.8])

    bin_edges = np.linspace(lower_limit, upper_limit, 45)

    # Ordinary Poisson posterior
    poisson_density, _ = np.histogram(poisson[parameter], bins=bin_edges, density=True)

    ax.stairs(poisson_density, bin_edges, color=poisson_color, linestyle=poisson_linestyle, linewidth=2.0, label="Poisson", zorder=2)

    # Posterior-marginalized likelihood cases
    for case in marginalized_cases:

        marginalized_density, _ = np.histogram(case["posterior"][parameter], bins=bin_edges, density=True)
        ax.stairs(marginalized_density, bin_edges, color=case["color"], linestyle=case_linestyles[case["width"]], linewidth=1.5, label=case["label"], zorder=3)

    ax.set_xlabel(label, fontsize=12)
    ax.tick_params(direction="in", which="both", top=True, right=True)
# ----------------------------------------------------------------------------- #



# -----------------------------  Bottom Row ---------------------------------- #
eccentricity = np.linspace(1.0e-4, 0.4, 600)

# Ordinary Poisson fit
poisson_eccentricity_pdf = median_beta_distribution(poisson, eccentricity)
poisson_median_ecc       = median_beta_eccentricity(poisson)

eccentricity_axis.plot(eccentricity, poisson_eccentricity_pdf, color="black", linestyle="-", linewidth=2.0)
eccentricity_axis.axvline(x = poisson_median_ecc, color="black", linestyle="--", linewidth=2.0)

# Marginalized fits
linestyles = ["--", "-", "-", "-"]

for case in marginalized_cases:

    eccentricity_pdf = median_beta_distribution(case["posterior"], eccentricity)
    median_ecc       = median_beta_eccentricity(case["posterior"])

    eccentricity_axis.plot(eccentricity, eccentricity_pdf, color=case["color"], linestyle=case_linestyles[case["width"]], linewidth=1.8)
    eccentricity_axis.axvline(x = poisson_median_ecc, color=case["color"], linestyle="--", linewidth=1.8)

eccentricity_axis.set_xlabel("Eccentricity", fontsize=12)
eccentricity_axis.set_ylabel("Probability density", fontsize=12)
eccentricity_axis.set_xlim(0.0, 0.4)
eccentricity_axis.set_ylim(bottom=0.0)

eccentricity_axis.tick_params(direction="in", which="both", top=True, right=True)
# ----------------------------------------------------------------------------- #


# --------------------------  Figure Legend  ---------------------------------- #
figure_legend = [Line2D([0], [0], color="black", linestyle="-", linewidth=2.0, label="Poisson")]

for case in marginalized_cases:
    figure_legend.append(Line2D([0], [0], color=case["color"], linestyle=case_linestyles[case["width"]], linewidth=1.8, label=case["label"]))

fig.legend(handles=figure_legend,loc="upper center",bbox_to_anchor=(0.5, 0.995), frameon=False, ncol=3, fontsize=9)

output_file = curr_dir / "fig21_Marginalized_Likelihood_Stress_Test.png"


fig.savefig(output_file, dpi=300, bbox_inches="tight")

plt.show()
# ----------------------------------------------------------------------------- #

