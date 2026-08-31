"""
Artificial Posterior Samples for the Detected Planets (Part 4 of 9)
-------------------------------------------
Generate controlled period/eccentricity likelihood samples for detected planets.

For each planet, Gaussian 1-sigma errors are set to 0, 0.5, 1.01, or 2.01
times the width of the local period/eccentricity bin in the tongue plot. For a
nonzero-error case, one mock measured center is first drawn from N(truth, sigma),
then the likelihood samples are drawn from N(measured center, sigma). Draws are
truncated to the demographics analysis domain. Radius and stellar mass are kept exact,
matching the assumption in the paper.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

# Number of generated posterior samples for one detected planet
SAMPLES_PER_PLANET = 2_000

ERROR_CASES = (
    ("zero", 0.0),
    ("half-bin", 0.5),
    ("one-bin", 1.01),
    ("two-bins", 2.01)
)

# Must match the fitting domain in 5e. Fitter.jl.
PERIOD_LIMITS = (0.03, 10.0)
ECC_LIMITS    = (0.0001, 0.99)


def local_bin_width(value, edges):
    """
    Return the width of the bin containing value, including the last edge.
    """
    # Compute the left bin index
    index = np.searchsorted(edges, value, side="right") - 1
    index = np.clip(index, 0, len(edges) - 2)

    return float(edges[index + 1] - edges[index])


def truncated_normal(rng, mean, sigma, lower, upper, size):
    """
    Draw from a normal distribution truncated to [lower, upper]
    """
    samples = []

    while len(samples) < size:
        x = rng.normal(mean, sigma, size=size)
        samples.extend(x[(x >= lower) & (x <= upper)])

    return np.asarray(samples[:size])


def build_posteriors(detected, period_edges, ecc_edges, sigma_bins, center_offsets, rng):
    rows = []
    n_samples = 1 if sigma_bins == 0 else SAMPLES_PER_PLANET

    for planet in detected.itertuples(index=False):

        # Extract period and ecc
        period = float(planet.P_yr)
        ecc    = float(planet.ecc)

        period_sigma = sigma_bins * local_bin_width(period, period_edges)
        ecc_sigma    = sigma_bins * local_bin_width(ecc, ecc_edges)

        # Extract the period and ecc offsets
        period_offset, ecc_offset = center_offsets[str(planet.PlanetID)]

        # Apply it to the true perio
        measured_period = period + period_sigma * period_offset
        measured_ecc    = ecc + ecc_sigma * ecc_offset

        period_samples = truncated_normal(rng, measured_period, period_sigma, *PERIOD_LIMITS, n_samples)
        ecc_samples = truncated_normal(rng, measured_ecc, ecc_sigma, *ECC_LIMITS, n_samples)

        rows.append(pd.DataFrame({
            "PlanetID": planet.PlanetID,
            "period": period_samples,
            "ecc": ecc_samples,
            "StarName": planet.StarName,
            "Rp_REarth": float(planet.Rp_REarth),
            "M_sol": float(planet.M_sol),
            "PeriodSigma": period_sigma,
            "EccSigma": ecc_sigma,
            "TruePeriod": period,
            "TrueEcc": ecc,
            "MeasuredPeriodCenter": measured_period,
            "MeasuredEccCenter": measured_ecc,
            "SigmaInBinWidths": sigma_bins
        }))

    return pd.concat(rows, ignore_index=True)



# ------------------------------------ I/O ------------------------------------- #
curr_dir  = Path(__file__).resolve().parent

detected   = pd.read_csv(curr_dir / "5d. Detected Planets.csv")
tplot_file = curr_dir  / "5b. 4D Tongue Plot.npz"
# ------------------------------------------------------------------------------ #


with np.load(tplot_file) as tongue_plot:
    period_edges = np.asarray(tongue_plot["per_edges"], dtype=float)
    ecc_edges    = np.asarray(tongue_plot["ecc_edges"], dtype=float)


seed_sequence = np.random.SeedSequence(SEED)
center_seed, *child_seeds = seed_sequence.spawn(len(ERROR_CASES) + 1)

# Offset the true P and ecc of each planet
center_rng = np.random.default_rng(center_seed)
center_offsets = {str(planet_id): tuple(center_rng.normal(size=2)) for planet_id in detected["PlanetID"]}

for (label, sigma_bins), child_seed in zip(ERROR_CASES, child_seeds):
    samples = build_posteriors(detected, period_edges, ecc_edges, sigma_bins, center_offsets, np.random.default_rng(child_seed))

    filename = f"5d. Artificial Planet Posteriors - {label}.csv"
    samples.to_csv(curr_dir / filename, index=False)

