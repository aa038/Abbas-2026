# Figure 21 -- Marginalized Likelihood Stress Test

This folder reproduces **Figure 21** from Abbas et al. (2026, ApJ). The figure compares the ordinary binned-Poisson occurrence rate fit, which uses the exact injected planet properties, with posterior marginalized fits using artificial period/eccentricity uncertainties of zero, one-half, one, and two local demographic cell widths.

The upper six panels show the one-dimensional posterior distributions for the six demographic parameters. The lower panel shows the eccentricity distributions implied by the posterior-median Beta-distribution parameters. The zero-uncertainty result provides the numerical limiting test and should closely reproduce the ordinary Poisson posterior. The wider cases test how progressively degraded orbital information propagates into the population inference.

## Files

- `Stress Test.py` -- Script to generate Figure 21 from the five MCMC posterior files
- `fig21_Marginalized_Likelihood_Stress_Test.png` -- Output figure (matches Figure 21 in the paper)

## Input data

The script reads the following supplied products from `Data/Poisson vs Fractional Likelihood Comparison`:

- `5e. Fit, N = 1e4.csv` -- Ordinary binned-Poisson fit using exact injected planet properties
- `5e. Fit, N = 1e4 - zero.csv` -- Posterior-marginalized fit with zero-width period/eccentricity posteriors
- `5e. Fit, N = 1e4 - half-bin.csv` -- Posterior-marginalized fit with 0.5-bin uncertainties
- `5e. Fit, N = 1e4 - one-bin.csv` -- Posterior-marginalized fit with approximately one-bin uncertainties
- `5e. Fit, N = 1e4 - two-bins.csv` -- Posterior-marginalized fit with approximately two-bin uncertainties

These data products are provided, so `Stress Test.py` can be run directly to reproduce the figure. See `Data/Poisson vs Fractional Likelihood Comparison/README.md` for the complete data-generation and fitting workflow.

## Dependencies

- Python >3.9
- `numpy`, `pandas`, `matplotlib`, `scipy`, `pathlib`

## Important note

- **You only need to regenerate the MCMC posterior files if you want a new random realization or a different likelihood stress test.**
- Radius and host-star mass are held exact in the artificial posteriors; only period and eccentricity uncertainty are varied.
- The nonzero uncertainty labels represent the Gaussian 1-sigma width relative to the local completeness-grid bin width.

## How to run

From the repo root, run:

```bash
python "Fig 21 - Marginalized Likelihood Stress Test/Stress Test.py"
```

This will regenerate `fig21_Marginalized_Likelihood_Stress_Test.png`.
