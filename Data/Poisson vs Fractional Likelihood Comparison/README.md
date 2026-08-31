# Data/Poisson vs Fractional Likelihood Comparison -- Likelihood stress test

This folder contains the controlled likelihood comparison used to verify the demographics-fitting framework. The ordinary binned Poisson likelihood assigns each detected planet to the cell containing its exact injected properties. The posterior-marginalized ("fractional") likelihood instead averages each planet's contribution over artificial period/eccentricity posteriors with widths of zero, one-half, approximately one, or approximately two local grid cells.

The zero-width case is the numerical limiting test: it should reproduce the ordinary Poisson result. The wider cases show how increasing orbital uncertainty changes the inferred occurrence-rate posterior. For the derivation and discussion, see Appendix `Marginalized Likelihood` in Abbas et al. 2026.

## Files

- `5a. Observing Sim.py`                         -- Script to observe the SAG13 planet population for 8 epochs using the fiducial HWO configuration and a fixed cadence
- `5a. Observing Log.csv`                        -- Visit-by-visit observing log, including detections (Result of `5a. Observing Sim.py`)
- `5b. Tongue Plot - 8 epochs.py`                -- Script to generate the 4D (`planet radius x period x eccentricity x star`) completeness map
- `5b. 4D Tongue Plot.npz`                       -- 4D completeness map and its bin centers/edges (Result of `5b. Tongue Plot - 8 epochs.py`)
- `5c. Plot Planets on Tongue Plot.py`           -- Script to marginalize the completeness map and overlay the simulated planets
- `5c. Tongue Plot with Planets.png`             -- Marginalized completeness map with detected and non-detected planets (Result of `5c. Plot Planets on Tongue Plot.py`)
- `5d. List of detected planets.py`              -- Script to isolate planets detected at least once
- `5d. Detected Planets.csv`                     -- Exact injected properties of the detected planets (Result of `5d. List of detected planets.py`)
- `5d2. Generate Artificial Planet Posteriors.py` -- Script to generate controlled period/eccentricity posterior samples for every detected planet
- `5d. Artificial Planet Posteriors - zero.csv`  -- Zero-width posteriors, with one sample at the exact injected values
- `5d. Artificial Planet Posteriors - half-bin.csv` -- Posteriors with 1-sigma widths equal to 0.5 local period/eccentricity bin widths
- `5d. Artificial Planet Posteriors - one-bin.csv`  -- Posteriors with 1-sigma widths equal to 1.01 local bin widths
- `5d. Artificial Planet Posteriors - two-bins.csv` -- Posteriors with 1-sigma widths equal to 2.01 local bin widths
- `5e. Fitter - Poisson.jl`                      -- Ordinary binned-Poisson occurrence-rate fitter using the exact detected-planet properties
- `5e. Fit, N = 1e4.csv`                         -- MCMC samples from the ordinary Poisson fit
- `5e. Fitter - Fractional.jl`                   -- Posterior-marginalized occurrence-rate fitter using one artificial-posterior file at a time
- `5e. Fit, N = 1e4 - {zero/half-bin/one-bin/two-bins}.csv` -- MCMC samples from the four posterior-marginalized fits
- `5f. Corner Plot - Code.py`                    -- Script to generate a corner plot from a selected MCMC result
- `5f. Corner Plot - {Poisson/Zero/Half-bin/One-bin/Two-bin}.png` -- Preserved corner plots for the five likelihood/error cases

## Likelihood cases

- **Poisson:** Each planet contributes an integer count to the bin containing its exact injected radius, period, eccentricity, and host-star mass.
- **Zero:** Each planet has a delta-function period/eccentricity posterior. This is the posterior-marginalized implementation's zero-uncertainty limit and should agree with the Poisson fit.
- **Half-bin, one-bin, and two-bins:** Each planet is retained as one catalog event, but its likelihood contribution is averaged over a progressively wider artificial period/eccentricity posterior. Radius and stellar mass remain exact.

All artificial posterior sets use `SEED = 42`. Nonzero cases contain 2,000 samples per detected planet and are truncated to the fitting domain. The labels describe the Gaussian 1-sigma width relative to the local completeness-grid bin width; `one-bin` and `two-bins` use factors of 1.01 and 2.01 to avoid boundary ambiguities.

## Dependencies

- Python >3.9
- `numpy`, `pandas`, `scipy`, `astropy`, `matplotlib`
- Julia with `CSV`, `NPZ`, `Glob`, `Optim`, `Turing`, `DataFrames`, `Statistics`, `MCMCChains`, `Distributed`, `Distributions`, `SpecialFunctions`, `StatsBase`, `StatsFuns`, and `ForwardDiff`
- The custom built orbit solver package `orbituary`. Installation instructions are in `REQUIREMENTS.md` in the root directory.
- The bundled `Data/Forecaster` radius-mass model
- The custom built plotting package `PlotStyle` (Optional). Installation instructions are in `REQUIREMENTS.md` in the root directory.

## How to run

From the repo root, generate the shared observing and completeness products in order:

```bash
python "Data/Poisson vs Fractional Likelihood Comparison/5a. Observing Sim.py"
python "Data/Poisson vs Fractional Likelihood Comparison/5b. Tongue Plot - 8 epochs.py"
python "Data/Poisson vs Fractional Likelihood Comparison/5c. Plot Planets on Tongue Plot.py"
python "Data/Poisson vs Fractional Likelihood Comparison/5d. List of detected planets.py"
python "Data/Poisson vs Fractional Likelihood Comparison/5d2. Generate Artificial Planet Posteriors.py"
```

Run the ordinary Poisson fit with:

```bash
julia "Data/Poisson vs Fractional Likelihood Comparison/5e. Fitter - Poisson.jl"
```

For each posterior-marginalized case, edit `posterior_samples_file` and the output filename near the bottom of `5e. Fitter - Fractional.jl`, then run:

```bash
julia "Data/Poisson vs Fractional Likelihood Comparison/5e. Fitter - Fractional.jl"
```

Likewise, select the desired fit CSV in `5f. Corner Plot - Code.py`, run the script, and rename its `5f. Corner Plot.png` output to preserve each case.

## Important note

- **You only need to regenerate the observing simulation and completeness map if you want a new realization.** For paper reproduction, use the supplied data products.
- `5b. Tongue Plot - 8 epochs.py` and both Julia fitters are parallelized. Set `N_CORES` in each script to suit your machine before running.
- Regenerating the full workflow is computationally intensive. The completeness map and MCMC fits may take hours on a multicore machine.
- For a new underlying planet population, rerun **ALL** scripts in `Data/Planet Generation` **FIRST**.
