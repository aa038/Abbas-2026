# Figure 20 -- Corner Plot Comparison
This folder reproduces **Figure 20** from Abbas et al. (2026, ApJ).
The script reads the posterior distributions obtained through a Poisson likelihood and a posterior marginalised likelihood with zero-width posteriors, and plots them for comparison

## Files
- `Corner Plot - Code.py`  -- Script to generate Fig 20
- `fig20_CornerPlotComparison.png` -- Output figure (matches Fig 20 in the paper).

## Dependencies
- Python >3.9  
- `numpy`, `pandas`, `pathlib`, `matplotlib`
- The custom built plotting package `PlotStyle` (Optional). Installation instructions are in `REQUIREMENTS.md` in the root directory.
- Requires the results of scripts `a-e` in `Data\Poisson vs Fractional Likelihood Comparison`: 
  1. `5a. Observing Sim.py`                             -- Script to observe the planet list for 8 epochs with the assumed IWA/contrast floor
  2. `5b. Tongue Plot - 8 epochs.py`                    -- Script to generate a 4D (`planet radius x period x ecc x star`) tongue plot for the target stars
  3. `5d. List of detected planets.py`                  -- Script to isolate the detected planets from the observing log
  4. `5d2. Generate Artificial Planet Posteriors.py`    -- Script to generate artificial Gaussian posteriors for the planets around their true values
  5. `5e. Fitter - Fractional.jl`                       -- OR fitting framework in `julia`, using a posterior marginalized likelihood
  6. `5e. Fitter - Poisson.jl`                          -- OR fitting framework in `julia`, using a Poisson likelihood


- These scripts will require additional dependencies:  `scipy`, `multiprocessing`, `concurrent` and the custom built package `orbituary`. In addition, to run the fitter, you will have to install Julia. 
- The results of all these scripts are provided as data products in the directory ``Data\Poisson vs Fractional Likelihood Comparison`. `Corner Plot - Code.py` references these and can thus be run directly to reproduce Fig 20.


## Important note
- **You only need to regenerate the planet catalog and run the scripts to generate the observing logs in the Data directory you want a new random realization.**   

## How to run
From the repo root, run:

```bash
python "Fig 20 - Corner Plot Comparison/Corner Plot - Code.py"
```
This will regenerate `fig20_CornerPlotComparison.png`