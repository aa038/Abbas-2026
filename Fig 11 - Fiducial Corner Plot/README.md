# Figure 11 -- Fiducial Corner Plot
This folder reproduces **Figure 11** from Abbas et al. (2026, ApJ), generating a corner plot for the results of the MCMC occurrence rate fit.

## Files
- `Corner Plot.py`  -- Script to generate Fig 11
- `fig11_CornerPlot.png` -- Output figure for the 2.5e-11 contrast floor (matches Fig 10 in the paper).

## Dependencies
- Python >3.9  
- `numpy`, `pandas`, `pathlib`, `matplotlib`
- The custom built plotting package `PlotStyle` (Optional). Installation instructions are in `REQUIREMENTS.md` in the root directory.
- Requires the results of scripts `a-b` in `Data/Part II - Demographics/5. Fiducial Case - IWA - 0.06, Contrast = 1e-10`: 
  1. `5a. Observing Sim.py`                -- Script to observe the planet list for 8 epochs with the assumed IWA/contrast floor
  2. `5b. Tongue Plot - 8 epochs.py`       -- Script to generate a 4D (`planet radius x period x ecc x star`) tongue plot for the target stars
  3. `5d. List of detected planets.py`     -- Script to isolate the detected planets from the observing log
  4. `5e. Fitter.jl`                       -- OR fitting framework in `julia`


- These scripts will require additional dependencies:  `scipy`, `multiprocessing`, `concurrent` and the custom built packages `solve_orbit` and `orbituary`. In addition, to run the fitter, you will have to install Julia. 
- The results of all these scripts are provided as data products in the directory ``Data/Part II - Demographics/5. Fiducial Case - IWA - 0.06, Contrast = 1e-10`. `Corner Plot.py` references these and can thus be run directly to reproduce Fig 11.


## Important note
- **You only need to regenerate the planet catalog and run the scripts to generate the observing logs in the Data directory you want a new random realization.**   

## How to run
From the repo root, run:

```bash
python "Fig 11 - Fiducial Corner Plot/Corner Plot.py"
```
This will regenerate `fig11_CornerPlot.png`