# Figure 14 -- Fiducial Tongue Plot
This folder reproduces **Figure 14** from Abbas et al. (2026, ApJ).  
This script marginalises over the fiducial 4D tongue plot over the exo-Earth regime, and plots it as a heatmap with the detected and non-detected exo-Earths overlaid, and colour-coded.

## Files
- `Plot Planets on Tongue Plot.py`  -- Script to generate Fig 14
- `fig14_ExoEarth_tplot.png` -- Output figure (matches Fig 14 in the paper).

## Dependencies
- Python >3.9  
- `numpy`, `pandas`, `pathlib`, `matplotlib`
- The custom built plotting package `PlotStyle` (Optional). Installation instructions are in `REQUIREMENTS.md` in the root directory.
- Requires the results of scripts `a-b` in `Data/Part II - Demographics/Exo-Earth Centric ORs/5. Fiducial Case - IWA - 0.06, Contrast = 1e-10`: 
  1. `5a. Adaptive Cadence - Observing Log.py`  -- Script to observe the planet list for 8 epochs with the assumed IWA/contrast floor
  2. `5b. Adaptive Tongue Plot - 8 epochs.py`   -- Script to generate a 4D (`planet radius x period x ecc x star`) tongue plot for the target stars


- These scripts will require additional dependencies:  `scipy`, `multiprocessing`, `concurrent` and the custom built package `orbituary`.
- The results of all these scripts are provided as data products in the directory `Data/Part II - Demographics/Exo-Earth Centric ORs/5. Fiducial Case - IWA - 0.06, Contrast = 1e-10`. `Plot Planets on Tongue Plot.py` references these and can thus be run directly to reproduce Fig 14.


## Important note
- **You only need to regenerate the planet catalog and run the scripts to generate the observing logs in the Data directory you want a new random realization.**   

## How to run
From the repo root, run:

```bash
python "Fig 14 - Exo-Earth Tongue Plot/Plot Planets on Tongue Plot.py"
```
This will regenerate `fig14_ExoEarth_tplot.png`