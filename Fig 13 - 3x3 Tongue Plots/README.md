# Figure 13 -- 3x3 tongue plots
This folder reproduces **Figure 13** from Abbas et al. (2026, ApJ).
The script marginalises over the 4D tongue plot across all 9 IWA/contrast floor combinations, and plots it as a 3x3 heatmap with the detected and non-detected planets overlaid, and colour-coded.

## Files
- `3x3 Plot - Tongue Plot with Planets`  -- Script to generate Fig 13
- `fig13_3x3_tplot.png` -- Output figure (matches Fig 12 in the paper).

## Dependencies
- Python >3.9  
- `numpy`, `pandas`, `pathlib`, `matplotlib`
- The custom built plotting package `PlotStyle` (Optional). Installation instructions are in `REQUIREMENTS.md` in the root directory.
- Requires the results of scripts `a-b` in `Data/Part II - Demographics/1-9.`: 
  1. `Xa. Observing Sim.py`                -- Script to observe the planet list for 8 epochs with the assumed IWA/contrast floor
  2. `Xb. Tongue Plot - 8 epochs.py`       -- Script to generate a 4D (`planet radius x period x ecc x star`) tongue plot for the target stars
where `X = 1,2,...9` for all the different IWA/contrast floor combinations.

- These scripts will require additional dependencies:  `scipy`, `multiprocessing`, `concurrent` and the custom built packages `solve_orbit` and `orbituary`. 
- The results of all these scripts are provided as data products in the directory `Data/Part II - Demographics/1-9.`. `3x3 Plot - Tongue Plot with Planets.py` references these and can thus be run directly to reproduce Fig 13.


## Important note
- **You only need to regenerate the planet catalog and run the scripts to generate the observing logs in the Data directory you want a new random realization.**   

## How to run
From the repo root, run:

```bash
python "Fig 13 - 3x3 Tongue Plots/3x3 Plot - Tongue Plot with Planets.py"
```
This will regenerate `fig13_3x3_tplot.png`