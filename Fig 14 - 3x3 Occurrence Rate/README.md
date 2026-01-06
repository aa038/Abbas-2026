# Figure 14 -- 3x3 OR Distribution
This folder reproduces **Figure 14** from Abbas et al. (2026, ApJ).
The script reads the posterior distributions of the fitted ORs for all the 9 IWA/contrast floors and plots them as 1D histograms.

## Files
- `3x3 Plot - OR Histograms`  -- Script to generate Fig 14
- `fig14_3x3_ORPlot.png` -- Output figure (matches Fig 14 in the paper).

## Dependencies
- Python >3.9  
- `numpy`, `pandas`, `pathlib`, `matplotlib`
- The custom built plotting package `PlotStyle` (Optional). Installation instructions are in `REQUIREMENTS.md` in the root directory.
- Requires the results of scripts `a-b` in `Data/Part II - Demographics/1-9.`: 
  1. `Xa. Observing Sim.py`                -- Script to observe the planet list for 8 epochs with the assumed IWA/contrast floor
  2. `Xb. Tongue Plot - 8 epochs.py`       -- Script to generate a 4D (`planet radius x period x ecc x star`) tongue plot for the target stars
where `X = 1,2,...9` for all the different IWA/contrast floor combinations.
  3. `Xd. List of detected planets.py`     -- Script to isolate the detected planets from the observing log
  4. `Xe. Fitter.jl`                       -- OR fitting framework in `julia` 


- These scripts will require additional dependencies:  `scipy`, `multiprocessing`, `concurrent` and the custom built packages `solve_orbit` and `Orbituary`. In addition, to run the fitter, you will have to install Julia. 
- The results of all these scripts are provided as data products in the directory `Data/Part II - Demographics/1-9.`. `3x3 Plot - OR Histograms.py` references these and can thus be run directly to reproduce Fig 14.


## Important note
- **You only need to regenerate the planet catalog and run the scripts to generate the observing logs in the Data directory you want a new random realization.**   

## How to run
From the repo root, run:

```bash
python "Fig 14 - 3x3 Occurrence Rate/3x3 Plot - OR Histograms.py"
```
This will regenerate `fig14_3x3_ORPlot.png`