# Figure 15 -- Scaled OR
This folder reproduces **Figure 15** from Abbas et al. (2026, ApJ), and plots the planets realised from an exaggerated SAG13 OR overlaid on the fiducial tongue plot, and the CDFs for the associated mock surveys.

## Files
- `Plot Planets on Tongue Plot.py`  -- Script to generate the tongue plot with the overlaid planets
- `fig15a_EOR_tplot.png`            -- Output figure (matches Fig 15a in the paper).
- `CDF Plot.py`                     -- Script to plot the CDFs for the mock surveys
- `fig15b_EOR_CumulativeDist.png`   -- Output figure (matches Fig 15b in the paper).

## Dependencies
- Python >3.9  
- `numpy`, `pandas`, `pathlib`, `matplotlib`
- The custom built plotting package `PlotStyle` (Optional). Installation instructions are in `REQUIREMENTS.md` in the root directory.
- Requires the results of scripts `a-e` in `Data/Part II - Demographics/Scaled Occurrence Rates\5. IWA - 0.06, OWA - 1, Contrast = 1e-10`: 
  1. `5a. Observing Sim.py`                -- Script to observe the planet list for 8 epochs with the assumed IWA/contrast floor
  2. `5b. Tongue Plot - 8 epochs.py`       -- Script to generate a 4D (`planet radius x per x ecc x star`) tongue plot for the target stars
  3. `5d. List of detected planets.py`     -- Script to isolate the detected planets from the observing log
  4. `5e. Fitter.jl`                       -- OR fitting framework in `julia`
  5. `5g. Mock Survey - Code.py`           -- Script to generate mock surveys from the MCMC posteriors
- Also requires the planet catalogue generated with the scaled occurrence rate:
  1. `1. SAG13 Planet Population - Radius + Orb Params.py` -- Script to generate the planet catalog using the scaled OR
  2. `2. Planet Radius to Mass and Albedo.py`              -- Add mass and albedo to the planets generated in `1.`
   

- These scripts will require additional dependencies:  `scipy`, `multiprocessing`, `concurrent` and the custom built packages `solve_orbit` and `orbituary`. In addition, to run the fitter, you will have to install Julia. 
- The result of `5g. Mock Survey - Code.py` is **NOT PROVIDED** since it exceeds the 200MB file size limit. To remake `fig15b_EOR_CumulativeDist.png`, `5g. Mock Survey - Code.py` will have to be **RUN FIRST**. It can be run as is without any changes.


## Important note
- **You only need to regenerate the planet catalog and run the scripts to generate the observing logs in the Data directory you want a new random realization.**   

## How to run
From the repo root, run:

```bash
python "Fig 15 - Extreme OR - Tongue Plot + Cum Dist/Plot Planets on Tongue Plot.py"
python "Fig 15 - Extreme OR - Tongue Plot + Cum Dist/CDF Plot.py"
```
This will regenerate the figures `fig15a_EOR_tplot.png` and `fig15b_EOR_CumulativeDist.png` respectively.