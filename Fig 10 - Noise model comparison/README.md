# Figure 10 -- Noise Model Comparison
This folder reproduces **Figure 10** from Abbas et al. (2026, ApJ).  
This script imposes a noise model on the fiducial architecture and compares planet yield to the baseline WA/contrast floor catalog.

## Files
- `1. Observing Sim.py`  -- Script to generate the observing log using the fiducial architecture with a noise model
- `1. Observing Log - X hr.csv` -- Observing log with X hours per target
- `2. Plots.py`  -- Script to generate Fig 10
- `fig10_ETC_Comparison.png`   -- Output figure (matches Fig 10 in the paper).

## Dependencies
- Python >3.9  
- `numpy`, `pandas`, `pathlib`, `matplotlib`
- The custom built plotting package `PlotStyle` (Optional). Installation instructions are in `REQUIREMENTS.md` in the root directory.

## Important note
- **You only need to regenerate the planet catalog and run the scripts to generate the observing logs in the Data directory you want a new random realization.**   

## How to run
From the repo root, run:

```bash
python "Fig 10 - Noise model comparison/2. Plots.py"
```
This will regenerate `fig10_ETC_Comparison.png`