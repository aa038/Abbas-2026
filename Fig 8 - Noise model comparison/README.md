# Figure 8 -- Noise Model Comparison
This folder reproduces **Figure 8** from Abbas et al. (2026, ApJ).  
This script imposes a noise model on the fiducial architecture and compares planet yield to the baseline WA/contrast floor catalog.

## Files
- `1. Adaptive Cadence - Observing Log.py`  -- Script to generate the observing log using the fiducial architecture with a noise model
- `1. Observing Log.csv` -- Corresponding observing log
- `2. Plots.py`  -- Script to generate Fig 8
- `fig8_ETC_Comparison.png`   -- Output figure (matches Fig 8 in the paper).

## Dependencies
- Python >3.9  
- `numpy`, `pandas`, `pathlib`, `matplotlib`
- The custom built plotting package `PlotStyle` (Optional). Installation instructions are in `REQUIREMENTS.md` in the root directory.

## Important note
- **You only need to regenerate the planet catalog and run the scripts to generate the observing logs in the Data directory you want a new random realization.**   

## How to run
From the repo root, run:

```bash
python "Fig 8 - Noise model comparison/2. Plots.py"
```
This will regenerate `fig8_ETC_Comparison.png`