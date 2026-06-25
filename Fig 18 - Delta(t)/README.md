# Figure 18 -- Adaptive Cadence Diagnostics
This folder reproduces **Figure 18** from Abbas et al. (2026, ApJ), and plots the variables Delta(t), p_geom(t) and J(t) from the adaptive cadence
applied on the planet in Figure 3.

## Files
- `1. Planet Catalog w Mass.csv`  -- Planet Properties from the Illustrative Case in Fig 3.
- `2. Adaptive Cadence.py` -- Script to fit the adaptive cadence on this planet.
- `2. Diagnostics - Epoch X.npz` -- Orbit fit details for all epochs 1-8.
- `3. Diagnostic Plot.py` -- Script to plot Delta(t), p_geom(t) and J(t) for every epoch. Reproduces **Figure 18**.

## Dependencies
- Python >3.9  
- `numpy`, `pandas`, `pathlib`, `matplotlib`
- The custom built plotting package `PlotStyle` (Optional). Installation instructions are in `REQUIREMENTS.md` in the root directory.



## Important note
- **You only need to regenerate the any files in this directory if you want a new random realization.**   

## How to run
From the repo root, run:

```bash
python "Fig 18 - Delta(t).py"
```
This will regenerate `fig18_AC_Diagnostics.png`