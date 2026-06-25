# Figure 7 -- Accuracy of SMA and Ecc after 8 epochs

This folder reproduces **Figure 7** from Abbas et al. (2026, ApJ).  
The figure compares the accuracy of the fitted sma and ecc for all planets in the fiducial catalog with their true value after 8 epochs
Applied to planets observed through the adaptive cadence strategy

## Files
- `SMA & Ecc Accuracy.py`  -- Script to generate Fig 7
- `fig7_recovered_sma_ecc.png` -- Output figure (matches Fig 7 in the paper).

## Dependencies
- Python >3.9  
- `numpy`, `pandas`, `pathlib`, `matplotlib`
- The custom built plotting package `PlotStyle` (Optional). Installation instructions are in `REQUIREMENTS.md` in the root directory.
- Requires the results of scripts `5a` and `5ab` in `5. Data / Part II - Demographics / 5. Fiducial Case - IWA - 0.06, Contrast = 1e-10`: 
  1. `5a. Observing Sim.py`                 -- Script to generate the observing log following an uniform 8-epoch cadence spaced 3 months apart for the fiducial configuration
  2. `5ab. Orbit Fits.py`                   -- Script to orbit fit each planet at each epoch in the observing log

- These scripts will require additional dependencies: `time` and the custom built packages `solve_orbit` and `orbituary`.
- Due to a file size limit, the results of the orbit fits cannot be provided as a .pkl file. However,`5ab. Orbit Fits.py` can be rerun to generate this file.

## How to run
From the repo root, run:

```bash
python "Fig 7 - SMA & Ecc Accuracy/SMA & Ecc Accuracy.py"
```
This will regenerate `fig7_recovered_sma_ecc.png` 