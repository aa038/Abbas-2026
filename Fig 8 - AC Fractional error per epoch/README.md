# Figure 8 -- Average fractional uncertainty in SMA and ecc as a function of epoch
This folder reproduces **Figure 8** from Abbas et al. (2026, ApJ).  
The figure shows the average fractional uncertainty in sma and ecc and the 16-84 percentile spread across all planets observed under a uniform 8 epoch 3-month cadence with the fiducial architecture.
## Files
- `Fractional error per epoch.py`  -- Script to generate Fig 8
- `fig8_sma_ecc_precision_vs_epoch.png` -- Output figure for the fiducial architecture (matches Fig 8 in the paper).

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
python "Fig 8 - AC Fractional error per epoch/Fractional error per epoch.py"
```
This will regenerate `fig8_sma_ecc_precision_vs_epoch.png`