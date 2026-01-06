# Figure 8 -- Average fractional uncertainty in SMA and ecc as a function of epoch
This folder reproduces **Figure 8** from Abbas et al. (2026, ApJ).  
The figure shows the average fractional uncertainty in sma and ecc and the 16-84 percentile spread across all exo-Earths as a function of epoch
Applied to planets observed through the adaptive cadence strategy

## Files
- `Fractional error per epoch.py`  -- Script to generate Fig 8
- `fig8_sma_ecc_precision_vs_epoch.png` -- Output figure for the 2.5e-11 contrast floor (matches Fig 8 in the paper).

## Dependencies
- Python >3.9  
- `numpy`, `pandas`, `pathlib`, `matplotlib`
- The custom built plotting package `PlotStyle` (Optional). Installation instructions are in `REQUIREMENTS.md` in the root directory.
- Requires the results of scripts `1-4` in `Data/Part I - Observing Cadence`: 
  1. `1. HZ Candidates - Planet Catalog.py` -- Script to generate the exo-Earth catalog from the full planet catalog (`Data/Planet Generation/SAG13 Planet Catalog.csv`)
  2. `2. UC Observing Log Generator.py`     -- Script to apply a uniform cadence to the planets in `1. Planet Catalog.csv`
  3. `3. AC Observing Log Generator.py`     -- Script to apply the adaptive cadence to the planets in `1. Planet Catalog.csv` 
  4. `4. Orbit Fits.py`                     -- Script to orbit fit each planet at each epoch in the observing log

- These scripts will require additional dependencies: `time` and the custom built packages `solve_orbit` and `orbituary`.
- The results of all these scripts are provided as data products in the directory `Data/Part I - Observing Cadence`. `Fractional error per epoch.py` references these and can thus be run directly to reproduce Fig 8.


## Important note
- **You only need to regenerate the planet catalog and run the scripts to generate the observing logs in the Data directory you want a new random realization.**   

## How to run
From the repo root, run:

```bash
python "Fig 8 - AC Fractional error per epoch/Fractional error per epoch.py"
```
This will regenerate `fig8_sma_ecc_precision_vs_epoch.png`