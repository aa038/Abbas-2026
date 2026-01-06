# Figure 7 -- Accuracy of SMA and Ecc after 8 epochs

This folder reproduces **Figure 7** from Abbas et al. (2026, ApJ).  
The figure compares the accuracy of the fitted sma and ecc for exo-Earths with their true value after 8 epochs
Applied to planets observed through the adaptive cadence strategy

## Files
- `SMA & Ecc Accuracy.py`  -- Script to generate Fig 7
- `fig7_recovered_sma_2.5e-11.png` -- Output figure for the 2.5e-11 contrast floor (matches Fig 7a in the paper).
- `fig7_recovered_sma_4e-11.png` -- Output figure for the 4e-11 contrast floor (matches Fig 7b in the paper).

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
- The results of all these scripts are provided as data products in the directory `Data/Part I - Observing Cadence`. `SMA & Ecc Accuracy.py` references these and can thus be run directly to reproduce Fig 7.


## Important note
- **You only need to regenerate the planet catalog and run the scripts to generate the observing logs in the Data directory you want a new random realization.**   

## How to run
From the repo root, run:

```bash
python "Fig 7 - SMA & Ecc Accuracy/SMA & Ecc Accuracy.py"
```
This will regenerate `fig7_recovered_sma_2.5e-11.png` OR `fig7_recovered_sma_4e-11.png` depending on the user's choice of input file