# Figure 6 -- HZ Confidence vs Calendar Time for exo-Earths

This folder reproduces **Figure 6** from Abbas et al. (2026, ApJ).  
The figure compares the number of planets classified as HZ as a function of observation time across cadences and contrast floors.

## Files
- `HZ Probability vs Time.py`  -- Script to generate Fig 6
- `fig6_HZ_Confidence_Time.png` -- Output figure (matches Fig 6 in the paper).

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
- The results of all these scripts are provided as data products in the directory `Data/Part I - Observing Cadence`. `HZ Probability vs Time.py` references these and can thus be run directly to reproduce Fig 5.


## Important note
- **You only need to regenerate the planet catalog and run the scripts to generate the observing logs in the Data directory you want a new random realization.**   
- `HZ Probability vs Time.py` also computes the false positive rate (planets incorrectly classified as HZ with X% confidence). Only the true positives are shown in the paper (since there were no false positives in our study), but users can experiment with displaying the false positive rate as well with minimal plotting changes to the script.

## How to run
From the repo root, run:

```bash
python "Fig 6 - HZ Confidence vs Time/HZ Probability vs Time.py"
```
This will regenerate `fig6_HZ_Confidence_Time.png`.