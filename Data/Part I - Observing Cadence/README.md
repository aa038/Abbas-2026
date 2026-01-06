# Data/Part I - Observing Cadence -- Comparison of observing cadences

This folder contains the exo-Earth catalog (subset of the full planet catalog), observing logs for both cadences for two contrast floors, and the associated orbit fits. For an in-depth description of the observing cadence strategies, see Sec 3.3, Abbas et al. 2026.

## Files
- `1. HZ Candidates - Planet Catalog.py` -- Script to generate the exo-Earth catalog from the full planet catalog (`Data/Planet Generation/SAG13 Planet Catalog.csv`)
- `1. Planet Catalog.csv`                -- Exo-Earth Catalog (Output from `1. HZ Candidates - Planet Catalog.py`).
- `2. UC Observing Log Generator.py`     -- Script to apply a uniform cadence to the planets in `1. Planet Catalog.csv`
- `2. UC Observing Log - 2.5e-11.csv`    -- Observing log for the uniform cadence with a coronagraph contrast floor of 2.5e-11
- `2. UC Observing Log - 4e-11.csv`      -- Observing log for the uniform cadence with a coronagraph contrast floor of 4e-11
- `3. AC Observing Log Generator.py`     -- Script to apply the adaptive cadence to the planets in `1. Planet Catalog.csv` 
- `3. AC Observing Log - 2.5e-11.csv`    -- Observing log for the adaptive cadence with a coronagraph contrast floor of 2.5e-11
- `3. AC Observing Log - 4e-11.csv`      -- Observing log for the adaptive cadence with a coronagraph contrast floor of 4e-11
- `4. Orbit Fits.py`                     -- Script to orbit fit each planet at each epoch in the observing log
- `4. {UC/AC} - {2.5/4}e-11.pkl`         -- Orbit fits to all the observing logs (For more detail on the .pkl file, read the docstring in `4. Orbit Fits.py`).
  
## Dependencies
- Python >3.9  
- `numpy`, `pandas`, `pathlib`, `matplotlib`, `time`
- The custom built orbit fitting package `Orbituary`. Installation instructions are in `REQUIREMENTS.md` in the root directory.
- The custom built orbit solver package `solve_orbit`. Installation instructions are in `REQUIREMENTS.md` in the root directory.
- The custom built plotting package `PlotStyle` (Optional). Installation instructions are in `REQUIREMENTS.md` in the root directory.
  

## Important note
- **You only need to regenerate these scripts if you want a new random realization.**   
- For a new random realization, rerun **ALL** scripts in `Data/Planet Generation` **FIRST**.

## How to run
From the repo root, run:

```bash
python "Data/Part I - Observing Cadence/1. HZ Candidates - Planet Catalog.py"
python "Data/Part I - Observing Cadence/2. UC Observing Log Generator.py"
python "Data/Part I - Observing Cadence/3. AC Observing Log Generator.py"
python "Data/Part I - Observing Cadence/4. Orbit Fits.py"
```

**WARNING**: Scripts 2-4 are computationally intensive (ranging from a few hours to a day on a single core), and should **ONLY** be run for new analyses. Casual users should use the data products provided.

## Citations
- SAG13 occurrence prescription: Belikov et al. 2017 (ExoPAG SAG13).  
- Radius-mass relation: Chen & Kipping 2017 (Forecaster).
- HZ Definition: Kopparappu et. al. 2013
- Orbit fitting framework adapts frameworks from Blunt et al. 2017 (OFTI) and Nielsen et al. 2014