# Figure 4a and 4b -- Cadence Comparison for a single planet: Uniform vs Adaptive

This folder reproduces **Figure 4a and 4b** from Abbas et al. (2026, ApJ).  
The scripts generate a planet with `R = 1 R_E`, `P = 2 yr`, `e = 0.2` and `i = 88.06 deg` around `HIP 72905 (M = 1.08 M_sol)`

## Files
- `5. Orbit Fitting.py`       -- Script to generate Fig 3a or 3b, depending on the choice of observing log in the script.
- `fig4a_OrbitalFit_SC.png`   -- Output figure for uniform cadence (matches paper).
- `fig4b_OrbitalFit_AC.png`   -- Output figure for adaptive cadence (matches paper).

## Dependencies
- Python >3.9  
- `numpy`, `pandas`, `pathlib`, `matplotlib`
- The custom built orbit fitting package `Orbituary`. Installation instructions are in `REQUIREMENTS.md` in the root directory.
- The custom built plotting package `PlotStyle` (Optional). Installation instructions are in `REQUIREMENTS.md` in the root directory.
- Requires the results of scripts `1-4`: 
  1.  `1. SAG13 Planet Population - Radius + Orb Params.py` -- Generates a single planet around `HIP 72905`.
  2.  `2. Planet Radius to Mass and Albedo.py`              -- Assigns this planet a mass and albedo
  3.  `3. Uniform Cadence.py`                               -- Observe this planet across 8 epochs using an uniform "3-month" cadence 
  4.  `4. Adaptive Cadence.py`                              -- Observe this planet using an adaptice cadence
- The results of all these scripts are also provided as data files in the directory. `5. Orbit Fitting.py` can thus be run directly.
- If you want a new planet, rerun **all** of these. 
- These scripts will require additional dependencies: `astropy`, `time` and the custom built package `solve_orbit`.
  

## Important note
- **You only need to regenerate the planet if you want a new random realization.**   
- If you rerun the scripts with a new seed, you will get a different planet population.

## How to run
From the repo root, run:

```bash
python "Fig 4 - Illustrative System - Inclined Eccentric Planet/5. Orbit Fitting.py"
```

This will regenerate `fig4a_OrbitalFit_SC.png` **OR** `fig4b_OrbitalFit_AC.png`, **depending on your choice of observing log**. The instructions for swapping observing logs (uniform vs adaptive) are clearly stated in `5. Orbit Fitting.py`.