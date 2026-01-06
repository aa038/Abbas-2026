# Data/Part II - Demographics/5. Fiducial Case - IWA - 0.06, Contrast = 1e-10

This folder contains the demographics analysis for a HWO-like configuration with `IWA = 0.06"`, `contrast floor = 1e-10`, and `OWA = 1"`. We consider this the fiducial HWO configuration in the paper.
For an in-depth description of the demographics analysis, see Sec 5, Abbas et al. 2026.

## Files
- `5a. Observing Sim.py`                -- Script to observe the planet list for 8 epochs with the assumed IWA/contrast floor
- `5a. Observing Log.csv`               -- Observing log for each epoch of observation of all the planets (Result of `5a. Observing Sim.py`)
- `5b. Tongue Plot - 8 epochs.py`       -- Script to generate a 4D (`planet radius x period x ecc x star`) tongue plot for the target stars
- `5b. 4D Tongue Plot.npz`              -- 4D tongue plot (Result of `5b. Tongue Plot - 8 epochs.py`)
- `5c. Plot Planets on Tongue Plot.py`  -- Script to plot the observed planets on the marginalised tongue plot (`radius x period`)
- `5c. Tongue Plot with Planets.png`    -- Result of `5c. Plot Planets on Tongue Plot.py`
- `5d. List of detected planets.py`     -- Script to isolate the detected planets from the observing log
- `5d. Detected Planets.csv`            -- List of detected planets (Result of `5d. List of detected planets.py`)
- `5e. Fitter.jl`                       -- OR fitting framework in `julia`
- `5e. Fit, N = 1e4.csv`                -- Full MCMC posteriors for all parameters in the OR model (Result of `5e. Fitter.jl`)
- `5f. Corner Plot - Code.py`           -- Script to generate the corner plot for the MCMC posteriors
- `5f. Corner Plot.png`                 -- Corner Plot (Result of `5f. Corner Plot - Code.py`)
- `5g. Mock Survey - Code.py`           -- Script to generate mock surveys from the MCMC posteriors
- `5g. Mock Survey.csv`                 -- Mock surveys (Result of `5g. Mock Survey - Code.py`)
- `5h. Survey Results.py`               -- Script to plot the CDFs generated from the mock surveys
- `5h. Survey Results.png`              -- Plot of CDFs (Result of `5h. Survey Results.py`)
- `5i. Compare Input Ecc with Output.py`-- Script to plot the input Rayleigh and fitted Beta ecc distributions
- `5i. Input Ecc vs Output Ecc.png`     -- Result of `5i. Compare Input Ecc with Output.py`
  
## Dependencies
- Python >3.9  
- `numpy`, `pandas`, `pathlib`, `matplotlib`, `time`, `scipy`, `multiprocessing`, `concurrent`
- The custom built plotting package `solve_orbit`. Installation instructions are in `REQUIREMENTS.md` in the root directory.
- The custom built plotting package `PlotStyle` (Optional). Installation instructions are in `REQUIREMENTS.md` in the root directory.
  

## Important note
- **You only need to regenerate these scripts if you want a new random realization.**   
- For a new random realization, rerun **ALL** scripts in `Data/Planet Generation` **FIRST**.

## How to run
From the repo root, run:

```bash
python "Data/Part II - Demographics/SCRIPT_NAME.py"
```

To run the `julia` script:

```bash
julia include("Data\\Part II - Demographics\\5e. Fitter.jl")
```

**WARNING**: Scripts `5b.` and `5e.` have been parallelised to run on 10 cores. Modify this to suit your machine before running.
