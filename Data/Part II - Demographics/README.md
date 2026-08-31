# Data/Part II - Demographics/5. Fiducial Case - IWA - 0.06, Contrast = 1e-10

This folder contains the demographics analysis for a HWO-like configuration with `OWA = 1"` and 9 combinations of IWA and OWA (`IWA = [0.04", 0.06", 0.08"], contrast floor = [1e-9, 1e-10, 1e-11]`).
For an in-depth description of the demographics analysis, see Sec 5, Abbas et al. 2026.

## Files
Each directory is organised as follows (`X = 1-9`):
- `Xa. Observing Sim.py`                -- Script to observe the planet list for 8 epochs with the assumed IWA/contrast floor
- `Xa. Observing Log.csv`               -- Observing log for each epoch of observation of all the planets (Result of `5a. Observing Sim.py`)
- `Xb. Tongue Plot - 8 epochs.py`       -- Script to generate a 4D (`planet radius x period x ecc x star`) tongue plot for the target stars
- `Xb. 4D Tongue Plot.npz`              -- 4D tongue plot (Result of `5b. Tongue Plot - 8 epochs.py`)
- `Xc. Plot Planets on Tongue Plot.py`  -- Script to plot the observed planets on the marginalised tongue plot (`radius x period`)
- `Xc. Tongue Plot with Planets.png`    -- Result of `5c. Plot Planets on Tongue Plot.py`
- `Xd. List of detected planets.py`     -- Script to isolate the detected planets from the observing log
- `Xd. Detected Planets.csv`            -- List of detected planets (Result of `5d. List of detected planets.py`)
- `Xe. Fitter.jl`                       -- OR fitting framework in `julia`
- `Xe. Fit, N = 1e4.csv`                -- Full MCMC posteriors for all parameters in the OR model (Result of `5e. Fitter.jl`)
- `Xf. Corner Plot - Code.py`           -- Script to generate the corner plot for the MCMC posteriors
- `Xf. Corner Plot.png`                 -- Corner Plot (Result of `5f. Corner Plot - Code.py`)
- `Xg. Mock Survey - Code.py`           -- Script to generate mock surveys from the MCMC posteriors
- `Xg. Mock Survey.csv`                 -- Mock surveys (Result of `5g. Mock Survey - Code.py`)
- `Xh. Survey Results.py`               -- Script to plot the CDFs generated from the mock surveys
- `Xh. Survey Results.png`              -- Plot of CDFs (Result of `5h. Survey Results.py`)
- `Xi. Compare Input Ecc with Output.py`-- Script to plot the input Rayleigh and fitted Beta ecc distributions
- `Xi. Input Ecc vs Output Ecc.png`     -- Result of `5i. Compare Input Ecc with Output.py`

In-depth docs and code is available **ONLY** for the fiducial configuration i.e. `X=5`

