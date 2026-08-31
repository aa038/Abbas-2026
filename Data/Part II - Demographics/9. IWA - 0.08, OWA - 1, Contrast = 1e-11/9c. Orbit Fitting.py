"""
Orbit fitting all planets in the observing logs
---------------------------------------------------
- Accepts an observing log, iterates through all the planets, and orbit fits all epochs of observation.
- The default number of observations for the paper is 8.
- If a planet is detected at epoch i (1 <= i <= 8), use all epochs <=i to compute orbital posteriors
- If a planet is undetected, reuse orbital posteriors from the most recent detection (since a non-detection does not significantly alter the posteriors)


Inputs:
The observing log for the current IWA x constrast floor combination

    9a. Observing Log.csv 

Outputs:
A multi-index dataframe with the orbit fits [Indices --> PlanetID, Epoch: Each planet at each observational epoch] stored as a .pkl file

    9c. Orbit Fits.pkl

Notes:
- The column names for the output multiindex Dataframe are:
    - PlanetID (Index)    -   Name of the planet (Defined in 1. Planet Catalog.csv)
    - M_sol               -   Mass of the host star in solar masses
    - L_sol               -   Luminosity of the host star in solar luminosity
    - d_pc                -   Distance to the host star in pc
    - EpochNum (Index)    -   Epoch i / 8 (1 <= i <= 8)
    - EpochTime           -   Calendar time of observation in years
    - CadenceName         -   See the variable "cadence_name" below
    - Visibility          -   Boolean that tracks if a planet was detected or not at the current epoch
    - Status              -   Fit Status (Fitted: Planet was detected and orbit fit, Extrapolated: Planet undetected, reused most recent fit, None: Planet detected but orbit fit failed OR Planet undetected and no fit to reuse)
    - orbit_df            -   7 column dataframe containing all the orbital posteriors at this epoch (epoch = EpochNum)
"""


import pandas as pd
from pathlib import Path

from orbituary.orbituary_interface import fit_orbit

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> USER-TUNABLE LIMITS <<<<<<<<<<<<<<<<<<<<<<<<<< #
obs_log_file     = "9a. Observing Log.csv"   # Observing Log from Part a.
output_file_name = "9c. Orbit Fits.pkl"
cadence_name     = "Adaptive, C: 1e-11"

import numpy as np

SEED = 42
np.random.seed(SEED)

SEP_UNCERTAINTY = 0.003             # arcsec; Should match part a.
SMA_BOUNDS      = (0.1, 15.0)       # AU; Should match part a.
NUM_ORBITS      = 500
MCMC_WALKERS    = 100
MCMC_STEPS      = 20_000
MCMC_BURNIN     = 2_000
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #

def fit_all_planets_in_log(log_path, cadence_name):
    """
    Fit Keplerian orbits for all planets in an observing log file across all epochs.
    
    Parameters
    ----------
    log_path : str or Path
        Path to the observing log CSV file 
    cadence_name : str
        A label indicating the cadence strategy 

    Returns
    -------
    A MultiIndex DataFrame with index (PlanetID, EpochNum).
    For more details, see the docstring at the top.
    """

    # Load observing log
    obs_df = pd.read_csv(log_path)

    # Create a list to collect results for each planet × epoch
    all_rows = []

    num_planets = len(obs_df["PlanetID"].unique())

    # Loop over every unique planet in the log
    for planet_idx, planet_id in enumerate(obs_df["PlanetID"].unique()):

        # Progress Tracking
        print(f"Planet ({planet_idx + 1} / {num_planets})")

        # Retrieve all the epochs of observations for the current planet
        planet_obs = obs_df[obs_df["PlanetID"] == planet_id].sort_values("LastObs").reset_index(drop=True)

        # Host star properties for orbit fitting / HZ limits
        # Read off the first row since they are repeated across rows
        m_star = planet_obs["M_sol"].iloc[0]
        d_pc   = planet_obs["d_pc"].iloc[0]
        L_sol  = planet_obs["L_sol"].iloc[0]

        # Most recent successful posterior
        last_valid_fit  = None
        previous_orbits = None

        # Process every observing epoch in chronological order
        for i, obs_row in planet_obs.iterrows():

            t          = float(obs_row["LastObs"])
            visibility = bool(obs_row["DetStatus"])

            row = {
                "PlanetID": planet_id,
                "M_sol": m_star,
                "L_sol": L_sol,
                "d_pc": d_pc,
                "EpochNum": i + 1,
                "EpochTime": t,
                "CadenceName": cadence_name,
                "Visibility": visibility
            }

            if visibility:
                # Use every detection obtained up to the current epoch
                det_so_far = planet_obs.iloc[:i + 1]
                det_so_far = det_so_far[det_so_far["DetStatus"] == 1]

                det_times = det_so_far["LastObs"].to_numpy()
                det_seps  = det_so_far["Sep"].to_numpy()
                det_pas   = det_so_far["PA"].to_numpy()

                # Astrometric uncertainties assumed in Part a
                sep_uncertainties = np.full(len(det_seps), SEP_UNCERTAINTY)
                pa_uncertainties  = np.degrees(sep_uncertainties / det_seps)

                try:
                    orbit_df, diagnostics = fit_orbit(
                        times=det_times,
                        separations=det_seps,
                        position_angles=det_pas,
                        stellar_mass=m_star,
                        distance=d_pc,
                        sep_uncertainty=sep_uncertainties,
                        pa_uncertainty=pa_uncertainties,
                        sma_bounds=SMA_BOUNDS,
                        initial_orbits=previous_orbits,
                        num_orbits=NUM_ORBITS,
                        mcmc_walkers=MCMC_WALKERS,
                        mcmc_steps=MCMC_STEPS,
                        mcmc_burnin=MCMC_BURNIN,
                        progress=False
                    )

                    row["Status"]   = "Fitted"
                    row["orbit_df"] = orbit_df

                    last_valid_fit  = orbit_df
                    previous_orbits = orbit_df

                except Exception as exc:
                    print(f"[ERROR] Fit failed for {planet_id} " f"at epoch {i + 1}: {exc}")

                    row["Status"] = "FitFailed"
                    row["orbit_df"] = None

            elif last_valid_fit is not None:
                # A nondetection reuses the most recent posterior
                row["Status"] = "Extrapolated"
                row["orbit_df"] = last_valid_fit

            else:
                # No successful fit is available yet
                row["Status"] = "None"
                row["orbit_df"] = None

            all_rows.append(row)

    # Convert list of rows into a DataFrame for storage
    # This is a MultiIndex Dataframe containing the orbit fits for every observed planet at every epoch
    # The indices are ["PlanetID", "EpochNum"]
    df = pd.DataFrame(all_rows)
    df.set_index(["PlanetID", "EpochNum"], inplace=True)

    return df


# ---------------------------------- I/O ------------------------------------ #
curr_dir = Path(__file__).resolve().parent

# Load the observing log
obs_log_path = curr_dir / obs_log_file
# --------------------------------------------------------------------------- #

# Orbit fit each planet for every epoch of observation
df_fits = fit_all_planets_in_log(obs_log_path, cadence_name)

# Save the MultiIndex DataFrame as a pickle file
output_path = curr_dir / output_file_name
df_fits.to_pickle(output_path)
print(f"Saved orbit fits to {output_path}")