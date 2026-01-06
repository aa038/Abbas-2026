"""
Orbit fitting all planets in the observing logs
---------------------------------------------------
- Accepts an observing log, iterates through all the planets, and orbit fits all epochs of observation.
- The default number of observations for the paper is 8.
- If a planet is detected at epoch i (1 <= i <= 8), use all epochs <=i to compute orbital posteriors
- If a planet is undetected, reuse orbital posteriors from the most recent detection (since a non-detection does not significantly alter the posteriors)


Inputs:
Any of the observing logs for either cadence

    2. UC Observing Log - 2.5e-11.csv

    OR

    2. UC Observing Log - 4e-11.csv

    OR

    2. AC Observing Log - 2.5e-11.csv

    OR

    2. AC Observing Log - 4e-11.csv



Outputs:
A multi-index dataframe with the orbit fits [Indices --> PlanetID, Epoch: Each planet at each observational epoch] stored as a .pkl file

    4. {Cadence Name} Fits - {Contrast Floor}.pkl

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

- For a guide on how to read this .pkl file and access the data, see the associated scripts with Figs 5, 6, 7 or 8
"""


import pandas as pd
from pathlib import Path

from orbituary.orbituary_interface import fit_orbit

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> USER-TUNABLE LIMITS <<<<<<<<<<<<<<<<<<<<<<<<<< #
obs_log_file = "2. UC Observing Log - 2.5e-11.csv"   # Observing Log from Part 2/3
#obs_log_file = "3. AC Observing Log - 2.5e-11.csv"
#obs_log_file = "2. UC Observing Log - 4e-11.csv"
#obs_log_file = "3. AC Observing Log - 4e-11.csv"

output_file_name = "4. UC Fits - 2.5e-11.pkl"
#output_file_name = "4. UC Fits - 4e-11.pkl"
#output_file_name = "4. AC Fits - 2.5e-11.pkl"
#output_file_name = "4. AC Fits - 4e-11.pkl"

cadence_name  = "UC: 3 months, C: 2.5e-11"
#cadence_name  = "UC: 3 months, C: 4e-11"
#cadence_name  = "AC: 3 months, C: 2.5e-11"
#cadence_name  = "AC: 3 months, C: 4e-11"
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #

def fit_all_planets_in_log(log_path: str | Path, cadence_name: str) -> pd.DataFrame:
    """
    Fit Keplerian orbits for all planets in an observing log file across all epochs.
    
    Parameters
    ----------
    log_path : str or Path
        Path to the observing log CSV file (e.g. "2. UC Observing Log - 2.5e-11.csv")
    cadence_name : str
        A label indicating the cadence strategy (e.g. 'Simple_1mo', 'Adaptive_2mo')

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
        planet_obs = obs_df[obs_df["PlanetID"] == planet_id].sort_values("LastObs")
        
        # All the observation times and detection status at these times
        all_times        = planet_obs["LastObs"].values
        visibility_flags = planet_obs["DetStatus"].astype(bool).values

        # Subset of detected observations for orbit fitting
        det_obs = planet_obs[planet_obs["DetStatus"] == 1]

        # Astrometric data of detections for orbit fitting
        det_times = det_obs["LastObs"].values
        det_seps  = det_obs["Sep"].values
        det_pas   = det_obs["PA"].values

        # Host star properties for orbit fitting / HZ limits
        # Read off the first row since they are repeated across rows
        m_star = planet_obs["M_sol"].iloc[0]
        d_pc   = planet_obs["d_pc"].iloc[0]
        L_sol  = planet_obs["L_sol"].iloc[0]

        try:
            # Run orbit fitting using visible detections only
            fit_results = fit_orbit(
                det_times, det_seps, det_pas,
                m_star, d_pc, 
                mcmc_steps=20000,
                mcmc_walkers=100,
                mcmc_burnin=2000,
                all_epochs=True,
                progress_bar=True
            )
        except Exception as e:
            print(f"[ERROR] Fit failed for {planet_id}: {e}")
            continue
        
        # - We just orbit fit detections (to simulate real-world operations)
        # - But we plot epochs of both detections and non-detections
        # - If the current epoch is a detection, orbit fit all available epochs including the current one
        # - Otherwise, use the most recent orbit fit (since a "non-detection" cannot change orbital posteriors)
        det_idx = 0            # Variable to iterate through the detected epochs, since they are the only ones with orbit fits
        last_valid_fit = None  # Orbital posteriors for the last detection. Initialised to None

        for i, t in enumerate(all_times):
            visibility = visibility_flags[i]

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

            # If planet is detected, save the latest orbit fit
            if visibility:
                this_orbit_df = fit_results[det_idx]

                row["Status"]   = "Fitted"
                row["orbit_df"] = this_orbit_df
                last_valid_fit  = this_orbit_df

                det_idx += 1

            # If the planet was undetected, but there are valid previous fits, use those
            elif last_valid_fit is not None:
                # Reuse last known posteriors 
                row["Status"]   = "Extrapolated"
                row["orbit_df"] = last_valid_fit

            else:
                # No prior detections to fall back on
                row["Status"]   = "None"
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