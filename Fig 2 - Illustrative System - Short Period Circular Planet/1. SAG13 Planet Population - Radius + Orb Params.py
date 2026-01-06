"""
Illustrative System Generator (Part 1 of 5)
-------------------------------------------
One planet version of the more detailed 1. SAG13 Planet Population - Radius and Orb Params.py in Data/Planet Generation.

Generates a single short-period, nearly face-on planet on a circular orbit 
around one star (the first entry of HWO Stars.csv). This illustrative system 
is used for Figure 2 (uniform vs adaptive cadence demo).

Output:
    1. Planet Catalog.csv

Notes:
- Fixed planet parameters (Rp=3.4 R_E, P=1 yr, ecc=0, inc~0) for simplicity.
- Randomized orbital angles (Mean Anomaly, Angle of Periastron, Longitude of Ascending Node).
- Produces exactly one planet around one star for visualization and orbit-fitting examples.
"""

# ------------------------ Scientific Assumptions --------------------------- #
# - ONE planet generated around ONE star
# - Radius, Period, Ecc and Inc are pre-chosen
# - Uniform priors on orbital angles
# --------------------------------------------------------------------------- #

import pandas as pd
from pathlib import Path
import numpy as np

# --------------------------- Reproducibility -------------------------- #
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)
# ---------------------------------------------------------------------- #

# ----------------------------- I/O ------------------------------------ #
curr_dir = Path(__file__).resolve().parent
parent_dir = curr_dir.parent
data_dir = parent_dir / "Data" / "Planet Generation"

# Load the star dataframe (ExEP Mission Star List - Mamajek & Stapelfeldt 2024)
stars = pd.read_csv(data_dir / "HWO Stars.csv")

# We choose the first star in the list as our reference
# You could choose any star at random
stars = stars[0:1]
# ---------------------------------------------------------------------- #


# ------------------------  PLANET GENERATION  ------------------------- #
# Empty list to store planet details
all_planets = []

# Left as a loop ober stars for easy scaling to multiple stars if desired
for i, row in stars.iterrows():
    
    # Obtain the mass of the star
    Mstar = row['M']

    # 1 planet per star, for visualisation purposes
    n_planets = 1

    # Define planet characteristics and orbital params
    # Pre-chosen parameters
    Rp           = 3.4
    P            = 1
    ecc          = 0
    cosi         = 0.99
    inc          = np.arccos(cosi)
    # Randomly assigned parameters
    mean_anomaly = rng.uniform(0, 1)
    aop          = rng.uniform(0, 2*np.pi)
    pan          = rng.uniform(0, 2*np.pi)


    # Compute SMA using Kepler's Third Law
    sma = (P**2 * Mstar) ** (1/3)

    # Convert Mean Anomaly to Epoch of Periastron Passage (2035 chosen as an arbitrary future reference)
    epp = 2035 - mean_anomaly / (2 * np.pi) * P

    # Package the planet into a dictionary, and save as .csv
    hip_id = row["HDName"]
    for j in range(n_planets):
        planet = {
            "PlanetID"  : f"{hip_id}_{j}",
            "HDName"    : hip_id,
            "Spec"      : row['SpecType'],
            "d_pc"      : row["Dist"],
            "L_sol"     : row["L"],
            "M_sol"     : row["M"],
            "Rp_Rearth" : Rp,
            "sma_AU"    : sma,
            "P_yr"      : P,
            "ecc"       : ecc,
            "inc_rad"   : inc,
            "aop_rad"   : aop,
            "pan_rad"   : pan,
            "epp_yr"    : epp
        }
        all_planets.append(planet)

# Convert to dataframe
planet_df = pd.DataFrame(all_planets)

# Format to 3 decimal places and save
planet_df.to_csv(curr_dir / "1. Planet Catalog.csv", index=False, float_format="%.3f")
#------------------------------------------------------------------------------#


