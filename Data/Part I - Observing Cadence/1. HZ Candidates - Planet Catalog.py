"""
HZ Candidates
---------------------------------------------------
Find all HZ candidates from the full subset of the SAG13 planets (Fig 1).
These will be used for comparing the uniform vs adaptive cadences (Sec 3.3, Abbas et al. 2026)

Inputs:
    1. Data / Planet Generation / SAG13 Planet Catalog.csv   #  Full planet catalog

Outputs:
    1. Planet Catalog.csv                                    # Subset of HZ candidates

"""

# ------------------------ Scientific Assumptions --------------------------- #
# - Exo-Earths are defined as planets:
# -     (a) with radius 0.8 R_E < R_p < 1.4 R_E
# -     (b) fully within the HZ
# - We adopt the optimistic HZ formalism from Kopparappu et al. 2013
# --------------------------------------------------------------------------- #

import numpy as np
from pathlib import Path
import pandas as pd

def HZ(L):
    """
    Calculate inner and outer edges of the habitable zone of a star,
    based off Kopparappu 2013.

    Parameters:
    L (float or np.array): Stellar luminosity in solar units.

    Returns:
    HZ_inner (float or np.array): Inner edge of the HZ (in AU).
    HZ_outer (float or np.array): Outer edge of the HZ (in AU).
    """
    HZ_inner = np.sqrt(L / 1.78)
    HZ_outer = np.sqrt(L / 0.32)
    return HZ_inner, HZ_outer


# ---------------------------------- I/O ------------------------------------ #
curr_dir = Path(__file__).resolve().parent
data_dir = curr_dir.parent

# Load the simulated planet catalog with mass
planets = pd.read_csv(data_dir / "Planet Generation" / "SAG13 Planet Catalog.csv")
# --------------------------------------------------------------------------- #

# Find the subset of planets with exo-Earth radii
planets = planets[(planets['Rp_Rearth'] > 0.8) & (planets['Rp_Rearth'] < 1.4)].reset_index(drop = True)

# Find the subset of planets in the HZ
planets['HZ_inner'], planets['HZ_outer'] = HZ(planets['L_sol'])

apastron = planets['sma_AU'] * (1 + planets['ecc'])
periastron = planets['sma_AU'] * (1 - planets['ecc'])

mask_1 = apastron < planets['HZ_outer']
mask_2 = periastron > planets['HZ_inner']

planets = planets[mask_1 & mask_2]

# Save as a .csv
planets.to_csv(curr_dir / "1. Planet Catalog.csv", index = False)



