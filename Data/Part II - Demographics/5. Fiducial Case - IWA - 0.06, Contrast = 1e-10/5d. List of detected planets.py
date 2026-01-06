"""
List of detected planets (Part 4 of 9)
-------------------------------------------
This script scans the observing logs and stores the properties of all planets detected atleast once.
This information is used by the demographics fitter in Part 5.

Input:
    5a. Observing Log.csv          # The observing log for the chosen telescope configuration (from Part 1)


Output:
    5d. Detected Planets.csv       # Properties of all planets detected atleast once
"""

import pandas as pd
import numpy as np
from pathlib import Path

def HZ(L):
    """
    Find the inner and outer edges of the HZ

    Parameters:
    L (float/np.array): 

    Returns:
    HZ_inner (float/np.array): Distance to the inner HZ limit (in AU)
    HZ_outer (float/np.array): Distance to the outer HZ limit (in AU)
    """
    HZ_inner = np.sqrt(L / 1.78)
    HZ_outer = np.sqrt(L / 0.32)

    return HZ_inner, HZ_outer

# ------------------------------------ I/O ------------------------------------- #
curr_dir   = Path(__file__).resolve().parent

# Load planet simulation results
obs_df = pd.read_csv(curr_dir / '5a. Observing Log.csv') 
# ------------------------------------------------------------------------------ #

# Group the planets PlanetID and aggregate over all epochs
grouped = obs_df.groupby('PlanetID').agg({
    'StarName': 'first',
    'M_sol': 'first',
    'L_sol': 'first',
    'SMA_AU': 'first',
    'Mp_MEarth': 'first',
    'Rp_REarth': 'first',
    'ecc': 'first',
    'NDet': 'sum'
}).reset_index()

# Compute the period using Kepler's Third Law
grouped['P_yr'] = np.sqrt(grouped['SMA_AU']**3 / grouped['M_sol']).round(3)

# Planets detected in atleast one epoch
detected = grouped[grouped['NDet'] >= 1]

# -----------------------  HZ Check  ----------------------- #
# Find the inner and outer HZ limits
HZ_inner, HZ_outer = HZ(detected['L_sol'])

# Check if the planet is within the HZ
sma = detected['SMA_AU']
ecc = detected['ecc']

peri = sma * (1 - ecc)
ap   = sma * (1 + ecc)

detected['HZ'] = (peri >= HZ_inner) & (ap <= HZ_outer)
# ---------------------------------------------------------- #

detected.to_csv(curr_dir / "5d. Detected Planets.csv", index = False)



