from pathlib import Path
import pandas as pd
import numpy as np

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

# Directory management
curr_dir = Path(__file__).resolve().parent

obs_df = pd.read_csv(curr_dir / "1a. Observing Log.csv")

# Group by PlanetID and aggregate over all epochs
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

grouped['P_yr'] = np.sqrt(grouped['SMA_AU']**3 / grouped['M_sol']).round(3)

detected = grouped[grouped['NDet'] >= 1]


col = 'SMA_AU'

# Trim the planets outside our fitting range
#detected = detected[(detected['Rp_REarth'] > 0.5) & (detected['SMA_AU'] < 7.5)]

# Find the inner and outer HZ limits
HZ_inner, HZ_outer = HZ(detected['L_sol'])

# Check if the planet is within the HZ
sma = detected['SMA_AU']
ecc = detected['ecc']

peri = sma * (1 - ecc)
ap   = sma * (1 + ecc)

detected['HZ'] = (peri >= HZ_inner) & (ap <= HZ_outer)
detected = detected[detected['P_yr'] < 10]

detected.to_csv(curr_dir / "1d. Detected Planets.csv", index = False)



