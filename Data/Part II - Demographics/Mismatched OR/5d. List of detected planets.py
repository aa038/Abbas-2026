"""
List of detected planets (Part 4 of 9)
-------------------------------------------
This script scans the observing logs and stores the properties of all planets detected atleast once.
This information is used by the demographics fitter in Part 5.

Input:
    5a. Observing Log.csv          # The observing log for the chosen telescope configuration (from Part 1)
    5b. Orbit Fits.pkl             # Orbituary posteriors from Part 2


Output:
    5d. Detected Planets.csv                  # Properties of all planets detected atleast once
    5d. Planet Posterior Samples.csv          # Final P/e posterior samples used by Part e
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
    'NDet': 'max'
}).reset_index()

# Planets detected in atleast one epoch
detected = grouped[grouped['NDet'] >= 1].copy()

# Trim the planets outside our fitting range
detected = detected[(detected['Mp_MEarth'] < 40) & (detected['SMA_AU'] < 7.5)]

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


# ---------------- Export final Orbituary posterior samples ----------------- #
fits_path = curr_dir / "5c. Orbit Fits.pkl"

fits_df = pd.read_pickle(fits_path)
if isinstance(fits_df.index, pd.MultiIndex):
    fits_df = fits_df.reset_index()


def is_valid_posterior(value):
    """
    True for a nonempty Orbituary posterior with P and e samples.
    """
    return isinstance(value, pd.DataFrame) and not value.empty


detected_lookup = detected.set_index('PlanetID')
posterior_tables = []

for planet_id, planet_fits in fits_df.groupby('PlanetID', sort=False):


    # Epoch 8 may be an extrapolated copy of an earlier successful fit. Taking
    # the final nonempty row gives one final posterior per planet without
    # counting repeated extrapolated posteriors as independent measurements.
    valid = planet_fits[planet_fits['orbit_df'].map(is_valid_posterior)]

    if valid.empty:
        continue

    final_row = valid.sort_values('EpochNum').iloc[-1]
    orbit_df  = final_row['orbit_df']
    samples   = orbit_df.loc[:, ['sma', 'ecc']].copy()
    samples   = samples.replace([np.inf, -np.inf], np.nan).dropna()


    planet = detected_lookup.loc[planet_id]
    samples.insert(0, 'PlanetID', planet_id)
    samples['StarName']            = planet['StarName']
    samples['Mp_MEarth']           = float(planet['Mp_MEarth'])
    samples['M_sol']               = float(planet['M_sol'])
    samples['FinalPosteriorEpoch'] = int(final_row['EpochNum'])
    samples['NDetectedEpochs']     = int(planet['NDet'])
    posterior_tables.append(samples)


posterior_samples = pd.concat(posterior_tables, ignore_index=True)
posterior_samples.to_csv(curr_dir / "5d. Planet Posterior Samples.csv", index=False)

print(f"Saved {len(detected)} detected planets to 5d. Detected Planets.csv")



