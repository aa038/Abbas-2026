"""
List of detected planets (Part 4 of 9)
-------------------------------------------
This script scans the observing logs and stores the properties of all planets detected atleast once.
This information is used by the demographics fitter in Part e.

Input:
    3a. Observing Log.csv               # The observing log for the chosen telescope configuration (from Part a)
    3c. Orbit Fits.pkl                  # Orbituary posteriors from Part c


Output:
    3d. Detected Planets.csv            # Properties of all planets detected atleast once
    3d. Planet Posterior Samples.csv    # Final P/e posterior samples used by part e
"""

import pandas as pd
import numpy as np
from pathlib import Path

def HZ(L):
    """
    Find the inner and outer edges of the HZ
    """

    HZ_inner = np.sqrt(L / 1.78)
    HZ_outer = np.sqrt(L / 0.32)

    return HZ_inner, HZ_outer

# ------------------------------------ I/O ------------------------------------- #
curr_dir   = Path(__file__).resolve().parent

# Load planet simulation results
obs_df = pd.read_csv(curr_dir / '3a. Observing Log.csv') 
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

# Compute the period using Kepler's Third Law
grouped['P_yr'] = np.sqrt(grouped['SMA_AU']**3 / grouped['M_sol']).round(3)

# Planets detected in atleast one epoch
detected = grouped[grouped['NDet'] >= 1].copy()

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

detected.to_csv(curr_dir / "3d. Detected Planets.csv", index = False)


# ---------------- Export final Orbituary posterior samples ----------------- #
def is_valid_posterior(value):
    """
    True for a nonempty posterior with P and e samples.
    """
    return isinstance(value, pd.DataFrame) and not value.empty

# Read the orbit fits
fits_path = curr_dir / "3c. Orbit Fits.pkl"
fits_df   = pd.read_pickle(fits_path)
fits_df   = fits_df.reset_index()

# Set the planet ID as the detected df index for easy lookup
detected_lookup  = detected.set_index('PlanetID')
posterior_tables = []

for planet_id, planet_fits in fits_df.groupby('PlanetID', sort=False):
    if planet_id not in detected_lookup.index:
        continue

    # Epoch 8 may be an extrapolated copy of an earlier successful fit. Taking
    # the final nonempty row gives one final posterior per planet without
    # counting repeated extrapolated posteriors as independent measurements
    valid = planet_fits[planet_fits['orbit_df'].map(is_valid_posterior)]

    if valid.empty:
        continue

    # Get the orbit fits for the last detected epoch
    final_row = valid.sort_values('EpochNum').iloc[-1]
    orbit_df  = final_row['orbit_df']

    # Extract period and ecc
    samples   = orbit_df.loc[:, ['period', 'ecc']]
    samples   = samples.replace([np.inf, -np.inf], np.nan).dropna()

    # Get planet properties
    planet = detected_lookup.loc[planet_id]
    samples.insert(0, 'PlanetID', planet_id)

    samples['StarName']            = planet['StarName']
    samples['Rp_REarth']           = planet['Rp_REarth']
    samples['M_sol']               = planet['M_sol']
    samples['FinalPosteriorEpoch'] = int(final_row['EpochNum'])
    samples['NDetectedEpochs']     = int(planet['NDet'])
    posterior_tables.append(samples)


posterior_samples = pd.concat(posterior_tables, ignore_index=True)
posterior_samples.to_csv(curr_dir / "3d. Planet Posterior Samples.csv", index=False)

print(f"Saved {len(detected)} detected planets to 3d. Detected Planets.csv")
print(f"Saved {len(posterior_samples)} posterior samples for {posterior_samples['PlanetID'].nunique()} planets to 3d. Planet Posterior Samples.csv")