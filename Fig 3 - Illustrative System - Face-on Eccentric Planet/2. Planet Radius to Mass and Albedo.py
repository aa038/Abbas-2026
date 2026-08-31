"""
Illustrative System Post-Processing (Part 2 of 5)
-------------------------------------------------
Reads the single-planet catalog from Part 1 and augments it with:

- Masses from radii (Chen & Kipping 2017, Forecaster-style piecewise relation).
- Geometric albedos, mass-class based.
- Habitable Zone (HZ) membership flag based on stellar luminosity.

Input:
    1. Planet Catalog.csv          # From Part 1 (Same directory)

Output:
    2. Planet Catalog w Mass.csv   # Catalog from part 1 augmented with planet mass and albedo

Notes:
- This is for the demo short-period circular system in Fig. 2.
- Produces the catalog used in Part 3 and 4 (observation cadence simulation).
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import astropy.units as u

# Load the forecaster modules


# ----------------------------  Reproducibility  ---------------------------- #
SEED = 42  # set to None for non-deterministic runs
rng  = np.random.default_rng(SEED)
np.random.seed(SEED)               # For Forecaster's internal draws
# --------------------------------------------------------------------------- #

# ---------------------------------  I/O  ----------------------------------- #
curr_dir = Path(__file__).resolve().parent

planet_df = pd.read_csv(curr_dir / "1. Planet Catalog.csv")
# --------------------------------------------------------------------------- #


# ------------------  FORECASTER: Mass from Radius  ------------------------- #

# Load the Forecaster modules
forecaster_dir = curr_dir.parent / "Data" / "Forecaster"
sys.path.insert(0, str(forecaster_dir))

import mr_forecast as mr

# Planet radii for the current sample
Rp = planet_df["Rp_Rearth"].to_numpy(dtype=float, copy=True)

# One draw from P(M | R) for each simulated planet
mass = mr.Rpost2M(Rp, unit="Earth", grid_size=5000, classify="No")

# Save the planet masses to the catalog
# Masses are in Earth Masses
planet_df["Mp_Mearth"] = np.round(mass, 4)
# --------------------------------------------------------------------------- #


# ----------------------  Albedo Assignment  -------------------------------- #
# Albedo treatment for rocky planets is based off simple empirical models
# The albedo treatments for gas giants and brown dwarfs are mostly simple placeholders,
# since these planets are not considered in this study.
def assign_albedo(mass_earth):
    """
    Assign geometric albedo by mass class (Earth masses).
    Rough priors: rocky ~0.3, sub-Neptunes dim, giants brighter, brown dwarfs very dim.
    """
    albedo = np.zeros_like(mass_earth, dtype=float)

    rocky = (mass_earth < 2.04)
    albedo[rocky] = np.clip(rng.normal(0.30, 0.05, size=rocky.sum()), 0.15, 0.60)

    sub_n = (mass_earth >= 2.04) & (mass_earth < 95.16)
    albedo[sub_n] = rng.beta(2, 5, size=sub_n.sum())  # peak around 0.2

    gas = (mass_earth >= 95.16) & (mass_earth < 317.8)
    albedo[gas] = np.clip(rng.normal(0.45, 0.07, size=gas.sum()), 0.25, 0.60)

    giants = (mass_earth >= 317.8) & (mass_earth < 0.080 * u.M_sun.to(u.M_earth))
    albedo[giants] = np.clip(rng.normal(0.45, 0.07, size=giants.sum()), 0.25, 0.60)

    bd = (mass_earth >= 0.080 * u.M_sun.to(u.M_earth))
    albedo[bd] = rng.uniform(0.01, 0.05, size=bd.sum())

    return albedo


planet_df["albedo"] = assign_albedo(planet_df['Mp_Mearth'].to_numpy())
# --------------------------------------------------------------------------- #


# --------------------------- Habitable Zone Flag --------------------------- #
L   = planet_df['L_sol'].to_numpy()
sma = planet_df['sma_AU'].to_numpy()
ecc = planet_df['ecc'].to_numpy()

# Optimistic HZ boundaries from Kopparappu 2013
hz_inner = np.sqrt(L / 1.78)
hz_outer = np.sqrt(L / 0.32)

# HZ if periastron/apastron are both within HZ bounds
hz_mask         = (sma * (1 + ecc) < hz_outer) & (sma * (1 - ecc) > hz_inner)
planet_df['HZ'] = hz_mask
# --------------------------------------------------------------------------- #

# Format to 4 significant figures and save
planet_df.to_csv(curr_dir / "2. Planet Catalog w Mass.csv", index=False, float_format="%.3f")
