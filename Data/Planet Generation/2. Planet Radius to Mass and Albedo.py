"""
SAG13 Post-Processing (Part 2 of 2)
-----------------------------------
Reads the planet catalog from Part 1, assigns masses from radii (Chen & Kipping 2017,
'Forecaster'-style piecewise relations), assigns geometric albedos by mass class,
flags Habitable Zone (HZ) membership, and writes back to the same CSV.

Notes
-----
- Reproducibility: set SEED below. Set to None for stochastic runs.
- I/O: by default, looks for 'Data/SAG13 Planet Catalog.csv' (Part 1's default),
       otherwise falls back to repo root. Saves back to the same path.
- Optional helper `EarthLikeAlbedo()` retained intentionally for users who want
  to use albedos with an Earth-like distribution.
"""

# ------------------------ Scientific Assumptions --------------------------- #
# - Mass–radius relation follows Chen & Kipping (2017) piecewise prescription
# - Log-normal intrinsic scatter applied per regime
# - Albedos assigned by mass class using simple empirical priors
# - HZ boundaries use fixed Kopparapu (2013) values
# - HZ membership requires entire orbit to lie within HZ
# --------------------------------------------------------------------------- #

from pathlib import Path
import numpy as np
import pandas as pd
import astropy.units as u


# ----------------------------- Reproducibility ------------------------------ #
SEED = 42  # set to None for non-deterministic runs
rng = np.random.default_rng(SEED)
# --------------------------------------------------------------------------- #


# ---------------------------------- I/O ------------------------------------ #
curr_dir = Path(__file__).resolve().parent

planet_file_path = curr_dir / "SAG13 Planet Catalog.csv"

planet_df = pd.read_csv(planet_file_path)
# --------------------------------------------------------------------------- #


# ------------------  FORECASTER: Mass from Radius  ------------------------- #
# Forecaster provides the framework to convert planet mass to radius
# Our planets are defined in radius space
# We therefore invert the Forecaster relations
# This proceeds as follows:
#   1. Convert Forecaster Mass breakpoints to radius breakpoints
#   2. In each mass (now radius) regime, invert the Mass -> Radius relation to a Radius -> Mass relation


# ---- 1. Mass breakpoints --> Radius breakpoints ---- #

# Mass breakpoints (Earth masses) between regimes defined in Forecaster
M_break = np.array([
    2.04,                           # Rocky         --> Sub-Neptunes
    95.16,                          # Sub-Neptunes  --> Gas Giants
    317.8,                          # Gas Giants    --> Giant Planets
    0.080 * u.M_sun.to(u.M_earth)   # Giant Planets --> Brown Dwarfs
])

# Jupiter Radius (Earth radii)
R_JUP = u.R_jupiter.to(u.R_earth)

# In each mass regime, Forecaster assumes:
# log10 R = C + S * log10 M
# Equivalently, R = 10 ** (C + S * log10 M) = 10 ** C x M ** S

# Slopes (S) and intercepts (C) for each mass regime
S = np.array([0.2790, 0.5037, 0.2273, 0.0, 0.8810])
C = np.array([0.0035, -0.0661, 0.4809, 1.0496, -2.8493])

def radius_from_mass(M, S, C):
    """
    Use the Forecaster Mass - Radius relation to convert planet mass to radius, 
    given the parameters S and C for the mass regime.
    """
    return 10 ** (C + S * np.log10(M))   # Using R = 10 ** (C + S * log10 M) described above

# Use the function to convert mass breakpoints to radius breakpoints
R_break = np.zeros_like(M_break)
for i in range(len(M_break)):
    R_break[i] = radius_from_mass(M_break[i], S[i], C[i])

# To model the ~constant radius for gas giants as a result of non-relativistic electron degeneracy, 
# Forecaster assumes S=0 in the gas giant regime, removing mass dependence for radius.
# But this makes it difficult to define break boundaries in radius space, since ~1-13 M_J objects have similar radii.
# We therefore add a tiny artificial scatter around R_JUP to separate the boundaries of the 'flat' gas giant radius bin
Rerr = 0.01  
R_break[2] = R_JUP - 2 * Rerr
R_break[3] = R_JUP + 2 * Rerr

# ---------------------------------------------------- #

# ---- 2. Radius --> Mass Conversion for all planets ---- #

# List of radii for all planets in the catalog generated in Part 1
# **** Ensure radius in Earth Radii !!!!!  ****
# This is ensured by default in Part 1, but be mindful of any changes you may have made 
Rp = planet_df["Rp_Rearth"].to_numpy(dtype=float, copy=True)

# Find the radius regime each planet falls into (almost entirely rocky with a few sub-Neptunes in our chosen case)
inds = np.digitize(Rp, np.hstack((0.0, R_break, np.inf)))

# Apply the inverse radius --> mass relation in each regime
mass = np.zeros_like(Rp)
for j in range(1, 6):
    sel = (inds == j)
    if not np.any(sel):
        continue
    if j == 4:
        # Flat radius regime: assign 1 M_J
        # **** This is a crude simplification made as a placeholder since this is not the mass regime we inspect in this work ****
        # **** Our planets have radii < 3.4 R_E by definition ****
        # **** If you would like to work with giant planets, modify this radius -> mass mapping!!!!! ****
        # **** You could even choose a different OR that uses mass instead of radius in Part 1 (eg. Nielsen 2019, Fulton 2021/Rosenthal 2021, Vigan 2021) ****
        mass[sel] = (u.M_jupiter).to(u.M_earth)
    else:
        # Invert piecewise relation: log10(R) = C + S*log10(M)  => log10(M) = (log10(R) - C)/S
        mass[sel] = 10.0 ** ((np.log10(Rp[sel]) - C[j - 1]) / S[j - 1])

# The generated masses are deterministic. Apply reasonable random scatter
# Chen & Kipping (2017) log10 mass scatter per regime (dex)
scatter = np.zeros_like(mass)
scatter[mass < 2.04] = 0.3                                # rocky
scatter[(mass >= 2.04) & (mass < 95.16)] = 0.15           # sub-Neptune
scatter[(mass >= 95.16) & (mass < 317.8)] = 0.10          # Jovian
scatter[mass >= 317.8] = 0.00                             # flat regime

# Apply log-normal scatter to masses
log_mass = np.log10(mass)
log_mass += rng.normal(0.0, scatter, size=mass.size)
mass = 10 ** log_mass

# Save the planet masses to the catalog
# Masses are in Earth Masses
planet_df["Mp_Mearth"] = np.round(mass, 4)
# --------------------------------------------------------------------------- #


# ----------------------  Albedo Assignment  -------------------------------- #
# Albedo treatment for rocky planets is based off simple empirical models
# If you have more advanced albedo models for these planets, I would be interested to see how the results change
# The albedo treatments for gas giants and brown dwarfs are mostly simple placeholders, 
# change (e.g. Cahoy 2010) if you want to work in this regime !!!!
def assign_albedo(mass_earth):
    """
    Assign geometric albedo by mass class (Earth masses).
    Rough priors: rocky ~0.3, sub-Neptunes dim, giants brighter, brown dwarfs very dim.
    """
    albedo = np.zeros_like(mass_earth, dtype=float)

    rocky = (mass_earth < 2.04)
    albedo[rocky] = np.clip(rng.normal(0.30, 0.05, size=rocky.sum()), 0.15, 0.60)

    sub_n = (mass_earth >= 2.04) & (mass_earth < 95.16)
    albedo[sub_n] = rng.beta(2, 8, size=sub_n.sum())  # peak around 0.1-0.15

    gas = (mass_earth >= 95.16) & (mass_earth < 317.8)
    albedo[gas] = np.clip(rng.normal(0.45, 0.07, size=gas.sum()), 0.25, 0.60)

    giants = (mass_earth >= 317.8) & (mass_earth < 0.080 * u.M_sun.to(u.M_earth))
    albedo[giants] = np.clip(rng.normal(0.45, 0.07, size=giants.sum()), 0.25, 0.60)

    bd = (mass_earth >= 0.080 * u.M_sun.to(u.M_earth))
    albedo[bd] = rng.uniform(0.01, 0.05, size=bd.sum())

    return albedo

def EarthLikeAlbedo(num_planets):
    """
    Albedos with an Earth-like distribution (geometric).
    Keeps this function around deliberately for users re-running with Earth-like priors.
    """
    # Mean ~0.367 for Earth; std dev ~0.05 (clip to [0,1])
    albedos = rng.normal(0.367, 0.05, num_planets)
    return np.clip(albedos, 0.0, 1.0)

planet_df["albedo"] = assign_albedo(planet_df['Mp_Mearth'].to_numpy())
# If desired, replace with Earth-like:
# planet_df["albedo"] = EarthLikeAlbedo(len(planet_df))
# --------------------------------------------------------------------------- #


# --------------------------- Habitable Zone Flag --------------------------- #
L = planet_df['L_sol'].to_numpy()
sma = planet_df['sma_AU'].to_numpy()
ecc = planet_df['ecc'].to_numpy()

# Optimistic HZ boundaries from Kopparappu 2013
hz_inner = np.sqrt(L / 1.78)
hz_outer = np.sqrt(L / 0.32)

# HZ if periastron/apastron are both within HZ bounds
hz_mask = (sma * (1 + ecc) < hz_outer) & (sma * (1 - ecc) > hz_inner)
planet_df['HZ'] = hz_mask
# --------------------------------------------------------------------------- #

# --------------------------------- Save ------------------------------------ #
# Write back to the SAME path we read from (preserves Part 1 location).
planet_df.to_csv(planet_file_path, index=False, float_format="%.3f")
# --------------------------------------------------------------------------- #
