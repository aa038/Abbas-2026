import numpy as np
import pandas as pd
from pathlib import Path
import astropy.units as u
import matplotlib.pyplot as plt

# Directory Management
curr_dir = Path(__file__).resolve().parent
data_dir = curr_dir / "Data"

planet_df = pd.read_csv(data_dir / "SAG13 Planet Catalog.csv")
n_planets = len(planet_df)

# ------------------  FORECASTER: Mass from Radius  ------------------ #
# Mass breakpoints between regimes (in Earth Masses)
T = np.array([
    2.04,                           # Rocky Planets  →  Sub-Neptunes
    95.16,                          # Sub-Neptunes   →  Gas Giants
    317.8,                          # Gas Giants     →  Giant Planets
    0.080 * u.M_sun.to(u.M_earth)   # Giant Planets  →  Brown Dwarfs
])

# Define the radius of Jupiter and Saturn for coefficient computations
R_Jup = u.R_jupiter.to(u.R_earth)
R_Sat = 8.522

# Slopes (S) and intercepts (C) for each regime
S = np.array([0.2790, 0.0, 0.0, 0.0, 0.881])
C = np.array([np.log10(1.008), 0.0, 0.0, 0.0, 0.0])

# Computing coefficients using the boundary conditions at each regime
S[1] = (np.log10(R_Sat) - (C[0] + S[0] * np.log10(T[0]))) / (np.log10(T[1]) - np.log10(T[0]))
C[1] = np.log10(R_Sat) - np.log10(T[1]) * S[1]

S[2] = (np.log10(R_Jup) - np.log10(R_Sat)) / (np.log10(T[2]) - np.log10(T[1]))
C[2] = np.log10(R_Jup) - S[2] * np.log10(T[2]) 

C[3] = np.log10(R_Jup)     # Flat in log space, so slope = 0

C[4] = np.log10(R_Jup) - S[4] * np.log10(T[3])

# Compute adjusted radius thresholds for binning (to handle flat R regime)
def calc_radius_from_mass(M, S, C):

    log_R = C + S * np.log10(M)
    R = 10**log_R

    return R

# Convert the mass thresholds to radius thresholds
Tinv = np.zeros_like(T)

for i in range(len(T)):
    Tinv[i] = calc_radius_from_mass(T[i], S[i], C[i])

Rerr = 0.01  # Artifically inflated to get a spread between 1 MJ and 100 MJ planets

Tinv[2] = R_Jup - 2 * Rerr
Tinv[3] = R_Jup + 2 * Rerr
# -------------------------------------------------------------------- #


# --------------------  Apply to Planet Radii  ----------------------- #
Rp = planet_df["Rp_Rearth"].values
Rp = np.array(Rp, ndmin=1)

mass = np.zeros(Rp.shape)
inds = np.digitize(Rp, np.hstack((0, Tinv, np.inf)))

for j in range(1, 6):
    if j == 4:
        mass[inds == j] = (u.M_jupiter).to(u.M_earth)  # Flat regime → assign 1 M_J
    else:
        mass[inds == j] = 10.0 ** ((np.log10(Rp[inds == j]) - C[j - 1]) / S[j - 1])

# Scatter values in dex (from Chen & Kipping 2017)
scatter = np.zeros_like(mass)

scatter[mass < 2.04] = 0.3     # Rocky
scatter[(mass >= 2.04) & (mass < 95.16)] = 0.15   # Sub-Neptune
scatter[(mass >= 95.16) & (mass < 317.8)] = 0.1   # Jovian
scatter[mass >= 317.8] = 0.0   # Flat regime → no scatter

# Apply log-normal scatter to simulate the distribution
log_mass = np.log10(mass)
log_mass += np.random.normal(0, scatter)
mass = 10**log_mass

planet_df["Mp_Mearth"] = np.round(mass, 4)
# -------------------------------------------------------------------- #

# ----------------------  Estimating Albedo  ------------------------- #
def assign_albedo(mass):
    """Assign geometric albedo based on planet mass in Earth masses."""
    albedo = np.zeros_like(mass)

    # Rocky planets (< 2.04 M_earth): Earth- or Mars-like
    rocky = mass < 2.04
    albedo[rocky] = np.random.uniform(0.15, 0.35, size=rocky.sum())

    # Sub-Neptunes (2.04 - 95.16 M_earth): often hazy/dark
    subneptune = (mass >= 2.04) & (mass < 95.16)
    albedo[subneptune] = np.random.uniform(0.05, 0.25, size=subneptune.sum())

    # Gas Giants (95.16 - 317.8 M_earth): cloud effects dominate
    gas_giants = (mass >= 95.16) & (mass < 317.8)
    albedo[gas_giants] = np.random.uniform(0.3, 0.6, size=gas_giants.sum())

    # Giant Planets (317.8 - ~0.08 M_sol): Jupiter analogs
    giants = (mass >= 317.8) & (mass < 0.080 * u.M_sun.to(u.M_earth))
    albedo[giants] = np.random.uniform(0.3, 0.5, size=giants.sum())

    # Brown dwarfs (> 0.08 M_sol)
    bdwarfs = mass >= 0.080 * u.M_sun.to(u.M_earth)
    albedo[bdwarfs] = np.random.uniform(0.01, 0.05, size=bdwarfs.sum())

    return albedo

def EarthLikeAlbedo(num_planets):
    """
    Assign random albedos to Earth-like planets using a normal distribution.

    Parameters:
    num_planets (int): Number of planets to assign albedos for.

    Returns:
    np.array: Array of albedo values.
    """
    albedos = np.random.normal(0.367, 0.05, num_planets)  # Mean=0.3, StdDev=0.05

    return np.clip(albedos, 0, 1)  

planet_df["albedo"] = EarthLikeAlbedo(len(planet_df))
# -------------------------------------------------------------------- #


# Format to 4 significant figures and save
planet_df = planet_df.applymap(lambda x: f"{x:.4g}" if isinstance(x, float) else x)
planet_df.to_csv(data_dir / "SAG13 Planet Catalog.csv", index=False)
