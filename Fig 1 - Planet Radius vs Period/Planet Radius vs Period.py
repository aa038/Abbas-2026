"""
Figure 1: Period vs Radius for SAG13 simulated planets
------------------------------------------------------
This script reproduces Fig. 1 from Abbas et al. 2026.
It plots all simulated SAG13 planets, highlighting
exo-Earth candidates (0.8-1.4 R_E within HZ).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.ticker as ticker

from PlotStyle import plotStyle
plotStyle()

# --- Habitable Zone function (Kopparapu 2013 scaling) ---
def HZ(L):
    """Return inner/outer HZ edges (AU) for stellar luminosity L (in L_sun)."""
    hz_inner = np.sqrt(L / 1.78)
    hz_outer = np.sqrt(L / 0.32)
    return hz_inner, hz_outer


# --- File management ---
curr_dir = Path(__file__).resolve().parent
parent_dir = curr_dir.parent
data_dir = parent_dir / "Data" / "Planet Generation"

# Load planet catalog (SAG13 distribution)
planets_df = pd.read_csv(data_dir / "SAG13 Planet Catalog.csv")

# Restrict to planets with orbital periods < 10 years
planets_df = planets_df[planets_df['P_yr'] < 10]

# --- Identify exo-Earths ---
hz_inner, hz_outer = HZ(planets_df['L_sol'])

# Planet in HZ if its periastron/apastron are inside [hz_inner, hz_outer]
in_hz = (
    (planets_df['sma_AU'] * (1 - planets_df['ecc']) >= hz_inner) &
    (planets_df['sma_AU'] * (1 + planets_df['ecc']) <= hz_outer)
)

# Exo-Earth definition: in HZ and 0.8-1.4 R_earth
exoEarths = in_hz & (
    (planets_df['Rp_Rearth'] >= 0.8) &
    (planets_df['Rp_Rearth'] <= 1.4)
)


# --- Plotting ---
# Background planets
plt.scatter(
    planets_df.loc[~exoEarths, 'P_yr'],
    planets_df.loc[~exoEarths, 'Rp_Rearth'],
    s=12,
    facecolors='C0',
    edgecolors='none',
    alpha=0.7,
    zorder=1
)

# Exo-Earths highlighted in muted green
plt.scatter(
    planets_df.loc[exoEarths, 'P_yr'],
    planets_df.loc[exoEarths, 'Rp_Rearth'],
    s=45,
    facecolors='#7fbc41',
    edgecolors='black',
    linewidths=1.2,
    zorder=3
)

# Axis labels and formatting
plt.ylabel('Planet Radius ($R_{\\oplus}$)')
plt.xlabel('Period (yr)')
plt.xscale('log')
plt.ylim([0, 4])
plt.xlim([0.01, 30])

# Tick locators
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

# Reference lines: SAG13 radius/period cuts
plt.axhline(y=0.5, color='red', ls='--', lw=1, alpha=0.5)
plt.axhline(y=3.4, color='red', ls='--', lw=1, alpha=0.5)
plt.axvline(x=0.03, color='blue', ls='--', lw=1, alpha=0.5)
plt.axvline(x=10, color='blue', ls='--', lw=1, alpha=0.5)

# Save figure (matches paper)
plt.savefig(
    curr_dir / "fig1_Period_vs_Radius.png",
    dpi=300,
    bbox_inches='tight'
)