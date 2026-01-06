"""
Input Ecc Dist vs MCMC Fit 
-------------------------------------------
Script to plot the input Rayleigh ecc distribution and the ecc distribution for the detected planets,
with random draws from the MCMC fit.

Input:
    Data/Part II - Demographics/5. Fiducial Case - IWA - 0.06, Contrast = 1e-10/5d. Detected Planets.csv               # List of detected planets (From Part 4)
    Data/Part II - Demographics/5. Fiducial Case - IWA - 0.06, Contrast = 1e-10/5e. Fit, N = 1e4.csv                   # The full list of MCMC posteriors (From Part 5)


Output:
    fig12_EccentricityDist.png        # Plot of the Ecc Distributions
"""


import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import beta
import matplotlib.pyplot as plt
from scipy.stats import rayleigh
import matplotlib.ticker as ticker

from PlotStyle import plotStyle
plotStyle()

# ------------------------------------ I/O ------------------------------------- #
# Get the path to the current directory
curr_dir   = Path(__file__).resolve().parent

data_dir = curr_dir.parent / "Data" / "Part II - Demographics" / "5. Fiducial Case - IWA - 0.06, Contrast = 1e-10"

# Detected planets
planets_df = pd.read_csv(data_dir / "5d. Detected Planets.csv")

# Load the MCMC Posteriors
mcmc_chain = pd.read_csv(data_dir / "5e. Fit, N = 1e4.csv")
# ------------------------------------------------------------------------------ #

# Eccentricity distribution of the detected planets
counts, bins = np.histogram(planets_df['ecc'], bins=20)
bin_width    = bins[1] - bins[0]

x = np.linspace(0, 1, 1000)

fig, ax = plt.subplots()

for i in range(1000):

    # Randomly select from the posterior space
    sample_idx = np.random.randint(len(mcmc_chain['e_alpha']))
    e_alpha = mcmc_chain['e_alpha'][sample_idx]
    e_beta  = mcmc_chain['e_beta'][sample_idx]

    # Posterior PDF --> scaled by total count x bin width so it overlays the hist
    pdf        = beta.pdf(x, e_alpha, e_beta)
    scaled_pdf = pdf * len(planets_df) * bin_width

    # First draw gets a legend entry; subsequent draws are faint background
    if i == 0:
        plt.plot(x, scaled_pdf, lw = 1, alpha = 1, color='#7FC5FF', label = "Posterior Draw")
    plt.plot(x, scaled_pdf, lw = 0.3, alpha = 0.1, color='#7FC5FF')

# Detected planet ecc distribution
plt.hist(planets_df['ecc'], bins = 20, histtype = "step", ls = "--", lw = 2, color = 'red', label = "Ecc of Simulated Planets")

# Input SAG13 Rayleigh Distribution
rayleigh_pdf = rayleigh.pdf(x, scale=0.061)
scaled_rayleigh_pdf = rayleigh_pdf * len(planets_df) * bin_width
plt.plot(x, scaled_rayleigh_pdf, label = "Input Rayleigh Dist", color = "#d4381a")

plt.ylabel("No. of planets")
plt.xlabel("Eccentricity")

ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))

ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

ax.set_xlim([0, 0.3])

plt.legend()
plt.savefig(curr_dir / "fig12_EccentricityDist.png", dpi = 300, bbox_inches = 'tight')