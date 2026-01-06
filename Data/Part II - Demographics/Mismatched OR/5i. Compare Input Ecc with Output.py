import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.stats import beta
import numpy as np
from scipy.stats import rayleigh

from PlotStyle import plotStyle

plotStyle()

# Directory Management
curr_dir = Path(__file__).resolve().parent

planets_df = pd.read_csv(curr_dir / "1d. Detected Planets.csv")

counts, bins = np.histogram(planets_df['ecc'], bins=20)
bin_width = bins[1] - bins[0]

# Beta distribtuion parameters
mcmc_chain = pd.read_csv(curr_dir / "1e. Fit, N = 1e4.csv")
x = np.linspace(0, 1, 1000)

for i in range(1000):
    # Randomly select a set of model parameters
    sample_idx = np.random.randint(len(mcmc_chain['alpha']))

    e_alpha = mcmc_chain['e_alpha'][sample_idx]
    e_beta = mcmc_chain['e_beta'][sample_idx]

    pdf = beta.pdf(x, e_alpha, e_beta)
    scaled_pdf = pdf * len(planets_df) * bin_width

    if i == 0:
        plt.plot(x, scaled_pdf, lw = 1, alpha = 1, color='#7FC5FF', label = "Posterior Draw")
    plt.plot(x, scaled_pdf, lw = 0.3, alpha = 0.1, color='#7FC5FF')

plt.hist(planets_df['ecc'], bins = 20, histtype = "step", ls = "--", lw = 2, color = 'red', label = "Ecc of Simulated Planets")

rayleigh_pdf = rayleigh.pdf(x, scale=0.25)
scaled_rayleigh_pdf = rayleigh_pdf * len(planets_df) * bin_width
plt.plot(x, scaled_rayleigh_pdf, label = "Input Rayleigh Dist", color = "#d4381a")

plt.ylabel("No. of planets")
plt.xlabel("Eccentricity")
plt.legend()
plt.savefig(curr_dir / "1i. Input Ecc vs Output Ecc.png", dpi = 300, bbox_inches = 'tight')