"""
Tongue Plot with planet catalog overlaid 
-------------------------------------------
This script reads in the MCMC fits for all 9 combinations of IWA/contrast floor and plots the recovered 
OR distributions as a 3x3 panel plot.

Input:
    The MCMC posteriors for all 9 IWA/contrast floor combinations in Data/Part II - Demographics/
    
Output:
    fig16_3x3_ExoEarthORPlot.png      # Matches Fig 16
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import beta as beta_dist

PERIOD_MIN = 0.03
PERIOD_MAX = 6.0
ECC_MIN = 1e-4
ECC_MAX = 0.99

from PlotStyle import plotStyle
plotStyle()

def hz_fraction(row, stellar_mass=1.0, stellar_luminosity=1.0):
    e = np.linspace(ECC_MIN, ECC_MAX, 2000)

    ecc_pdf = beta_dist.pdf(e, row['e_alpha'], row['e_beta'])
    ecc_pdf /= np.trapz(ecc_pdf, e)

    hz_inner = np.sqrt(stellar_luminosity / 1.78)
    hz_outer = np.sqrt(stellar_luminosity / 0.32)

    # Allowed semimajor-axis interval for an orbit fully inside the HZ.
    sma_min = hz_inner / (1.0 - e)
    sma_max = hz_outer / (1.0 + e)

    period_low = np.sqrt(sma_min**3 / stellar_mass)
    period_high = np.sqrt(sma_max**3 / stellar_mass)

    period_low = np.maximum(period_low, PERIOD_MIN)
    period_high = np.minimum(period_high, PERIOD_MAX)

    valid = period_high > period_low
    period_fraction = np.zeros_like(e)

    exponent = row['beta'] + 1.0

    if abs(exponent) < 1e-6:
        denominator = np.log(PERIOD_MAX / PERIOD_MIN)
        period_fraction[valid] = (
            np.log(period_high[valid] / period_low[valid])
            / denominator
        )
    else:
        denominator = (
            PERIOD_MAX**exponent - PERIOD_MIN**exponent
        )
        period_fraction[valid] = (
            period_high[valid]**exponent
            - period_low[valid]**exponent
        ) / denominator

    return np.trapz(ecc_pdf * period_fraction, e)

# ------------------------------------ I/O ------------------------------------- #
curr_dir   = Path(__file__).resolve().parent
fit_dir    = curr_dir.parent / "Data" / "Part II - Demographics" / "Exo-Earth Centric ORs"
# ------------------------------------------------------------------------------ #

# Directory names containing each individual fit
run_dirs = [
    '1. IWA - 0.04, OWA - 1, Contrast = 1e-9',
    '2. IWA - 0.06, OWA - 1, Contrast = 1e-9',
    '3. IWA - 0.08, OWA - 1, Contrast = 1e-9',
    '4. IWA - 0.04, OWA - 1, Contrast = 1e-10',
    '5. Fiducial Case - IWA - 0.06, Contrast = 1e-10',
    '6. IWA - 0.08, OWA - 1, Contrast = 1e-10',
    '7. IWA - 0.04, OWA - 1, Contrast = 1e-11',
    '8. IWA - 0.06, OWA - 1, Contrast = 1e-11',
    '9. IWA - 0.08, OWA - 1, Contrast = 1e-11'
]

text = [
    'a) $0.04^{\prime\prime}, 10^{-9}$',
    'b) $0.06^{\prime\prime}, 10^{-9}$',
    'c) $0.08^{\prime\prime}, 10^{-9}$',
    'd) $0.04^{\prime\prime}, 10^{-10}$',
    'e) $0.06^{\prime\prime}, 10^{-10}$',
    'f) $0.08^{\prime\prime}, 10^{-10}$',
    'g) $0.04^{\prime\prime}, 10^{-11}$',
    'h) $0.06^{\prime\prime}, 10^{-11}$',
    'i) $0.08^{\prime\prime}, 10^{-11}$'
]

# -----------------------------  Plotting Setup  --------------------------------- #
fig, axs = plt.subplots(3, 3, figsize=(15, 12))
axs = axs.flatten()

# Input SAG13 values for comparison
alpha     = -1.19
beta      = -0.74
gamma     = 0
e_alpha   = 3.7
e_beta    = 46.2
freq_full = 3.98

# Compute the OR in the region over which the exo-Earth demographics was done
freq  = freq_full * (1.4 ** (alpha+1) - 0.8 ** (alpha+1)) / (3.4 ** (alpha+1) - 0.5 ** (alpha+1))
freq *= (6 ** (beta+1) - 0.03 ** (beta+1)) / (10 ** (beta+1) - 0.03 ** (beta+1))

reference = {
    'freq': freq,
    'beta': beta,
    'e_alpha': e_alpha,
    'e_beta': e_beta,
}

# Apply the ecc cut to get the true exo-Earth OR
eta_Earth = 100.0 * freq * hz_fraction(reference)


# Loop through all the directories and plot the OR distributions individually
for idx, run_dir_name in enumerate(run_dirs):

    run_dir   = fit_dir / run_dir_name
    data_file = run_dir / f'{idx+1}e. Fit, N = 1e4.csv'

    # Load the fit data
    df = pd.read_csv(data_file)

    # Compute the exo-Earth OR for a 1 M_sol star
    # Radius is already restricted to 0.8-1.4 R_E by the exo-Earth fit.
    df['eta_Earth'] = [100.0 * row['freq'] * hz_fraction(row) for _, row in df.iterrows()]

    ax = axs[idx]

    # Plot histogram for 'freq'
    ax.hist(df['eta_Earth'], bins=30, histtype='step', density=True, lw = 2)

    # Plot dotted vertial line at actual value
    ax.axvline(x = eta_Earth, ls = "--", color = 'red')

    # Set axis labels and limits
    ax.set_xlim(0, 40)  
    ax.set_yticks([])  # Hide y-axis ticks for clean look

    ax.text(0.03, 0.95, text[idx], transform=ax.transAxes, ha='left', va='top')

    if idx >= 6:
        ax.set_xlabel('Exo-Earth OR (%)')
    else:
        ax.set_xticklabels([])

    # Customize x-axis ticks
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=14)
# -------------------------------------------------------------------------------- #

# Tight layout
plt.tight_layout(rect=[0, 0.05, 1, 1])

# Save the figure
plt.savefig(curr_dir / 'fig16_3x3_ExoEarthORPlot.png', dpi=300, bbox_inches='tight')
plt.show()