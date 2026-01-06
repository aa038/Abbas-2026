"""
Mock Survey based off the MCMC posteriors (Part 7 of 9)
-------------------------------------------
This script generates 10,000 mock surveys using the MCMC posteriors to test the validity of the fit.

The process:
    - Draw a random set of posteriors from the posterior space

    - For each star, a Poisson draw around 𝑓 gives the number of planets around it

    - Use the other parameters (𝛼, 𝛽, 𝛾, 𝑒_𝛼, 𝑒_𝛽) to generate planet properties

    - Use the tongue plot to compute detection probability for the planet

    - Uniform random draw to check if planet is detected

    - Repeat over all stars = 1 mock survey

Input:
    5b. 4D Tongue Plot.npz      # 4D tongue plot (From Part 2)
    5e. Fit, N = 1e4.csv        # The full list of MCMC posteriors for each parameter in the OR model (From Part 5)


Output:
    5g. Mock Survey.csv         # 10,000 mock surveys with planet properties for each survey
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import beta

def sample_power_law(x_min, x_max, power):
    """ 
    Generate samples from a power-law distribution of the form x^alpha for x in [x_min, x_max]. 
    """
    u = np.random.random()

    exponent = power + 1  
    samples = (u * (x_max**exponent - x_min**exponent) + x_min**exponent) ** (1 / exponent)

    return samples


def sample_beta_dist(x_min, x_max, e_alpha, e_beta):
    """
    Draw samples from a Beta(alpha, beta) distribution truncated to [x_min, x_max].
    """

    while True:
        sample = beta.rvs(e_alpha, e_beta, size=1)[0]

        if x_min <= sample <= x_max:
            break

    return sample

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
# Get the path to the current directory
curr_dir   = Path(__file__).resolve().parent
parent_dir = curr_dir.parent
data_dir   = parent_dir.parent / "Data" / "Planet Generation"

# Read in the MCMC fit
mcmc_chain = pd.read_csv(curr_dir / "5e. Fit, N = 1e4.csv")
mcmc_chain = mcmc_chain[['alpha', 'beta', 'gamma', 'freq', 'e_alpha', 'e_beta']]

# Tongue Plot
tplot_path = curr_dir / "5b. 4D Tongue Plot.npz"
data       = np.load(tplot_path)

# Stellar data
star_file_path = data_dir / "HWO Stars.csv"
star_data      = pd.read_csv(star_file_path)
stellar_masses = star_data['M']
stellar_lum    = star_data['L']
# ------------------------------------------------------------------------------ #

tplot = data['completeness']      # Shape: (n_rad, n_per, n_ecc, n_stars)

# Bin centres for the tongue plot dimensions
rad_centers  = data['rad_centers']
per_centers  = data['per_centers']
ecc_centers  = data['ecc_centers']

# Bin edges
rad_edges  = data['rad_edges']
per_edges  = data['per_edges']
ecc_edges  = data['ecc_edges']
# Upper and lower bin edges
rad_lower, rad_upper = rad_edges[:-1], rad_edges[1:]
per_lower, per_upper = per_edges[:-1], per_edges[1:]
ecc_lower, ecc_upper = ecc_edges[:-1], ecc_edges[1:]

# Radius, period and ecc range over which the OR is defined
# Should match the numbers in 5e. Fitter.jl (Variable Name: direct_imaging_analysis_regions)
total_radius_range = [0.5, 3.4]
total_period_range = [0.03, 10]
total_ecc_range    = [1e-4, 0.95]

# Create lists to collect all the mock planet data
all_planets_data = []
simulation_summaries = []

for sim_idx in range(10000):

    if sim_idx % 100 == 0:
        print(sim_idx)  

    # Randomly select a set of model parameters
    sample_idx = np.random.randint(len(mcmc_chain['alpha']))
    alpha      = mcmc_chain['alpha'][sample_idx]
    Beta       = mcmc_chain['beta'][sample_idx]
    gamma      = mcmc_chain['gamma'][sample_idx]
    freq       = mcmc_chain['freq'][sample_idx]
    e_alpha    = mcmc_chain['e_alpha'][sample_idx]
    e_beta     = mcmc_chain['e_beta'][sample_idx]
 
    # Track detected planets for this simulation
    pers = []
    radii = []
    star_masses = []

    for star_idx, mS in enumerate(stellar_masses):

        # Find the inner and outer HZ limits
        HZ_inner, HZ_outer = HZ(stellar_lum[star_idx])

        # Compute the expected number of planets for this star
        expected_planet_count = freq * mS**gamma

        # Draw actual planet count from a Poisson distribution centred at the mean
        planet_count = np.random.poisson(expected_planet_count)
    
        if planet_count == 0:
            continue

        for _ in range(planet_count):

            # Randomly draw the planet rad and per from power law distributions
            r_P = sample_power_law(total_radius_range[0], total_radius_range[1], alpha)  
            per = sample_power_law(total_period_range[0], total_period_range[1], Beta) 
            ecc = sample_beta_dist(total_ecc_range[0], total_ecc_range[1], e_alpha, e_beta)

            sma = (per**2 * mS) ** (1/3) 

            # Check if the planet is within the HZ
            peri = sma * (1 - ecc)
            ap   = sma * (1 + ecc)

            if peri >= HZ_inner and ap <= HZ_outer:
                HZ_planet = True
            else:
                HZ_planet = False

            # Get the indices corresponding to this rad, period and ecc
            rad_index  = np.searchsorted(rad_lower, r_P) - 1
            per_index  = np.searchsorted(per_lower, per) - 1
            ecc_index  = np.searchsorted(ecc_lower, ecc) - 1

            # Find the detection probability at this per, rad, ecc and star
            det_prob = tplot[rad_index, per_index, ecc_index, star_idx]

            # Uniform random draw to check if planet is detected
            if np.random.random() <= det_prob:
                
                # Store the detected planet
                pers.append(per)
                radii.append(r_P)
                star_masses.append(mS)
                
                # Add to the comprehensive list with simulation ID
                all_planets_data.append({
                    'simulation_id': sim_idx,
                    'alpha'        : alpha,
                    'beta'         : Beta, 
                    'gamma'        : gamma,
                    'freq'         : freq,
                    'planet_rad'   : r_P,
                    'per'          : per,
                    'ecc'          : ecc,
                    'star_mass'    : mS,
                    'HZ'           : HZ_planet
                })

# Convert to pandas DataFrames
planets_df = pd.DataFrame(all_planets_data)

# Save as CSV files
planets_df.to_csv(curr_dir / '5g. Mock Survey.csv', index=False)

