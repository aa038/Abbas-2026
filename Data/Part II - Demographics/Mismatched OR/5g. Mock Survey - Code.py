import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import beta

def sample_power_law(x_min, x_max, power):
    """ Generate samples from a power-law distribution m^alpha in [m_min, m_max]. """
    u = np.random.random()
    exponent = power + 1  # α + 1 term
    samples = (u * (x_max**exponent - x_min**exponent) + x_min**exponent) ** (1 / exponent)
    return samples


def sample_beta_dist(x_min, x_max, e_alpha, e_beta):
    """Draw samples from a Beta(alpha, beta) distribution truncated to [lower, upper]."""

    while True:
        sample = beta.rvs(e_alpha, e_beta, size=1)[0]

        if x_min <= sample <= x_max:
            break

    return sample


def sample_separation(sep_lower, sep_upper, probabilities):

    # Normalize probabilities to sum to 1
    normalized_probs = probabilities / np.sum(probabilities)
    
    # Create cumulative distribution
    cdf = np.cumsum(normalized_probs)

    # Generate random number
    u = np.random.random()
    
    # Find where this falls in the CDF
    index = np.searchsorted(cdf, u) - 1

    lower = sep_lower[index]
    upper = sep_upper[index]

    sep = np.random.uniform(lower, upper)
    
    # Return the corresponding separation
    return sep

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

# Get the path to the current directory
curr_dir = Path(__file__).resolve().parent
parent_dir = curr_dir.parent.parent
data_dir = parent_dir / "Data"

# Stellar data
star_file_path = data_dir / "HWO Stars.csv"
star_data = pd.read_csv(star_file_path)
stellar_masses = star_data['M']
stellar_lum = star_data['L']

# Tongue Plot
tplot_path = curr_dir / "5b. 4D Tongue Plot.npz"
data = np.load(tplot_path)

tplot        = data['completeness']      # Shape: (n_mass, n_sma, n_ecc, n_stars)
mass_centers = data['mass_centers']
sma_centers  = data['sma_centers']
ecc_centers  = data['ecc_centers']

mass_edges = data['mass_edges']
sma_edges  = data['sma_edges']
ecc_edges  = data['ecc_edges']

# Upper and lower bin edges
mass_lower, mass_upper = mass_edges[:-1], mass_edges[1:]
sma_lower, sma_upper = sma_edges[:-1], sma_edges[1:]
ecc_lower, ecc_upper = ecc_edges[:-1], ecc_edges[1:]

# Define the stellar mass range over which the model is fit
mass_range_indices = np.where((stellar_masses > 0.3) & (stellar_masses < 1.5))[0]

# Remove the tplot and stellar data outside this range
tplot = tplot[:, :, :, mass_range_indices]
stellar_masses = stellar_masses[mass_range_indices]
stellar_lum = stellar_lum[mass_range_indices]

# Read in the MCMC fit
mcmc_chain = pd.read_csv(curr_dir / "5e. Fit, N = 1e4.csv")
mcmc_chain = mcmc_chain[['alpha', 'beta', 'gamma', 'freq', 'e_alpha', 'e_beta']]

# Create lists to collect all planet data
all_planets_data = []
simulation_summaries = []


for sim_idx in range(10000):

    if sim_idx % 100 == 0:
        print(sim_idx)

    sample_idx = np.random.randint(len(mcmc_chain['alpha']))

    # Randomly select a set of model parameters
    alpha   = mcmc_chain['alpha'][sample_idx]
    Beta    = mcmc_chain['beta'][sample_idx]
    gamma   = mcmc_chain['gamma'][sample_idx]
    freq    = mcmc_chain['freq'][sample_idx]
    e_alpha = mcmc_chain['e_alpha'][sample_idx]
    e_beta  = mcmc_chain['e_beta'][sample_idx]
 
    # Track detected planets for this simulation
    smas = []
    masses = []
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

            # Randomly draw the planet mass and sma from power law distributions
            m_P = sample_power_law(0.01, 40.0, alpha)  
            sma = sample_power_law(0.1, 7.5, Beta) 
            ecc = sample_beta_dist(1e-4, 0.99, e_alpha, e_beta)

            # Check if the planet is within the HZ
            peri = sma * (1 - ecc)
            ap   = sma * (1 + ecc)

            if peri >= HZ_inner and ap <= HZ_outer:
                HZ_planet = True
            else:
                HZ_planet = False

            # Get the indices corresponding to this mass and sma
            mass_index = np.searchsorted(mass_lower, m_P) - 1
            sma_index  = np.searchsorted(sma_lower, sma) - 1
            ecc_index  = np.searchsorted(ecc_lower, ecc) - 1

            # Find the detection probability at this sma, mass and star
            det_prob = tplot[mass_index, sma_index, ecc_index, star_idx]

            if det_prob <= 0:
                continue

            # Check if planet is detected
            if np.random.random() <= det_prob:
                
                # Store the detected planet
                smas.append(sma)
                masses.append(m_P)
                star_masses.append(mS)
                
                # Add to the comprehensive list with simulation ID
                all_planets_data.append({
                    'simulation_id': sim_idx,
                    'alpha': alpha,
                    'beta': Beta, 
                    'gamma': gamma,
                    'freq': freq,
                    'planet_mass': m_P,
                    'sma': sma,
                    'ecc': ecc,
                    'star_mass': mS,
                    'HZ': HZ_planet
                })

# Convert to pandas DataFrames
planets_df = pd.DataFrame(all_planets_data)

# Save as CSV files
planets_df.to_csv(curr_dir / '5g. Mock Survey.csv', index=False)

