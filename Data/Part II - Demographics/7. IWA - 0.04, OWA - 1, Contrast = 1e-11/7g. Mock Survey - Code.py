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
parent_dir = curr_dir.parent
data_dir = parent_dir / "Data"

# Stellar data
star_file_path = data_dir / "HWO Stars.csv"
star_data = pd.read_csv(star_file_path)
stellar_masses = star_data['M']
stellar_lum = star_data['L']

# Tongue Plot
tplot_path = curr_dir / "7b. 4D Tongue Plot.npz"
data = np.load(tplot_path)

tplot        = data['completeness']      # Shape: (n_mass, n_per, n_ecc, n_stars)
rad_centers = data['rad_centers']
per_centers  = data['per_centers']
ecc_centers  = data['ecc_centers']

rad_edges = data['rad_edges']
per_edges  = data['per_edges']
ecc_edges  = data['ecc_edges']

# Upper and lower bin edges
rad_lower, rad_upper = rad_edges[:-1], rad_edges[1:]
per_lower, per_upper = per_edges[:-1], per_edges[1:]
ecc_lower, ecc_upper = ecc_edges[:-1], ecc_edges[1:]

# Define the stellar rad range over which the model is fit
mass_range_indices = np.where((stellar_masses > 0.3) & (stellar_masses < 1.5))[0]

# Remove the tplot and stellar data outside this range
tplot = tplot[:, :, :, mass_range_indices]
stellar_masses = stellar_masses[mass_range_indices]
stellar_lum = stellar_lum[mass_range_indices]

# Read in the MCMC fit
mcmc_chain = pd.read_csv(curr_dir / "7e. Fit, N = 1e4.csv")
mcmc_chain = mcmc_chain[['alpha', 'beta', 'gamma', 'freq', 'e_alpha', 'e_beta']]

# Create lists to collect all planet data
all_planets_data = []
simulation_summaries = []


for sim_idx in range(2000):

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
            r_P = sample_power_law(0.5, 3.4, alpha)  
            per = sample_power_law(0.02, 40.0, Beta) 
            ecc = sample_beta_dist(1e-4, 0.99, e_alpha, e_beta)

            sma = (per**2 * mS) ** (1/3) 

            # Check if the planet is within the HZ
            peri = sma * (1 - ecc)
            ap   = sma * (1 + ecc)

            if peri >= HZ_inner and ap <= HZ_outer:
                HZ_planet = True
            else:
                HZ_planet = False

            # Get the indices corresponding to this rad and sma
            rad_index = np.searchsorted(rad_lower, r_P) - 1
            per_index  = np.searchsorted(per_lower, per) - 1
            ecc_index  = np.searchsorted(ecc_lower, ecc) - 1

            # Find the detection probability at this per, rad and star
            det_prob = tplot[rad_index, per_index, ecc_index, star_idx]

            if det_prob <= 0:
                continue

            # Check if planet is detected
            if np.random.random() <= det_prob:
                
                # Store the detected planet
                pers.append(per)
                radii.append(r_P)
                star_masses.append(mS)
                
                # Add to the comprehensive list with simulation ID
                all_planets_data.append({
                    'simulation_id': sim_idx,
                    'alpha': alpha,
                    'beta': Beta, 
                    'gamma': gamma,
                    'freq': freq,
                    'planet_rad': r_P,
                    'per': per,
                    'ecc': ecc,
                    'star_mass': mS,
                    'HZ': HZ_planet
                })

# Convert to pandas DataFrames
planets_df = pd.DataFrame(all_planets_data)

# Save as CSV files
planets_df.to_csv(curr_dir / '7g. Mock Survey.csv', index=False)

