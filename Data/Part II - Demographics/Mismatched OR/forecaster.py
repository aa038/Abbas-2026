import numpy as np
from scipy.stats import norm
import h5py
import os

# Read and prepare the representative parameter sets once at startup
curr_dir = os.path.dirname(os.path.abspath(__file__))
hyper_file = os.path.join(curr_dir, 'fitting_parameters.h5')
h5 = h5py.File(hyper_file, 'r')
all_hyper = h5['hyper_posterior'][:]
h5.close()

# Randomly select representative samples
N_REPRESENTATIVE_SAMPLES = 500  # We can adjust this number based on accuracy needs
representative_indices = np.random.choice(len(all_hyper), N_REPRESENTATIVE_SAMPLES, replace=False)
representative_hyper = all_hyper[representative_indices]

def piece_linear_vectorized(hyper, M, prob_R):
    """
    Vectorized version of the piece-wise linear transformation.
    Now handles multiple masses and hyper-parameters simultaneously.
    
    Parameters:
    hyper: array of hyperparameters for one representative sample
    M: array of log masses
    prob_R: array of random probabilities
    """
    n_pop = 4  # Number of populations
    
    # Split hyper-parameters
    c0 = hyper[0]
    slope = hyper[1:1 + n_pop]
    sigma = hyper[1 + n_pop:1 + 2 * n_pop]
    trans = hyper[1 + 2 * n_pop:]
    
    # Initialize c array
    c = np.zeros(n_pop)
    c[0] = c0
    for i in range(1, n_pop):
        c[i] = c[i - 1] + trans[i - 1] * (slope[i - 1] - slope[i])
    
    # Initialize output array
    R = np.zeros_like(M)
    
    # Create boundary points array for population splits
    ts = np.array([-np.inf] + list(trans) + [np.inf])
    
    # Process each population
    for i in range(n_pop):
        mask = (M >= ts[i]) & (M < ts[i+1])
        if np.any(mask):
            mu = c[i] + slope[i] * M[mask]  # Now the broadcasting should work
            R[mask] = norm.ppf(prob_R[mask], loc=mu, scale=sigma[i])
    
    return R

def optimized_mass_to_radius(masses, n_samples=10):
    """
    Convert masses to radii using multiple representative samples and averaging.
    
    Parameters:
    masses (array): Array of masses in Earth masses
    n_samples: Number of different hyperparameter sets to average over
    
    Returns:
    array: Array of radii in Earth radii
    """
    all_radii = np.zeros((n_samples, len(masses)))
    
    for i in range(n_samples):
        # Convert to log mass
        logm = np.log10(masses)
        
        # Generate random probabilities for the sample
        prob = np.random.random(len(masses))
        
        # Randomly select a parameter set from our representative samples
        hyper = representative_hyper[np.random.randint(0, N_REPRESENTATIVE_SAMPLES)]
        
        # Convert to radius using the piece-wise linear model
        logr = piece_linear_vectorized(hyper, logm, prob)
        
        # Convert back from log space
        all_radii[i] = 10.0 ** logr
    
    # Return mean radius for each mass
    return np.mean(all_radii, axis=0)