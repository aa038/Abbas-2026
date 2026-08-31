import numpy as np
import emcee
from scipy.stats import beta
import warnings


from .ofti_core import calculate_chi_squared
from .solve_orbit import solve_all_orbits


def parameters_to_orbits(theta_batch, stellar_mass, reference_epoch):
    """
    Convert MCMC parameters into physical orbital parameters.
    """

    log_sma, ecc, cos_inc, aop, pan, tau = theta_batch.T

    sma    = np.exp(log_sma)
    inc    = np.degrees(np.arccos(cos_inc))
    period = np.sqrt(sma**3 / stellar_mass)
    epp    = reference_epoch + tau * period

    return np.column_stack([sma, ecc, inc, aop, pan, epp, period])


def orbits_to_parameters(orbits, reference_epoch):
    """
    Convert physical orbital parameters into MCMC parameters.
    """

    sma, ecc, inc, aop, pan, epp, period = orbits.T

    log_sma = np.log(sma)
    cos_inc = np.cos(np.radians(inc))
    tau     = ((epp - reference_epoch) / period) % 1.0

    return np.column_stack([log_sma, ecc, cos_inc, aop, pan, tau])


def log_prior(theta_batch, sma_bounds):
    """
    Evaluate the prior for the transformed MCMC parameters.
    """

    log_sma, ecc, cos_inc, aop, pan, tau = theta_batch.T

    log_sma_min = np.log(sma_bounds[0])
    log_sma_max = np.log(sma_bounds[1])

    valid = (
        (log_sma_min < log_sma) & (log_sma < log_sma_max)
        & (0.0 < ecc) & (ecc < 1.0)
        & (-1.0 <= cos_inc) & (cos_inc <= 1.0)
        & (0.0 <= aop) & (aop < 360.0)
        & (0.0 <= pan) & (pan < 360.0)
        & (0.0 <= tau) & (tau < 1.0)
    )

    log_probability = np.full(len(theta_batch), -np.inf)

    # Eccentricity priors
    alpha_ecc = 0.867
    beta_ecc  = 3.03
    log_probability[valid] = (alpha_ecc - 1.0) * np.log(ecc[valid]) + (beta_ecc - 1.0) * np.log1p(-ecc[valid])

    return log_probability

    
def log_likelihood(theta_batch, sep_obs, pa_obs, obs_times, distance, stellar_mass, sep_err, pa_err):
    """
    Evaluate the astrometric likelihood of the MCMC parameters.
    """

    reference_epoch = obs_times[0]

    # Convert the 7 orbital parameters to the ones MCMC samples
    orbits = parameters_to_orbits(theta_batch, stellar_mass, reference_epoch)

    # COmpute chi^2 for each orbit
    sep_pred, pa_pred = solve_all_orbits(orbits, obs_times, distance)
    chi_squared       = calculate_chi_squared(sep_obs, sep_pred, pa_obs, pa_pred, sep_err, pa_err)

    log_probability = -0.5 * chi_squared
    log_probability[~np.isfinite(log_probability)] = -np.inf

    return log_probability
    

def log_posterior(theta_batch, sep_obs, pa_obs, obs_times, distance, stellar_mass, sep_err, pa_err, sma_bounds):
    """
    Evaluate the log posterior of the MCMC parameters.
    """

    posterior = np.full(len(theta_batch), -np.inf)

    # Evaluate the priors
    prior     = log_prior(theta_batch, sma_bounds)
    
    # Use valid samples to evaluate the likelihood
    valid = np.isfinite(prior)
    if np.any(valid):
        likelihood       = log_likelihood(theta_batch[valid], sep_obs, pa_obs, obs_times, distance, stellar_mass, sep_err, pa_err)
        posterior[valid] = prior[valid] + likelihood

    return posterior
    

def initialize_walkers(previous_orbits, n_walkers, reference_epoch):
    """
    Initialize walkers from distinct OFTI posterior samples.
    """

    parameters = orbits_to_parameters(previous_orbits, reference_epoch)

    if len(parameters) < n_walkers:
        raise ValueError("Not enough OFTI samples to initialize the walkers")

    indices = np.random.choice(len(parameters), size=n_walkers, replace=False)

    return parameters[indices]


def run_mcmc(num_orbits, sep_obs, pa_obs, obs_times, distance, stellar_mass, previous_orbits, sep_err, pa_err,sma_bounds,
    n_walkers=50, max_steps=8000, burnin=2000, progress=None):
    """
    Sample the orbital posterior with emcee.
    """

    reference_epoch = obs_times[0]

    # Initialize the walkers
    initial_positions = initialize_walkers(previous_orbits, n_walkers, reference_epoch)

    # Run the sampler
    sampler = emcee.EnsembleSampler(n_walkers, 6, log_posterior, args=(sep_obs, pa_obs, obs_times, distance, stellar_mass, sep_err, pa_err, sma_bounds),
        vectorize=True)

    if progress is None or progress is False:
        sampler.run_mcmc(initial_positions, max_steps)
    else:
        for _ in sampler.sample(initial_positions, iterations=max_steps):
            progress.update_computation()

    # Prepare the chain after discarding burnin
    samples = sampler.get_chain(discard=burnin, flat=True)

    finite  = np.all(np.isfinite(samples), axis=1)
    samples = samples[finite]

    # Pick orbits randomly from the posterior
    indices = np.random.choice(len(samples), size=min(num_orbits, len(samples)), replace=False)

    # Convert the MCMC parameters to the 7 orbital params
    orbits = parameters_to_orbits(samples[indices], stellar_mass, reference_epoch)

    return orbits, sampler
