import numpy as np
import emcee
from scipy.stats import beta, rayleigh
import warnings


from orbituary.ofti_core import *
from orbituary.solve_orbit import solve_orbit, solve_all_epochs_vectorized
from orbituary.uncertainty_utils import calculate_parameter_uncertainties


def initialize_walkers_old(n_walkers, prev_fits, MStar, obs_times, sep_obs, pa_obs, dStar):
    """
    Initialize MCMC walkers using multiple strategies for orbit fitting.
    
    This function creates initial positions for MCMC walkers using a combination of 
    previously successful orbital fits and new random orbits. It employs multiple
    strategies to ensure good starting positions:
    1. Direct use of previously successful fits
    2. Perturbed versions of successful fits
    3. Random valid orbits as a fallback
    
    Parameters
    ----------
    n_walkers : int
        Number of MCMC walkers to initialize
    prev_fits : ndarray
        Array of previously successful orbital fits, shape (n_fits, 7)
        Each row contains: [semi-major axis, eccentricity, inclination, argument of 
        periastron, position angle of nodes, epoch of periastron passage, period]
    MStar : float
        Mass of central star in solar masses
    obs_times : array-like
        Observation times in years
    sep_obs : array-like
        Observed separations in arcseconds
    pa_obs : array-like
        Observed position angles in degrees (0 to 360)
    dStar : float
        Distance to the system in parsecs

    Returns
    -------
    ndarray
        Array of initial walker positions with shape (n_walkers, 6)          # Period to be computed through Kepler's Third Law
        Each row contains orbital parameters in the same format as prev_fits

    Notes
    -----
    - Previous fits are screened against the latest epoch before use
    - Perturbed fits receive 1% Gaussian perturbations
    - New random orbits are validated to roughly match observations
    - Parameters are kept within valid ranges during perturbation
    """

    positions = []
    
    # Filter fits from the last epoch that also fit the current epoch
    filtered_fits = filter_orbits_for_next_epoch(
        prev_fits,
        obs_times,
        sep_obs,
        pa_obs,
        dStar,
        0.1,  # Slightly relaxed tolerance for MCMC since there will be some parameter exploration
        2,    # Slightly relaxed tolerance for MCMC
        sigma_threshold=5
    )
 
    # Three strategies to initialise walkers
    if len(filtered_fits) > 0:

        ref_epoch = obs_times[0]  

        P_prev = filtered_fits[:, 6]
        epp_prev = filtered_fits[:, 5]
        tau_prev = (epp_prev - ref_epoch) / P_prev

        tau_prev = np.mod(tau_prev, 1.0)  # enforce [0, 1)

        filtered_fits[:, 5] = tau_prev

        # Cut out period since it is deterministic 
        filtered_fits = filtered_fits[:, :6].copy()

        # Strategy 1: Use filtered fits directly
        n_base = min(len(filtered_fits), n_walkers * 1//4)
        positions.extend(filtered_fits[:n_base])
        
        # Strategy 2: Add perturbed versions of good fits
        n_perturbed = min(n_walkers - n_base, len(filtered_fits))
        
        for _ in range(n_perturbed):
            base_fit = filtered_fits[np.random.randint(len(filtered_fits))]
            perturbation = np.random.normal(0, 0.01, size=6)  # 1% perturbation
            new_fit = base_fit * (1 + perturbation)
            # Ensure parameters remain valid
            new_fit[0] = max(new_fit[0], 1e-6)         # sma > 0
            new_fit[1] = np.clip(new_fit[1], 0, 0.99)  # eccentricity
            new_fit[2] = np.clip(new_fit[2], 0, 180)   # inclination
            new_fit[3:5] = new_fit[3:5] % 360          # angles
            new_fit[5] = np.mod(new_fit[5], 1.0)       # tau in [0,1)
            positions.append(new_fit)
    
    
    # Strategy 3: Fill remaining positions with random valid orbits
    while len(positions) < n_walkers:
        sma = np.random.uniform(0.5, 30)
        ecc = generate_eccentricity(1)[0]
        inc = generate_inclination(1)[0]
        aop, pan = generate_orbital_angles(1)
        P = np.sqrt(sma**3 / MStar)
        tau = np.mod(generate_orbital_phase(1)[0], 1.0)
        
        walker = np.array([sma, ecc, inc, aop[0], pan[0], tau, P])
        
        # Quick check if orbit roughly matches observations
        try:
            sep_pred, pa_pred, _, _ = solve_orbit(*walker, obs_times[-1], dStar)
            if (abs(sep_pred - sep_obs[-1]) < 0.1 and 
                abs((pa_pred - pa_obs[-1] + 180) % 360 - 180) < 10):
                walker = walker[:6].copy()
                positions.append(walker)
        except:
            continue
    
    return np.array(positions)

def jitter_orbit(orbit, scale=0.01):
    """Applies multiplicative Gaussian noise to an orbit and clips physical params."""
    jitter = np.random.normal(0, scale, size=orbit.shape)
    new_orbit = orbit * (1 + jitter)

    # Enforce valid ranges
    new_orbit[0] = max(new_orbit[0], 1e-6)
    new_orbit[1] = np.clip(new_orbit[1], 0, 0.99)        # eccentricity
    new_orbit[2] = np.clip(new_orbit[2], 0, 180)         # inclination (radians)
    new_orbit[3] = new_orbit[3] % 360                    # argument of periapsis
    new_orbit[4] = new_orbit[4] % 360                    # longitude of nodes
    new_orbit[5] = np.mod(new_orbit[5], 1.0)             # tau in [0,1)
    
    return new_orbit

def initialize_walkers(n_walkers, prev_fits, obs_times):
    """
    Initialize MCMC walkers using multiple strategies for orbit fitting.
    
    This function creates initial positions for MCMC walkers using a combination of 
    previously successful orbital fits and new random orbits. It employs multiple
    strategies to ensure good starting positions:
    1. Direct use of previously successful fits
    2. Perturbed versions of successful fits
    3. Random valid orbits as a fallback
    
    Parameters
    ----------
    n_walkers : int
        Number of MCMC walkers to initialize
    prev_fits : ndarray
        Array of previously successful orbital fits, shape (n_fits, 7)
        Each row contains: [semi-major axis, eccentricity, inclination, argument of 
        periastron, position angle of nodes, epoch of periastron passage, period]
    MStar : float
        Mass of central star in solar masses
    obs_times : array-like
        Observation times in years
    sep_obs : array-like
        Observed separations in arcseconds
    pa_obs : array-like
        Observed position angles in degrees (0 to 360)
    dStar : float
        Distance to the system in parsecs

    Returns
    -------
    ndarray
        Array of initial walker positions with shape (n_walkers, 6)   # Removing period, since it can be computed through Kepler's 3rd Law
        Each row contains orbital parameters in the same format as prev_fits

    Notes
    -----
    - Previous fits are screened against the latest epoch before use
    - Perturbed fits receive 1% Gaussian perturbations
    - New random orbits are validated to roughly match observations
    - Parameters are kept within valid ranges during perturbation
    """

    positions = []

    ref_epoch = obs_times[0]  
    P_prev = prev_fits[:, 6]
    epp_prev = prev_fits[:, 5]

    tau_prev = (epp_prev - ref_epoch) / P_prev
    tau_prev = np.mod(tau_prev, 1.0)
    prev_fits[:, 5] = tau_prev

    prev_fits = prev_fits[:, :6].copy()
    
    # Draw walkers from previous posterior with jitter
    n_fits = prev_fits.shape[0]
    rng = np.random.default_rng()

    while len(positions) < n_walkers:
        idx = rng.integers(0, n_fits)
        jittered = jitter_orbit(prev_fits[idx], scale=0.05)
        positions.append(jittered)

    return np.array(positions)

def logPrior(theta_batch, MStar):
    """
    Compute log prior probability for multiple sets of orbital parameters.
    
    This function evaluates the prior probability distributions for each orbital parameter
    in vectorized form. It implements the following priors:
    - Log-uniform prior for semi-major axis (0.1 to 100 AU)
    - Beta distribution for eccentricity (α=0.867, β=3.03 from Kipping 2013)
    - Sinusoidal prior for inclination (to account for geometric projection)
    - Uniform priors for angular parameters (aop, pan)
    - Validation of Kepler's Third Law
    
    Parameters
    ----------
    theta_batch : ndarray
        Array of shape (n_orbits, 6) containing orbital parameters
        Each row contains: [semi-major axis, eccentricity, inclination, argument of 
        periastron, position angle of nodes, epoch of periastron passage]
    MStar : float
        Mass of central star in solar masses

    Returns
    -------
    ndarray
        Array of log prior probabilities for each parameter set
        Returns -inf for parameter sets that violate physical constraints

    Notes
    -----
    - Parameter ranges enforced:
        * 0.1 AU ≤ semi-major axis ≤ 100 AU
        * 0 ≤ eccentricity < 0.99
        * 0° ≤ inclination ≤ 180°
        * 0° ≤ argument of periastron < 360°
        * 0° ≤ position angle of nodes < 360°
    - Kepler's Third Law is enforced with 0.1% tolerance
    - Uses vectorized operations for efficient batch processing
    """

    # Initialize priors with zeros
    log_priors = np.zeros(len(theta_batch))
    
    # Unpack parameters
    sma, ecc, inc, aop, pan, tau = theta_batch.T

    # Create mask for valid parameters
    valid_mask = (
        (0.1 <= sma) & (sma <= 30) &
        (0 <= ecc) & (ecc < 0.99) &
        (0 <= inc) & (inc <= 180) &
        (0 <= aop) & (aop < 360) &
        (0 <= pan) & (pan < 360) &
        (0.0 <= tau) & (tau < 1.0)
    )
    
    # Set invalid parameters to -inf
    log_priors[~valid_mask] = -np.inf
    
    # For valid parameters, compute log priors
    if np.any(valid_mask):
        # Log-uniform prior for sma
        log_priors[valid_mask] += -np.log(sma[valid_mask])
        
        # Beta distribution for eccentricity
        #alpha, Beta = 0.867, 3.03
        #log_priors[valid_mask] += np.log(beta.pdf(ecc[valid_mask], alpha, Beta))
        log_priors[valid_mask] += np.log(beta(a=0.867, b=3.03).pdf(ecc[valid_mask]))
        
        # Sinusoidal prior for inclination
        log_priors[valid_mask] += np.log(np.abs(np.sin(np.radians(inc[valid_mask]))))
    
    return log_priors

    
def logLikelihood(params_batch, sep_obs, pa_obs, obs_times, dStar, MStar, sep_err=0.005, pa_err=0.1):
    """
    Compute the log-likelihood for multiple orbital parameter sets against observational data.
    
    This function calculates the likelihood using a chi-square formulation assuming Gaussian 
    measurement errors. It handles position angle wrapping and processes multiple orbital 
    parameter sets simultaneously through vectorized operations.

    Parameters
    ----------
    params_batch : ndarray
        Array of shape (n_orbits, 6) containing orbital parameters for multiple orbits.
        Each row contains: [semi-major axis, eccentricity, inclination, argument of 
        periastron, position angle of nodes, epoch of periastron passage]
    sep_obs : array-like
        Observed separations in arcseconds
    pa_obs : array-like
        Observed position angles in degrees (0 to 360)
    obs_times : array-like
        Observation times in years
    dStar : float
        Distance to the system in parsecs
    sep_err : float, optional
        Uncertainty in separation measurements in arcseconds (default: 0.003)
    pa_err : float, optional
        Uncertainty in position angle measurements in degrees (default: 0.1)

    Returns
    -------
    ndarray
        Array of log-likelihood values for each parameter set in params_batch.
        Returns -inf for parameter sets that produce invalid orbits.

    Notes
    -----
    - The function handles position angle wrapping by computing the minimum angular 
      difference between predicted and observed angles
    - Uses vectorized operations for efficient computation of multiple orbits
    - Error handling returns -inf for failed computations to ensure MCMC rejection
    """


    params6 = params_batch.copy() 
    tau = params6[:, 5]

    P  = np.sqrt(params6[:, 0]**3 / MStar)
    params7 = np.column_stack((params6, P))
    
    ref_epoch = obs_times[0]
    epp = tau * P + ref_epoch
    params7[:, 5] = epp

    sep_pred, pa_pred = solve_all_epochs_vectorized(params7, obs_times, dStar)
    sep_diff = (sep_obs - sep_pred) / sep_err
    pa_diff  = np.abs((pa_obs - pa_pred + 180) % 360 - 180) / pa_err

    log_likes = -0.5 * (np.sum(sep_diff**2, axis=1) + np.sum(pa_diff**2, axis=1))

    # NaN guard: treat any NaN as reject
    bad = ~np.isfinite(log_likes)
    if np.any(bad):
        log_likes[bad] = -np.inf

    return log_likes

    
def logPosterior(params_batch, sep_obs, pa_obs, obs_times, dStar, MStar):
    """
    Compute log posterior probability for multiple sets of orbital parameters.
    
    This function combines the prior and likelihood calculations to compute the posterior
    probability according to Bayes' theorem: P(params|data) ∝ P(data|params) * P(params).
    It handles multiple parameter sets simultaneously through vectorized operations.
    
    Parameters
    ----------
    params_batch : ndarray
        Array of shape (n_orbits, 6) containing orbital parameters
        Each row contains: [semi-major axis, eccentricity, inclination, argument of 
        periastron, position angle of nodes, epoch of periastron passage]
    sep_obs : array-like
        Observed separations in arcseconds
    pa_obs : array-like
        Observed position angles in degrees (0 to 360)
    obs_times : array-like
        Observation times in years
    dStar : float
        Distance to system in parsecs
    MStar : float
        Mass of central star in solar masses

    Returns
    -------
    ndarray
        Array of log posterior probabilities for each parameter set
        Returns -inf for parameter sets that violate physical constraints or fail likelihood computation

    Notes
    -----
    - Efficiently combines prior and likelihood calculations using vectorization
    - Returns -inf for any parameter sets that have invalid priors
    - Only computes likelihood for parameter sets with valid priors
    - Handles error propagation from both prior and likelihood calculations
    """
    
    # Compute prior for all parameter sets
    log_prior = logPrior(params_batch, MStar)
    
    # If any parameters are invalid, return -inf for those sets
    valid_mask = np.isfinite(log_prior)
    if not np.any(valid_mask):
        return -np.inf * np.ones(len(params_batch))
    
    # Compute likelihood for valid parameter sets
    log_like = logLikelihood(params_batch[valid_mask], sep_obs, pa_obs, obs_times, dStar, MStar)
    
    # Combine prior and likelihood
    result = -np.inf * np.ones(len(params_batch))
    result[valid_mask] = log_prior[valid_mask] + log_like
    
    return result

def check_chain_convergence(chain, log_probs, min_samples = 500):
    """
    Check MCMC convergence using metrics specifically designed for constrained orbital parameter spaces.
    
    This function implements multiple convergence criteria tailored for orbital parameter estimation,
    where traditional metrics like Gelman-Rubin or autocorrelation time may not be suitable due to
    the highly constrained solution space. It analyzes the stability of both the log-probability 
    distribution and individual parameter distributions across chain segments.

    Parameters
    ----------
    chain : ndarray
        Array of shape (n_steps, n_params) containing the MCMC chain
    log_probs : ndarray
        Array of shape (n_steps,) containing log probability values for each sample
    min_samples : int, optional
        Minimum number of valid samples required for convergence checking (default: 500)
        
    Returns
    -------
    bool
        Whether the chain has met all convergence criteria
    dict
        Detailed convergence diagnostics including:
        - 'converged': bool, overall convergence status
        - 'log_prob_stable': bool, stability of log probability distribution
        - 'param_stable': bool, stability of parameter distributions
        - 'valid_fraction': float, fraction of valid samples
        - 'n_valid_samples': int, number of valid samples
        - 'parameter_diagnostics': dict, per-parameter stability metrics
        - 'log_prob_means': list, mean log probability per segment
        - 'log_prob_stds': list, standard deviation of log probability per segment

    Notes
    -----
    Convergence is determined by three main criteria:
    1. Stability of log-probability distribution between chain segments
    2. Stability of parameter distributions (median shift < 0.2 * IQR)
    3. Sufficient fraction of valid samples (> 50%)

    The function divides the chain into segments and compares statistics between
    them to assess stability. A chain is considered converged only if all criteria
    are met simultaneously.
    """

    # First handle invalid values
    valid_log_prob_mask = np.isfinite(log_probs)
    valid_chain_mask = np.all(np.isfinite(chain), axis=1)
    
    # Combined mask for samples that are valid in both chain and log_probs
    valid_mask = valid_log_prob_mask & valid_chain_mask
    
    # Get fraction of valid samples
    valid_fraction = np.mean(valid_mask)
    
    # If we don't have enough valid samples, return early
    if np.sum(valid_mask) < min_samples:
        return False, {
            'converged': False,
            'reason': 'insufficient_valid_samples',
            'valid_samples': np.sum(valid_mask),
            'valid_fraction': valid_fraction,
            'required_samples': min_samples
        }
    
    # Filter to valid samples only
    valid_chain = chain[valid_mask]
    valid_log_probs = log_probs[valid_mask]
    
    # Split into quarters for analysis
    n_chunks = 4
    chunk_size = len(valid_log_probs) // n_chunks
    
    if chunk_size < min_samples // 4:
        return False, {
            'converged': False,
            'reason': 'insufficient_chunk_size',
            'chunk_size': chunk_size,
            'min_chunk_size': min_samples // 4
        }
    
    # Analyze log probability distribution
    chunk_means = []
    chunk_stds = []
    
    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = valid_log_probs[start:end]
        chunk_means.append(np.mean(chunk))
        chunk_stds.append(np.std(chunk))
    
    # Check if last two chunks are consistent
    log_prob_stable = np.abs(chunk_means[-1] - chunk_means[-2]) < 0.5 * chunk_stds[-1]
    
    # Check parameter stability
    param_stable = True
    param_metrics = {}
    
    for i in range(chain.shape[1]):
        param_chunks = np.array_split(valid_chain[:, i], n_chunks)
        
        # Compare medians and IQRs of last two chunks
        q1_prev, med_prev, q3_prev = np.percentile(param_chunks[-2], [25, 50, 75])
        q1_last, med_last, q3_last = np.percentile(param_chunks[-1], [25, 50, 75])
        
        iqr_prev = q3_prev - q1_prev
        iqr_last = q3_last - q1_last
        
        # Parameter is stable if median shift is small compared to IQR
        param_stable &= np.abs(med_last - med_prev) < 0.2 * min(iqr_prev, iqr_last)
        
        param_metrics[f'param_{i}_stable'] = {
            'median_shift': np.abs(med_last - med_prev),
            'iqr_prev': iqr_prev,
            'iqr_last': iqr_last,
            'stable': np.abs(med_last - med_prev) < 0.2 * min(iqr_prev, iqr_last)
        }
    
    # Combine all criteria and compile diagnostics
    converged = log_prob_stable and param_stable and valid_fraction > 0.5
    
    diagnostics = {
        'converged': converged,
        'log_prob_stable': log_prob_stable,
        'param_stable': param_stable,
        'valid_fraction': valid_fraction,
        'n_valid_samples': np.sum(valid_mask),
        'parameter_diagnostics': param_metrics,
        'log_prob_means': chunk_means,
        'log_prob_stds': chunk_stds
    }
    
    return converged, diagnostics
    

def MCMC(num_orbits, sep_obs, pa_obs, obs_times, dStar, MStar, prev_fits, nwalkers=100, max_steps=8000, burnin=2000, progress=True):
    """
    Run MCMC orbital fitting with optimized vectorized operations.

    This implementation uses emcee for sampling and employs specialized convergence criteria
    suitable for constrained orbital parameter spaces. It incorporates previous orbital fits
    to initialize walkers and implements quality filtering of the posterior samples.

    Parameters
    ----------
    num_orbits : int
        Number of orbital solutions to return
    sep_obs : array-like
        Observed separations in arcseconds
    pa_obs : array-like
        Observed position angles in degrees (0 to 360)
    obs_times : array-like
        Observation times in years
    dStar : float
        Distance to system in parsecs
    MStar : float
        Mass of central star in solar masses
    prev_fits : ndarray
        Array of previously successful orbital fits, shape (n_fits, 7)
        Each row contains: [semi-major axis, eccentricity, inclination, argument of 
        periastron, position angle of nodes, epoch of periastron passage, period]
    n_walkers : int, optional
        Number of MCMC walkers (default: 100, minimum: 4 * ndim)
    max_steps : int, optional
        Maximum number of MCMC steps (default: 8000)
    burnin : int, optional
        Number of initial steps to discard as burn-in (default: 2000)
    progress : bool, optional
        Whether to display progress updates (default: True)

    Returns
    -------
    ndarray
        Array of selected orbital solutions, shape (num_orbits, 7)
        Parameters are: [sma, ecc, inc, aop, pan, tau, P]
    ndarray
        Array of uncertainties for each parameter, shape (num_orbits, 7)

    Notes
    -----
    - Walkers are initialized using a combination of previous fits and their perturbed versions
    - The sampling uses vectorized operations for efficient likelihood computations
    - Final solutions are selected from the top 25% of samples based on log probability
    - Uncertainty estimation uses the full chain after burn-in
    - Measurement uncertainties assumed: PA ~ 0.1 degrees, separation ~ 0.003 arcseconds
    """

    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar subtract")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in sqrt")
    
    ndim = 6
    
    # Initialize walkers (using existing function)
    pos = initialize_walkers_old(nwalkers, prev_fits, MStar, obs_times, sep_obs, pa_obs, dStar)
    
    # Create sampler with optimized likelihood
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim,
        lambda p, *args: logPosterior(p, *args),
        args=(sep_obs, pa_obs, obs_times, dStar, MStar),
        vectorize=True
    )
    
    # Run MCMC with progress tracking if enabled
    if progress:
        # Use the progress object's compute bar
        for _ in sampler.sample(pos, iterations=max_steps):
            progress.update_computation()
    else:
        # Default to emcee's progress bar
        sampler.run_mcmc(pos, max_steps)

    # Extract samples with stricter quality threshold
    samples  = sampler.get_chain(flat=True, discard=burnin)
    log_prob = sampler.get_log_prob(flat=True, discard=burnin)

    finite = np.isfinite(log_prob)

    p10 = np.percentile(log_prob[finite], 10)
    quality_mask = finite & (log_prob >= p10)

    good_samples = samples[quality_mask]
    good_logprob = log_prob[quality_mask]

    # fallback: if still empty, use top-k finite samples directly
    if len(good_samples) == 0:
        k = min(num_orbits, np.count_nonzero(finite))
        idx = np.argsort(log_prob[finite])[-k:]
        good_samples = samples[finite][idx]
        good_logprob = log_prob[finite][idx]

    if len(good_samples) == 0:
        raise RuntimeError("No samples available after filtering; sampler likely stuck.")

    # choose indices safely
    k = min(num_orbits, len(good_samples))
    selected_indices = np.random.choice(len(good_samples), size=k, replace=False)
    best_orbits = good_samples[selected_indices]

    # Compute period
    P = np.sqrt(best_orbits[:, 0]**3 / MStar)
    
    tau = best_orbits[:, 5]
    ref_epoch = obs_times[0]
    epp = tau * P + ref_epoch

    best_orbits[:, 5] = epp
    best_orbits = np.column_stack((best_orbits, P))

    # Calculate uncertainties using percentiles
    uncertainties = calculate_parameter_uncertainties(best_orbits)
        
    # Create array of uncertainties with same shape as best_orbits
    orbit_uncertainties = np.tile(uncertainties, (k, 1))
    
    return best_orbits, orbit_uncertainties