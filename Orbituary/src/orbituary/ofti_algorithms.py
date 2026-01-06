import numpy as np
import time
from scipy.stats import rayleigh, beta

from orbituary.solve_orbit import solve_all_epochs_vectorized, solve_orbit_vectorized
import orbituary.ofti_core as ofti_core

def fit_single_epoch(num_trials, observation_time, sep_obs, pa_obs, Mstar, dStar):
    """
    Generate orbital solutions for a single observation epoch.

    Parameters
    ----------
    num_trials : int
        Number of trial orbits to generate
    observation_time : float
        Time of observation (years)
    sep_obs : float
        Observed separation (arcseconds)
    pa_obs : float
        Observed position angle (degrees)
    Mstar : float
        Stellar mass (solar masses)
    dStar : float
        Distance to system (parsecs)

    Returns
    -------
    list
        List of orbital parameter sets that exactly match the observation

    Notes
    -----
    This is a special case of the OFTI algorithm for a single epoch.
    All returned orbits will exactly match the observation (within numerical precision)
    since there are no additional epochs to constrain the solution.
    """
    
    # Generate trial orbital parameters
    sma = ofti_core.init_sma(num_trials)
    ecc = ofti_core.generate_eccentricity(num_trials)
    inc = ofti_core.generate_inclination(num_trials)
    aop, pan = ofti_core.generate_orbital_angles(num_trials)
    P = np.sqrt(sma**3 / Mstar)
    epp = ofti_core.calculate_periastron_epoch(num_trials, P, observation_time)

    # Combine into parameter array for vectorized operations
    orbital_params = np.array([sma, ecc, inc, aop, pan, epp, P]).T

    # Rescale all orbits to match the observation
    rescaled_params = ofti_core.rescale_orbits(orbital_params, observation_time, sep_obs, pa_obs, Mstar, dStar)

    return rescaled_params

def fit_two_epochs(num_orbits_needed, num_trials_per_batch, t_obs, sep_obs, pa_obs, Mstar, dStar, first_epoch_orbits, sep_err=0.001, pa_err=0.1, sigma_threshold=5.0, max_ofti_time=60):
    """
    Specialized OFTI implementation for finding orbital solutions given exactly two epochs of data.
    
    Parameters
    ----------
    num_orbits_needed : int
        Number of valid orbits to find (currently 100 in Fitter.py)
    num_trials_per_batch : int 
        Number of trial orbits per batch
    t_obs : array-like, length 2
        Two observation times
    sep_obs : array-like, length 2
        Two separation measurements
    pa_obs : array-like, length 2
        Two position angle measurements
    Mstar : float
        Stellar mass in solar masses
    dStar : float
        Distance in parsecs
    first_epoch_orbits : ndarray
        Valid orbits from single_epoch for first observation
    sep_err : float or array-like, optional
        Separation measurement uncertainty in arcsec (default: 0.003)
    pa_err : float or array-like, optional
        Position angle measurement uncertainty in degrees (default: 0.2)
    sigma_threshold : float, optional
        Number of sigma for acceptance threshold (default: 3.0)
    
    Returns
    -------
    accepted_orbits : ndarray
        Array of valid orbital solutions
    switch_mcmc : bool
        Flag indicating whether to switch to MCMC
    """

    # Convert scalar errors to arrays if needed
    if np.isscalar(sep_err):
        sep_err = np.full_like(sep_obs, sep_err)
    if np.isscalar(pa_err):
        pa_err = np.full_like(pa_obs, pa_err)

    # 1. Extract parameter distributions from first epoch orbits that also fit the second epoch
    valid_prev_epoch_orbits = ofti_core.filter_orbits_for_next_epoch(first_epoch_orbits, t_obs, sep_obs, pa_obs, dStar, sep_err, pa_err, sigma_threshold)

    # 2. If we don't find good orbits, reuse the old posterior, and draw based on our priors...
    if len(valid_prev_epoch_orbits) == 0:
        print("Warning: No orbits matched all epochs. Falling back to first-epoch orbits for sampling only.")
        param_ranges = ofti_core.get_orbital_parameter_ranges(first_epoch_orbits)
        accepted_orbits = []
        orbits_found = 0
    # 2. ....If we do find good orbits, grab those too before random draws
    else:
        param_ranges = ofti_core.get_orbital_parameter_ranges(valid_prev_epoch_orbits)
        accepted_orbits = list(valid_prev_epoch_orbits)
        orbits_found = len(accepted_orbits)

    start_time = time.time()

    while orbits_found < num_orbits_needed:
        # 3. Generate batch of trial orbits using param_ranges
        trial_orbits = generate_trial_batch(num_trials_per_batch, param_ranges, Mstar, dStar, t_obs, sep_obs, pa_obs)

        # 4. Vectorized validation against both epochs
        seps, pas = solve_all_epochs_vectorized(trial_orbits, t_obs, dStar)

        # 5. Efficient batch validation
        valid_mask = ofti_core.filter_current_valid_orbits(seps, pas, sep_obs, pa_obs, sep_err, pa_err, sigma_threshold)

        # 6. Store valid orbits
        new_valid_orbits = trial_orbits[valid_mask]
        accepted_orbits.extend(new_valid_orbits)
        orbits_found += len(new_valid_orbits)

        # 7. Adaptive MCMC switching criteria
        if time.time() - start_time > max_ofti_time:
            return np.array(accepted_orbits), True
    
        
        # 8. Update param_ranges based on successful orbits
        if len(accepted_orbits) > 10:
            param_ranges = ofti_core.get_orbital_parameter_ranges(accepted_orbits)

    return np.array(accepted_orbits), False


def generate_trial_batch(num_trials, param_ranges, Mstar, dStar, t_obs, sep_obs, pa_obs):
    """
    Generate a batch of trial orbits using fixed sma/pan and other parameters within ranges.
    
    Parameters
    ----------
    num_trials : int
        Number of trial orbits to generate
    param_ranges : ndarray
        Array of shape (3, 2) containing [min, max] ranges for:
        eccentricity, inclination, argument of periastron, 
        epoch of periastron passage, and period
        
    Returns
    -------
    ndarray
        Array of shape (num_trials, 7) containing trial orbits
        [sma, e, inc, aop, pan, epp, P]
    """
    # Initialize output array
    trial_orbits = np.zeros((num_trials, 7))

    # 1. Sample eccentricity
    e_min, e_max = param_ranges[0]
    ecc = np.clip(beta(a=0.867, b=3.03).rvs(num_trials), e_min, e_max)
    trial_orbits[:, 1] = ecc

    # 2. Sample inclination
    i_min, i_max = np.radians(param_ranges[1])
    u = np.random.uniform(0, 1, num_trials)
    inc = np.degrees(np.arccos(np.cos(i_min) - (np.cos(i_min) - np.cos(i_max)) * u))
    trial_orbits[:, 2] = inc

    # 3. Sample argument of periastron
    aop_min, aop_max = param_ranges[2]
    trial_orbits[:, 3] = np.random.uniform(aop_min, aop_max, num_trials)

    # 4. Placeholder SMA = 1 AU, random PAN
    trial_orbits[:, 0] = 1.0
    _, pan = ofti_core.generate_orbital_angles(num_trials)
    trial_orbits[:, 4] = pan

    trial_orbits[:, 6] = np.sqrt(trial_orbits[:, 0]**3 / Mstar)
    trial_orbits[:, 5] = ofti_core.calculate_periastron_epoch(num_trials, trial_orbits[:, 6], t_obs[0])

    # Rescale the orbits to match observed separation and position angle
    rescaled_orbits = ofti_core.rescale_orbits(trial_orbits, t_obs[0], sep_obs[0], pa_obs[0], Mstar, dStar)

    return rescaled_orbits