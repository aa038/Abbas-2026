# Low-level utilities and parameter generation

import numpy as np
from scipy.stats import beta

from .solve_orbit import solve_all_orbits, solve_orbit


def generate_eccentricity(nTot):
    """
    Generate orbital eccentricities following the Beta function from Kipping 2013
    """

    ecc = np.random.beta(0.867, 3.03, nTot)

    return ecc


def generate_inclination(nTot):
    """
    Generate inclination angles in degrees following a sin(i) distribution.
    """

    inc_rad = np.arccos(1 - 2 * np.random.rand(nTot))

    return np.degrees(inc_rad)


def generate_orbital_angles(nTot):
    """
    Generate uniform argument of periastron and node angles in degrees.
    """

    return 360.0 * np.random.rand(2, nTot)


def generate_periastron_epoch(nTot, P, observation_time):
    """
    Generate epochs of periastron passage from uniform mean anomalies.
    """
        
    # Generate uniform mean anomalies [0, 2π]
    mean_anomalies = np.random.uniform(0, 2 * np.pi, nTot)
    
    # Calculate corresponding epochs of periastron passage
    epp = observation_time - (P * mean_anomalies) / (2 * np.pi)
    
    return epp


def angular_difference(angle1, angle2):
    """
    Return signed angular difference angle1 - angle2 in degrees.
    """

    return (angle1 - angle2 + 180.0) % 360.0 - 180.0


def calculate_chi_squared(sep_obs, sep_pred, pa_obs, pa_pred, sep_err, pa_err):
    """
    Calculate chi-squared for separation and position-angle measurements.
    """

    delta_sep = (sep_obs - sep_pred) / sep_err
    delta_pa = angular_difference(pa_obs, pa_pred) / pa_err

    return np.sum(delta_sep**2 + delta_pa**2, axis=1)


def draw_rejection_mask(chi_squared):
    """
    Randomly accept trial orbits according to their Gaussian likelihood.
    """

    acceptance_probability = np.exp(-0.5 * chi_squared)
    random_draws = np.random.random(chi_squared.shape)

    return random_draws < acceptance_probability


def rescale_orbits(orbital_params, observation_time, sep_obs, pa_obs, Mstar, dStar, sep_err, pa_err):
    """
    Rotate and scale each orbit through the passed anchor point by changing PAN and SMA
    """

    # Unpack parameters into separate arrays
    sma, e, inc, aop, pan, epp, P = orbital_params.T

    # Get initial predictions using vectorized solve_orbit
    sep0, pa0, _, _ = solve_orbit(sma, e, inc, aop, pan, epp, P, observation_time, dStar)

    # Generate one noisy anchor point for every orbit
    n_orbits = orbital_params.shape[0]

    sep_anchor = np.random.normal(loc=sep_obs, scale=sep_err, size=n_orbits)
    pa_anchor = np.random.normal(loc=pa_obs, scale=pa_err, size=n_orbits) % 360.0
    
    # ---- Scale and rotate each orbit through its drawn anchor point ---- #
    
    # Recompute pan
    pan_corr = angular_difference(pa_anchor, pa0)
    pan_new = (pan - pan_corr) % 360.0

    # Scale SMA
    scale_factor = sep_anchor / sep0
    sma_new      = sma * scale_factor

    # Update periods 
    P_new = np.sqrt(sma_new**3 / Mstar)
    
    # Update epochs of periastron
    mean_anomaly = 2 * np.pi * (observation_time - epp) / P
    epp_new = observation_time - (mean_anomaly * P_new) / (2 * np.pi)
    # -------------------------------------------------------------------- # 

    # Stack results
    return np.column_stack([sma_new, e, inc, aop, pan_new, epp_new, P_new])


def filter_orbits_for_next_epoch(prev_epoch_orbits, t_obs, sep_obs, pa_obs, dStar, sep_err, pa_err, sigma_threshold):
    """
    Filter first epoch orbits that roughly match second epoch observation.
    
    Parameters
    ----------
    first_epoch_orbits : ndarray
        Array of orbital solutions from the previous epoch
    t_curr : float
        Time of the latest observation
    sep_curr, pa_curr : float
        Separation and position angle of the latest observation (sep in arcsec, PA in degrees)
    dStar : float
        Distance to star in parsecs
    sep_err_curr : array-like
        Measurement uncertainties in separation for the latest epoch (arcsec)
    pa_err_curr : array-like
        Measurement uncertainties in position angle for the latest epoch (degrees)
    sigma_threshold : float, optional
        Number of sigma for acceptance threshold 
    
    Returns
    -------
    ndarray
        Filtered array of orbits that roughly match second epoch
    """

    # Find the predicted 2D separation and PA for each orbit
    sep_pred, pa_pred = solve_all_orbits(prev_epoch_orbits, t_obs, dStar)

    # Ensure the observation data is a Numpy array
    sep_obs = np.asarray(sep_obs)
    pa_obs = np.asarray(pa_obs)

    # Vectorized residuals
    sep_resid = np.abs(sep_pred - sep_obs)
    pa_resid = np.abs((pa_pred - pa_obs + 180) % 360 - 180)

    # Valid if all epochs match within sigma threshold
    valid_mask = np.all((sep_resid < sigma_threshold * sep_err) &
                        (pa_resid < sigma_threshold * pa_err), axis=1)
    
    return prev_epoch_orbits[valid_mask]

