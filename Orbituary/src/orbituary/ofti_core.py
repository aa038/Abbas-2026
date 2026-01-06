# Low-level utilities and parameter generation

import numpy as np
from scipy.stats import beta, rayleigh

from orbituary.solve_orbit import solve_all_epochs_vectorized, solve_orbit_vectorized

def init_sma(nTot):
    """
    Generate array of semi-major axes initialized to 1 AU for OFTI scaling.
    
    Parameters
    ----------
    nTot : int
        Number of semi-major axes to generate
        
    Returns
    -------
    np.ndarray
        Array of ones with shape (nTot,)
    
    Raises
    ------
    ValueError
        If nTot is not positive
    """
    if nTot <= 0:
        raise ValueError("nTot must be positive")
    
    return np.ones(nTot, dtype=np.float32)

# Module-level constants for eccentricity
_BETA_ALPHA = 0.867  # Kipping (2013) parameters
_BETA_BETA = 3.03
_MAX_ECC = 0.99
_beta_dist = beta(_BETA_ALPHA, _BETA_BETA)

def generate_eccentricity(nTot):
    """
    Generate orbital eccentricities using default Rayleigh distribution parameters.
    
    Parameters
    ----------
    nTot : int
        Number of eccentricities to generate
        
    Returns
    -------
    np.ndarray
        Array of eccentricities between 0 and _MAX_ECC
    """
    #return _beta_dist.rvs(size=nTot) * _MAX_ECC
    _beta_dist = beta(a=0.867, b=3.03)
    ecc = _beta_dist.rvs(size=nTot)
    return np.clip(ecc, 0, _MAX_ECC)

# Module-level constant
_RAD_TO_DEG = 180.0 / np.pi

def generate_inclination(nTot):
    """
    Generate inclination angles following a sin(i) distribution.
    
    Parameters
    ----------
    nTot : int
        Number of inclination angles to generate
        
    Returns
    -------
    np.ndarray
        Inclination angles in degrees [0, 180]
    """
    return _RAD_TO_DEG * np.arccos(1 - 2 * np.random.rand(nTot))

def generate_orbital_angles(nTot):
    """
    Generate uniform random angles for argument of periastron and position angle of nodes.
    
    Parameters
    ----------
    nTot : int
        Number of angle pairs to generate
        
    Returns
    -------
    tuple
        (aop, pan) : Arrays of angles in degrees [0, 360)
        aop : Argument of periastron
        pan : Position angle of nodes
    """

    # Generate both sets of angles at once, then scale to [0, 360)
    angles = 360.0 * np.random.rand(2, nTot)
    return angles[0], angles[1]

def calculate_periastron_epoch(nTot, P, observation_time):
    """
    Generate epochs of periastron passage from uniform mean anomalies.
    
    Parameters
    ----------
    nTot : int
        Number of epochs to generate
    P : float or array-like
        Orbital period(s) in years
    observation_time : float
        Time of observation in years
        
    Returns
    -------
    np.ndarray
        Epochs of periastron passage in years
    """
        
    # Generate uniform mean anomalies [0, 2π]
    mean_anomalies = np.random.uniform(0, 2 * np.pi, nTot)
    
    # Calculate corresponding epochs of periastron passage
    epp = observation_time - (P * mean_anomalies) / (2 * np.pi)
    
    return epp

def generate_orbital_phase(nTot):
    """
    Generate epochs of periastron passage from uniform mean anomalies.
    
    Parameters
    ----------
    nTot : int
        Number of epochs to generate
    P : float or array-like
        Orbital period(s) in years
    observation_time : float
        Time of observation in years
        
    Returns
    -------
    np.ndarray
        Epochs of periastron passage in years
    """
        
    # Generate uniform mean anomalies [0, 2π]
    mean_anomalies = np.random.uniform(0, 2 * np.pi, nTot)
    
    return mean_anomalies

def calculate_chi_squared(sep_obs, sep_pred, pa_obs, pa_pred, sep_err=0.003, pa_err=0.1):
    """
    Compute the chi-squared value for observed vs. predicted separations and position angles.
    Handles periodic boundary conditions for position angles.

    Parameters
    ----------
    sep_obs : np.ndarray
        Observed separations in arcseconds
    sep_pred : np.ndarray
        Predicted separations in arcseconds
    pa_obs : np.ndarray
        Observed position angles in degrees [0, 360)
    pa_pred : np.ndarray
        Predicted position angles in degrees [0, 360)
    sep_err : float or np.ndarray, optional
        Uncertainties in separations (arcseconds). Default is 0.1.
    pa_err : float or np.ndarray, optional
        Uncertainties in position angles (degrees). Default is 1.0.

    Returns
    -------
    np.ndarray
        Chi-squared values for each orbit

    Notes
    -----
    Position angles are treated as periodic variables on [0, 360) degrees.
    The minimum angular distance is used when computing position angle differences.
    """

    # Separation component
    delta_sep = (sep_obs - sep_pred) / sep_err
    
    # Position angle component with periodic boundary handling
    # Convert to radians for numerical stability
    pa_obs_rad = np.radians(pa_obs)
    pa_pred_rad = np.radians(pa_pred)
    
    # Calculate angular difference using complex exponential
    # This handles the periodic boundary condition automatically
    delta_pa_rad = np.angle(np.exp(1j * pa_obs_rad) / np.exp(1j * pa_pred_rad))
    
    # Convert back to degrees and normalize by uncertainty, to compute the chiSq
    delta_pa = np.degrees(delta_pa_rad) / pa_err

    # Compute total chi-squared
    chisq = np.sum(delta_sep**2, axis=1) + np.sum(delta_pa**2, axis=1)
    
    return chisq

def rescale_orbits(orbital_params, observation_time, sep_obs, pa_obs, Mstar, dStar, sep_err=0.003, pa_err=0.1):
    """
    Vectorized version of rescaleOrbits that handles multiple orbits simultaneously.
    
    Parameters
    ----------
    orbital_params_batch : np.ndarray
        Array of shape (n_orbits, 7) containing orbital parameters
        Each row: [sma, e, inc, aop, pan, epp, P]
    observation_time : float
        Time of observation
    sep_obs : float
        Observed separation
    pa_obs : float 
        Observed position angle
    Mstar : float
        Stellar mass
    dStar : float
        Distance to system
        
    Returns
    -------
    np.ndarray
        Array of shape (n_orbits, 7) containing rescaled orbital parameters
    """

    # Unpack parameters into separate arrays
    sma, e, inc, aop, pan, epp, P = orbital_params.T

    # Get initial predictions using vectorized solve_orbit
    sep0, pa0 = solve_orbit_vectorized(orbital_params, observation_time, dStar)

    # Adjust position angles
    pan_corr = (pa_obs - pa0 + pa_err) % 360
    pan_new = (pan + pan_corr) % 360

    # Scale semi-major axes
    scale_factor = (sep_obs + sep_err) / sep0
    sma_new = sma * scale_factor

    # Update periods 
    P_new = np.sqrt(sma_new**3 / Mstar)
    
    # Update epochs of periastron
    mean_anomaly = 2 * np.pi * (observation_time - epp) / P
    epp_new = observation_time - (mean_anomaly * P_new) / (2 * np.pi)
    
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
    sep_pred, pa_pred = solve_all_epochs_vectorized(prev_epoch_orbits, t_obs, dStar)

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

def filter_current_valid_orbits(sep_pred, pa_pred, sep_obs, pa_obs, sep_err, pa_err, sigma_threshold):
    """
    Validate predicted separations and position angles against the latest observation.
    
    Parameters
    ----------
    seps : ndarray
        Predicted separations for each orbit at both epochs, shape (n_orbits, 2)
    pas : ndarray
        Predicted position angles for each orbit at both epochs, shape (n_orbits, 2)
    sep_obs : array-like
        Observed separations [sep1, sep2]
    pa_obs : array-like
        Observed position angles [pa1, pa2]
    sep_err : float or array-like, optional
        Separation measurement uncertainty in arcsec (default: 0.003)
    pa_err : float or array-like, optional
        Position angle measurement uncertainty in degrees (default: 0.2)
    sigma_threshold : float, optional
        Number of sigma for acceptance threshold (default: 3.0)
    
    Returns
    -------
    ndarray
        Boolean mask indicating which orbits are valid
    """

    # Compute residuals
    sep_valid = np.abs(sep_pred - sep_obs) < sigma_threshold * sep_err
    pa_resid = np.abs((pa_pred - pa_obs + 180) % 360 - 180)
    pa_valid = pa_resid < sigma_threshold * pa_err

    return np.all(sep_valid & pa_valid, axis=1)

def get_orbital_parameter_ranges(orbits):
    """
    Get ranges for OFTI-sampled orbital parameters.

    Only eccentricity, inclination, and argument of periastron
    are sampled. SMA and PAN are rescaled to fit the first epoch;
    P and EPP are derived from rescaled SMA.

    Parameters
    ----------
    orbits : ndarray
        Array of shape (n_orbits, 7), orbital parameters:
        [sma, e, inc, aop, pan, epp, P]

    Returns
    -------
    ndarray
        Array of shape (3, 2) for [ecc, inc, aop] min/max
    """

    orbits = np.asarray(orbits)
    
    ranges = np.zeros((3, 2))

    # Define which parameters to analyze (skipping sma and pan)
    parameter_indices = [1, 2, 3]  # e, inc, aop
    
    # Calculate ranges for each parameter
    for i, param_idx in enumerate(parameter_indices):
        param_values = orbits[:, param_idx]
        ranges[i, 0] = np.min(param_values)
        ranges[i, 1] = np.max(param_values)
    
    return ranges