import numpy as np
import time

from .solve_orbit import solve_all_orbits
from . import ofti_core

def fit_single_epoch(num_trials, t_obs, sep_obs, pa_obs, Mstar, dStar, sep_err, pa_err):
    """
    Generate OFTI samples constrained by one astrometric epoch
    """
    
    sma = np.ones(num_trials)
    ecc = ofti_core.generate_eccentricity(num_trials)
    inc = ofti_core.generate_inclination(num_trials)
    aop, _ = ofti_core.generate_orbital_angles(num_trials)
    pan = np.zeros(num_trials)

    # Compute period and the epoch of periastron passage
    period = np.sqrt(sma**3 / Mstar)
    epp    = ofti_core.generate_periastron_epoch(num_trials, period, t_obs)

    trial_orbits    = np.column_stack([sma, ecc, inc, aop, pan, epp, period])
    rescaled_orbits = ofti_core.rescale_orbits(trial_orbits, t_obs, sep_obs, pa_obs, Mstar, dStar, sep_err, pa_err)
    
    return rescaled_orbits


def fit_multiple_epochs(num_orbits_needed, t_obs, sep_obs, pa_obs, Mstar, dStar, sep_err, pa_err, *,
    max_ofti_time=60, anchor_epoch=0, num_trials_per_batch=1000):
    """
    Generate OFTI posterior samples from two or more astrometric epochs.
    """
    # Find the epochs not used for OFTI anchoring
    non_anchor_epochs = np.arange(len(t_obs)) != anchor_epoch

    # The errors are supplied as scalars. Broadcast them to arrays
    sep_err = np.broadcast_to(sep_err, np.shape(sep_obs))
    pa_err = np.broadcast_to(pa_err, np.shape(pa_obs))

    accepted_orbits   = []
    num_accepted      = 0
    start_time        = time.monotonic()

    while num_accepted < num_orbits_needed:
        
        # Generate orbits constrained by the anchor epoch
        trial_orbits = fit_single_epoch(num_trials_per_batch, t_obs[anchor_epoch], sep_obs[anchor_epoch], pa_obs[anchor_epoch],
            Mstar, dStar, sep_err[anchor_epoch], pa_err[anchor_epoch])

        # Predict separations at non-anchor epochs
        sep_pred, pa_pred = solve_all_orbits(trial_orbits, t_obs[non_anchor_epochs], dStar)

        # Evaluate the likelihood of the remaining observations
        chi_squared = ofti_core.calculate_chi_squared(sep_obs[non_anchor_epochs], sep_pred,
            pa_obs[non_anchor_epochs], pa_pred,
            sep_err[non_anchor_epochs], pa_err[non_anchor_epochs])
        
        # Apply a likelihood based mask on all generated orbits
        accepted = trial_orbits[ofti_core.draw_rejection_mask(chi_squared)]

        if len(accepted):
            accepted_orbits.append(accepted)
            num_accepted += len(accepted)

        if time.monotonic() - start_time > max_ofti_time:
            samples = (np.concatenate(accepted_orbits) if accepted_orbits else np.empty((0, 7)))

            return samples[:num_orbits_needed], True

    return np.concatenate(accepted_orbits)[:num_orbits_needed], False
