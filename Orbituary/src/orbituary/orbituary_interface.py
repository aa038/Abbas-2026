import numpy as np
import pandas as pd

from .ofti_algorithms import fit_multiple_epochs, fit_single_epoch
from .ofti_core import calculate_chi_squared
from .mcmc_solver import run_mcmc
from .solve_orbit import solve_all_orbits

def fit_orbit(
    times,
    separations,
    position_angles,
    stellar_mass,
    distance,
    sep_uncertainty,
    pa_uncertainty,
    *,
    sma_bounds,
    initial_orbits=None,
    num_orbits=500,
    max_ofti_time=300,
    ofti_batch_size=1000,
    mcmc_walkers=50,
    mcmc_steps=30_000,
    mcmc_burnin=5_000,
    progress=False,
):
    
    """
    Fit relative-astrometry observations with OFTI and, when useful, MCMC.

    Parameters
    ----------
    times : array-like
        Observation times in years.
    separations : array-like
        Angular separations in arcseconds.
    position_angles : array-like
        Position angles in degrees.
    stellar_mass : float
        Total system mass in solar masses.
    distance : float
        System distance in parsecs.
    sep_uncertainty, pa_uncertainty : float or array-like
        Measurement uncertainties. Scalars are applied to every epoch.
    sma_bounds : tuple
        Lower and upper semimajor-axis bounds in AU.
    num_orbits : int, optional
        Number of posterior orbits to return.

    Returns
    -------
    orbits : pandas.DataFrame
        Posterior samples with columns sma, ecc, inc, aop, pan, epp, and period.
    diagnostics : dict
        Fitting method and available OFTI or MCMC diagnostics.
    """

    times = np.asarray(times, dtype=float)
    separations = np.asarray(separations, dtype=float)
    position_angles = np.asarray(position_angles, dtype=float)

    n_epochs = len(times)

    sep_uncertainty = np.broadcast_to(sep_uncertainty, n_epochs)
    pa_uncertainty  = np.broadcast_to(pa_uncertainty, n_epochs)

    
    # Single-epoch OFTI
    if n_epochs == 1:
        orbits = fit_single_epoch(num_orbits, times[0], separations[0], position_angles[0], stellar_mass, distance, sep_uncertainty[0], pa_uncertainty[0])

        diagnostics = {
            "method": "ofti",
            "ofti_timed_out": False,
            "sampler": None
        }

    # Two-epoch OFI
    elif n_epochs == 2:
        orbits, timed_out = fit_multiple_epochs(max(num_orbits, mcmc_walkers), times, separations, position_angles, stellar_mass, distance, sep_uncertainty, pa_uncertainty,
            max_ofti_time=max_ofti_time, num_trials_per_batch=ofti_batch_size)

        # Switch to MCMC if the orbit is too constrained
        if timed_out:
            if len(orbits) < mcmc_walkers:
                raise RuntimeError("OFTI timed out before producing enough samples to initialize the MCMC walkers")

            orbits, sampler = run_mcmc(num_orbits, separations, position_angles, times, distance, stellar_mass, orbits, sep_uncertainty, pa_uncertainty,
                sma_bounds, n_walkers=mcmc_walkers, max_steps=mcmc_steps, burnin=mcmc_burnin)

            diagnostics = {
                "method": "mcmc",
                "ofti_timed_out": True,
                "sampler": sampler
            }

        # If OFTI is successful, return the orbits
        else:
            orbits = orbits[:num_orbits]

            diagnostics = {
                "method": "ofti",
                "ofti_timed_out": False,
                "sampler": None
            }

    # Multi-epoch MCMC
    else:
        if initial_orbits is None:
            # Generate seeds through OFTI to initialise MCMC walkers
            seed_indices = np.argsort(times)[:2]
            seed_pool_size = max(1000, 20 * mcmc_walkers)

            seed_orbits, timed_out = fit_multiple_epochs(seed_pool_size, times[seed_indices], separations[seed_indices], position_angles[seed_indices],
                stellar_mass, distance, sep_uncertainty[seed_indices], pa_uncertainty[seed_indices],
                max_ofti_time=max_ofti_time, num_trials_per_batch=ofti_batch_size)

        else:
            seed_orbits = np.asarray(initial_orbits)
            timed_out   = False

        # Stay inside the SMA bounds
        seed_orbits = seed_orbits[(seed_orbits[:, 0] > sma_bounds[0])& (seed_orbits[:, 0] < sma_bounds[1])]

        if len(seed_orbits) < mcmc_walkers:
            raise RuntimeError("OFTI did not produce enough samples to initialize the MCMC walkers")
        
        # Solve the initial guesses for all the epochs and pick the best ones
        seed_sep, seed_pa = solve_all_orbits(seed_orbits, times, distance)
        seed_chi_squared  = calculate_chi_squared(separations, seed_sep, position_angles, seed_pa, sep_uncertainty, pa_uncertainty)
        best_indices      = np.argsort(seed_chi_squared)[:mcmc_walkers]
        initial_orbits    = seed_orbits[best_indices]

        orbits, sampler = run_mcmc(num_orbits, separations, position_angles, times, distance, stellar_mass,
            initial_orbits, sep_uncertainty, pa_uncertainty, sma_bounds, n_walkers=mcmc_walkers, max_steps=mcmc_steps, burnin=mcmc_burnin)

        diagnostics = {
            "method": "mcmc",
            "ofti_timed_out": timed_out,
            "sampler": sampler
        }

    # Compute chi^2 for each orbit
    sep_pred, pa_pred = solve_all_orbits(orbits, times, distance)  
    chi_squared       = calculate_chi_squared(separations, sep_pred, position_angles, pa_pred, sep_uncertainty, pa_uncertainty)

    diagnostics["chi_squared"] = chi_squared

    orbits = pd.DataFrame(orbits, columns=["sma", "ecc", "inc", "aop", "pan", "epp", "period"])

    return orbits, diagnostics



