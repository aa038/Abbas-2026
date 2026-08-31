import numpy as np
import warnings

"""
Solve Keplerian orbits to predict astrometric observations.

This module converts orbital elements into observable quantities (separations 
and position angles) for astrometric measurements of binary stars and exoplanets.

Core Features:
    - Convert orbital elements to sky-projected positions
    - Calculate observable separations and position angles
    - Generate Cartesian coordinates for orbit visualization
    - Support both single values and arrays of orbital elements

Coordinate System & Units:
    - Origin: Central body (star)
    - Sky plane: 
        x: East (+), appears to the left in sky projection
        y: North (+), appears up in sky projection
        z: Away from observer (+)
    - Angles: Degrees
    - Distances: parsecs (distance to the system) AND AU (all distances in the orbital plane)
    - Times: Years

Technical Notes:
    - Uses Newton-Raphson method for Kepler's equation
    
Example:
    >>> # Calculate separation and position angle for a simple orbit
    >>> sma = 1.2     # Semimajor axis in AU
    >>> ecc = 0.1     # Eccentricity, unitless
    >>> inc = 85      # Inclination angle in degrees
    >>> aop = 30      # Angle of periastron in degrees
    >>> pan = 45      # Position angle of nodes in degrees
    >>> epp = 2034.2  # Epoch of periastron passage in years
    >>> P = 1.5       # Period in years
    >>> t = 2035.0    # Time at which the separation and PA are to be calculated
    >>> sep, pa, r, phase = solve_orbit(sma, ecc, inc, aop, pan, epp, P, t, distance_pc)

Version: 1.1
Author: Asif Abbas
"""

def _ecc_anomaly_to_true(E, e):
    """
    Convert eccentric anomaly to true anomaly.
    """

    half_E = 0.5 * E

    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(half_E), 
                          np.sqrt(1 - e) * np.cos(half_E))


def _solve_kepler(M, e, tol=1e-5, max_iter=50):
    """
    Solve Kepler's Eqn using the Newton-Raphson Method

    Kepler's Eqn: M = E - esinE
    """

    # Initial guess for E
    E = M + e * np.sin(M) + 0.5 * e**2 * np.sin(2*M) 

    for _ in range(max_iter):

        # Newton-Raphson iteration
        E_new = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))

        # Convergence check
        if np.all(np.abs(E_new - E) < tol):
            return E_new

        E = E_new

    # Newton-Raphson failure (non-convergence) block
    warnings.warn("Kepler solver did not converge")

    return E


def _rotate_coordinates(x_orb, y_orb, inc, aop, pan):
    """
    Transform orbital coordinates to sky-projected coordinates through three rotations:
    
    1. Rotation by argument of periastron (aop) in orbital plane
    2. Rotation by inclination (inc) to tilt the orbit
    3. Rotation by position angle of nodes (pan) to orient in sky plane
    
    Angles are in radians. Coordinates are in AU.
    """

    cos_aop, sin_aop = np.cos(aop), np.sin(aop)
    cos_pan, sin_pan = np.cos(pan), np.sin(pan)
    cos_inc, sin_inc = np.cos(inc), np.sin(inc)

    # Argument of periastron rotation
    x1 = x_orb * cos_aop - y_orb * sin_aop
    y1 = x_orb * sin_aop + y_orb * cos_aop
    
    # Inclination rotation
    y2    = y1 * cos_inc
    z_sky = y1 * sin_inc
    
    # Node rotation
    x_sky = x1 * cos_pan - y2 * sin_pan
    y_sky = x1 * sin_pan + y2 * cos_pan
    
    return x_sky, y_sky, z_sky


def calculate_sky_position(sma, e, inc, aop, pan, epp, P, t):
    """
    Calculate sky-plane position and 3D planet-star orbital separation
    """

    # Mean anomaly (in radians) from period and periastron passage
    M = 2.0 * np.pi * (t - epp) / P

    # Solve Kepler's Eqn for E
    E = _solve_kepler(M, e)

    # Compute the true anomaly from E
    nu = _ecc_anomaly_to_true(E, e)

    # 3D orbital separation
    r = sma * (1.0 - e * np.cos(E))

    # Orbital coordinates 
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)

    # Convert angles to radians
    inc_rad, aop_rad, pan_rad = np.radians([inc, aop, pan])

    # Compute sky coordinates
    x_sky, y_sky, z_sky = _rotate_coordinates(x_orb, y_orb, inc_rad, aop_rad, pan_rad)

    return x_sky, y_sky, z_sky, r


def solve_orbit(sma, e, i, aop, pan, epp, P, t, distance_pc):
    """
    Compute the observable quantities for a given orbit for a set of observation times
        
    Returns
    -------
    tuple of ndarrays
        - sep_angle_arcsec   : Projected separation in arcseconds
        - position_angle_deg : Position angle in degrees (0° to 360°, measured East of North)
        - r                  : Orbital radius in AU
        - phFunc             : Phase function for reflected light calculations
    """

    # Calculate position in sky plane
    x_sky, y_sky, z_sky, r = calculate_sky_position(
        sma, e, i, aop, pan, epp, P, t
    )

    # Calculate separation and position angle
    r2D              = np.sqrt(x_sky**2 + y_sky**2)
    sep_angle_arcsec = r2D / distance_pc
    
    position_angle_rad = np.arctan2(x_sky, y_sky)
    position_angle_deg = (np.degrees(position_angle_rad) + 360) % 360

    # Calculate the phase angle
    # The phase angle is the angle at the planet between two directions:
    #   - Planet to the star
    #   - Planet to the observer
    r_mag       = np.sqrt(x_sky**2 + y_sky**2 + z_sky**2)
    cos_phAngle = z_sky / r_mag
    cos_phAngle = np.clip(cos_phAngle, -1.0, 1.0)
    phAngle     = np.arccos(cos_phAngle)

    # Lambertian Phase Function
    phFunc  = (np.sin(phAngle) + (np.pi - phAngle) * np.cos(phAngle)) / np.pi

    return sep_angle_arcsec, position_angle_deg, r, phFunc


def solve_all_orbits(params_batch, obs_times, distance_pc):
    """
    Compute the observable quantities for an array of orbits for a set of observation times
    
    Parameters
    ----------
    params_batch : ndarray
        Array of shape (n_orbits, 7) containing orbital parameters
        [sma, e, inc, aop, pan, epp, P]
    obs_times    : ndarray
        Array of observation times
    distance_pc  : float
        Distance to system in parsecs
        
    Returns
    -------
    tuple
        (sep_predictions, pa_predictions)
        Each array has shape (n_orbits, n_epochs)
    """

    # Unpack the orbital parameters for all given orbits
    sma = params_batch[:, 0][:, None]
    e   = params_batch[:, 1][:, None]
    inc = params_batch[:, 2][:, None]
    aop = params_batch[:, 3][:, None]
    pan = params_batch[:, 4][:, None]
    epp = params_batch[:, 5][:, None]
    P   = params_batch[:, 6][:, None]

    # Array of observation times
    t = obs_times[None, :]

    x_sky, y_sky, _, _ = calculate_sky_position(
        sma, e, inc, aop, pan, epp, P, t
    )

    # Calculate separation and position angle
    r2D              = np.sqrt(x_sky**2 + y_sky**2)
    sep_angle_arcsec = r2D / distance_pc
    
    position_angle_rad = np.arctan2(x_sky, y_sky)
    position_angle_deg = (np.degrees(position_angle_rad) + 360) % 360

    
    return sep_angle_arcsec, position_angle_deg
