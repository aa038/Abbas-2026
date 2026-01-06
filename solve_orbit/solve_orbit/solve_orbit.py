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
    - Supports eccentricities from 0 to <1
    
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
    >>> sep, pa = solve_orbit(sma, ecc, inc, aop, pan, epp, P, t)   # sep in arcsec, PA in degrees


References:
    [1] Meeus, J. (1998). Astronomical Algorithms.
    [2] Murray & Dermott (1999). Solar System Dynamics.

Version: 1.0
Author: Asif Abbas
"""

def period_from_sma(sma, m_star):
    """
    Calculate orbital period using Kepler's Third Law.
    
    Parameters
    ----------
    sma : float or ndarray
        Semi-major axis in AU
    m_star : float
        Stellar mass in solar masses
        
    Returns
    -------
    float or ndarray
        Orbital period in years
        
    Notes
    -----
    Uses the simplified form of Kepler's Third Law:
    P² = a³/M, where P is in years, a in AU, and M in solar masses.
    
    Raises
    ------
    ValueError
        If m_star is zero
    """

    if m_star <= 0:
        raise ValueError("Stellar mass must be positive")
    
    return np.sqrt(sma**3 / m_star)

def ecc_anomaly_to_true(E, e):
    """
    Convert eccentric anomaly to true anomaly.
    
    Parameters
    ----------
    E : float or ndarray
        Eccentric anomaly in radians
    e : float or ndarray
        Orbital eccentricity (0 ≤ e < 1)
        
    Returns
    -------
    float or ndarray
        True anomaly in radians
        
    Notes
    -----
    Uses the conversion:
    tan(ν/2) = sqrt((1+e)/(1-e)) * tan(E/2)
    where ν is true anomaly, e is eccentricity and E is eccentric anomaly.
    
    References
    ----------
    [1] Meeus, J. (1998). Astronomical Algorithms, Ch. 30
    
    Raises
    ------
    ValueError
        If eccentricity is outside [0,1)
    """

    if np.any((e >= 1)):  # Only check upper bound for numerical stability
        raise ValueError("Eccentricity must be less than 1")

    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), np.sqrt(1 - e) * np.cos(E/2))

def _calculate_orbital_position(sma, e, i, aop, pan, epp, P, t, kepler_tol=1e-4):
    """
    Calculate orbital position in sky-projected coordinates.
    
    Solves Kepler's equation using the Newton-Raphson method and applies
    coordinate transformations to get sky-projected positions.
    
    Parameters
    ----------
    sma : float or array-like
        Semi-major axis in AU
    e : float or array-like
        Eccentricity
    i : float or array-like
        Inclination in degrees
    aop : float or array-like 
        Argument of periastron in degrees
    pan : float or array-like
        Position angle of ascending node in degrees
    epp : float or array-like
        Epoch of periastron passage in years
    P : float or array-like
        Orbital period in years
    t : float or array-like
        Time of observation in years
    kepler_tol : float, optional
        Convergence tolerance for Kepler solver
    
    Returns
    -------
    tuple
        (x_sky, y_sky, z_sky) coordinates in AU
    """

    # Calculate the mean anomaly
    M = np.where(P != 0, 2 * np.pi * (t - epp) / P, np.nan)

    # Solve Kepler's equation using Newton-Raphson method
    # Better initial guess for Kepler equation than E = M
    E = M + e * np.sin(M) + 0.5 * e**2 * np.sin(2*M)  
    
    # Newton-Raphson iteration
    for _ in range(100):
        delta = E - e * np.sin(E) - M
        E_new = E - delta / (1 - e * np.cos(E))
        
        if np.all(np.abs(E_new - E) < kepler_tol):
            break
        E = E_new
    else:
        warnings.warn("Kepler solver did not converge to desired tolerance")
    
    # True anomaly using stable formula
    nu = ecc_anomaly_to_true(E, e)
    
    # Orbital radius 
    r = sma * (1 - e**2) / (1 + e * np.cos(nu))
    
    # Position in orbital plane
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)
    
    # Convert angles once
    i_rad, aop_rad, pan_rad = np.radians([i, aop, pan])
    
    # Rotation matrices application (could be made more efficient)
    x_sky, y_sky, z_sky = rotate_coordinates(x_orb, y_orb, i_rad, aop_rad, pan_rad)
    
    return x_sky, y_sky, z_sky, r

def rotate_coordinates(x_orb, y_orb, i, aop, pan):
    """
    Transform orbital coordinates to sky-projected coordinates through three rotations.
    
    Applies three successive rotations to convert from the orbital plane to the sky plane:
    1. Rotation by argument of periastron (aop) in orbital plane
    2. Rotation by inclination (i) to tilt the orbit
    3. Rotation by position angle of nodes (pan) to orient in sky plane
    
    Parameters
    ----------
    x_orb : float or ndarray
        x-coordinate in orbital plane (along major axis) [AU]
    y_orb : float or ndarray
        y-coordinate in orbital plane (along minor axis) [AU]
    i : float or ndarray
        Inclination angle in radians
        i = 0°: face-on, counterclockwise
        i = 90°: edge-on orbit
        i = 180°: face-on, clockwise
    aop : float or ndarray
        Argument of periastron in radians
        Angle from ascending node to periastron
    pan : float or ndarray
        Position angle of ascending node in radians
        Measured East of North in sky plane
    
    Returns
    -------
    x_sky : float or ndarray
        Sky-projected x coordinate (positive East) [AU]
    y_sky : float or ndarray
        Sky-projected y coordinate (positive North) [AU]
    z_sky : float or ndarray
        Line-of-sight coordinate (positive away from observer) [AU]
    
    Notes
    -----
    The three rotations follow the Thiele-Innes formalism:
    1. aop rotation: Around z-axis in orbital plane
    2. i rotation: Around new x-axis to incline orbit
    3. pan rotation: Around z-axis to align in sky plane
    
    References
    ----------
    [1] Murray & Dermott (1999). Solar System Dynamics, Ch. 2
    [2] Smart, W. M. (1930). Textbook on Spherical Astronomy, Ch. 4
    """

    cos_aop, sin_aop = np.cos(aop), np.sin(aop)
    cos_pan, sin_pan = np.cos(pan), np.sin(pan)
    cos_i, sin_i = np.cos(i), np.sin(i)

    # Argument of periastron rotation
    x1 = x_orb * cos_aop - y_orb * sin_aop
    y1 = x_orb * sin_aop + y_orb * cos_aop
    
    # Inclination rotation
    x2 = x1
    y2 = y1 * cos_i
    z2 = y1 * sin_i
    
    # Node rotation
    x_sky = x2 * cos_pan - y2 * sin_pan
    y_sky = x2 * sin_pan + y2 * cos_pan
    z_sky = z2
    
    return x_sky, y_sky, z_sky

def solve_orbit(sma, e, i, aop, pan, epp, P, t, distance_pc, kepler_tol=1e-4):
    """
    Compute observable astrometric quantities and orbital parameters from orbital elements.
    
    This function solves Kepler's equation to find the position of an orbiting body 
    at a given time, then projects the position onto the sky plane to calculate 
    observable quantities like separation and position angle.
    
    Parameters
    ----------
    sma : float or ndarray
        Semi-major axis in astronomical units (AU)
    e : float or ndarray
        Orbital eccentricity (0 ≤ e < 1)
    i : float or ndarray
        Inclination in degrees (0° to 180°)
    aop : float or ndarray
        Argument of periastron in degrees (0° to 360°)
    pan : float or ndarray
        Position angle of ascending node in degrees (0° to 360°)
    epp : float or ndarray
        Epoch of periastron passage in years
    P : float or ndarray
        Orbital period in years
    t : float or ndarray
        Observation time in years
    distance_pc : float
        Distance to system in parsecs
    kepler_tol : float, optional
        Convergence tolerance for Kepler equation solver
        
    Returns
    -------
    tuple of ndarrays
        - sep_angle_arcsec : Projected separation in arcseconds
        - position_angle_deg : Position angle in degrees (0° to 360°, measured East of North)
        - r : Orbital radius in AU
        - phFunc : Phase function for reflected light calculations
    """
    # Input validation
    if not isinstance(distance_pc, (int, float)) or distance_pc <= 0:
        raise ValueError("Distance must be a positive number")
        
    # Convert inputs to numpy arrays
    sma = np.asarray(sma)
    e = np.asarray(e)
    i = np.asarray(i)
    aop = np.asarray(aop)
    pan = np.asarray(pan)
    epp = np.asarray(epp)
    P = np.asarray(P)
    t = np.asarray(t)

    # Calculate position in sky plane
    x_sky, y_sky, z_sky, r = _calculate_orbital_position(
        sma, e, i, aop, pan, epp, P, t, kepler_tol
    )

    # Calculate separation and position angle
    r2D = np.sqrt(x_sky**2 + y_sky**2)
    sep_angle_arcsec = r2D / distance_pc
    
    position_angle_rad = np.arctan2(-x_sky, y_sky)
    position_angle_deg = (np.degrees(position_angle_rad) + 360) % 360

    # Calculate phase function
    r_vector = np.array([y_sky, x_sky, z_sky])
    r_mag = np.linalg.norm(r_vector)
    r_hat = r_vector / r_mag
    s_hat = np.array([0, 0, -1])

    cos_phAngle = np.dot(-r_hat.T, s_hat)
    sin_phAngle = np.sqrt(1 - cos_phAngle**2)
    phAngle = np.arctan2(sin_phAngle, cos_phAngle)
    phFunc = (np.sin(phAngle) + (np.pi - phAngle) * np.cos(phAngle)) / np.pi

    return sep_angle_arcsec, position_angle_deg, r, phFunc

def solve_orbit_XY(sma, e, i, aop, pan, epp, P, t, kepler_tol=1e-4):
    """
    Calculate Cartesian coordinates in the sky plane for orbit visualization.
    
    Similar to solve_orbit(), but returns the x-y coordinates in the sky plane 
    instead of polar coordinates. This is particularly useful for plotting orbits
    and visualizing the projected orbital motion.
    
    Parameters
    ----------
    sma : float or ndarray
        Semi-major axis in astronomical units (AU)
    e : float or ndarray
        Orbital eccentricity (0 ≤ e < 1)
    i : float or ndarray
        Inclination in degrees (0° to 180°)
    aop : float or ndarray
        Argument of periastron in degrees (0° to 360°)
    pan : float or ndarray
        Position angle of ascending node in degrees (0° to 360°)
    epp : float or ndarray
        Epoch of periastron passage in years
    P : float or ndarray
        Orbital period in years
    t : float or ndarray
        Observation time in years
    kepler_tol : float, optional
        Convergence tolerance for Kepler equation solver
        
    Returns
    -------
    tuple of ndarrays
        - x_sky : X coordinates in the sky plane (AU, positive toward East)
        - y_sky : Y coordinates in the sky plane (AU, positive toward North)
    
    Notes
    -----
    The sky plane coordinate system follows the standard convention where:
    - +x axis points East
    - +y axis points North
    - +z axis points away from the observer (not returned)
    """
    # Convert inputs to numpy arrays
    sma = np.asarray(sma)
    e = np.asarray(e)
    i = np.asarray(i)
    aop = np.asarray(aop)
    pan = np.asarray(pan)
    epp = np.asarray(epp)
    P = np.asarray(P)
    t = np.asarray(t)

    # Calculate position in sky plane
    x_sky, y_sky, z_sky, _ = _calculate_orbital_position(
        sma, e, i, aop, pan, epp, P, t, kepler_tol
    )

    return x_sky, y_sky



def rotate_coordinates_vectorized(x_orb, y_orb, i, aop, pan):
    """
    Vectorized version of coordinate rotation.
    Handles multiple sets of orbital parameters simultaneously.
    
    Parameters
    ----------
    x_orb : ndarray (n_orbits,)
        x-coordinates in orbital plane
    y_orb : ndarray (n_orbits,)
        y-coordinates in orbital plane
    i : ndarray (n_orbits,)
        Inclination angles in radians
    aop : ndarray (n_orbits,)
        Argument of periastron in radians
    pan : ndarray (n_orbits,)
        Position angle of nodes in radians
        
    Returns
    -------
    tuple of ndarrays
        (x_sky, y_sky) coordinates for all orbits
    """
    # Compute trig functions once
    cos_aop, sin_aop = np.cos(aop), np.sin(aop)
    cos_pan, sin_pan = np.cos(pan), np.sin(pan)
    cos_i, sin_i = np.cos(i), np.sin(i)

    # Argument of periastron rotation
    x1 = x_orb * cos_aop - y_orb * sin_aop
    y1 = x_orb * sin_aop + y_orb * cos_aop
    
    # Inclination rotation
    x2 = x1
    y2 = y1 * cos_i
    
    # Node rotation
    x_sky = x2 * cos_pan - y2 * sin_pan
    y_sky = x2 * sin_pan + y2 * cos_pan
    
    return x_sky, y_sky

def _calculate_orbital_position_vectorized(sma, e, i, aop, pan, epp, P, t, kepler_tol=1e-4):
    """
    Vectorized calculation of orbital positions for multiple parameter sets.
    
    Parameters
    ----------
    sma, e, i, aop, pan, epp, P : ndarrays of shape (n_orbits,)
        Orbital parameters for multiple orbits
    t : float or array-like
        Time(s) of observation
    kepler_tol : float
        Convergence tolerance for Kepler solver
        
    Returns
    -------
    tuple of ndarrays
        (x_sky, y_sky) for all orbits
    """
    # Convert angles to radians if needed
    i_rad, aop_rad, pan_rad = np.radians([i, aop, pan])
    
    # Calculate mean anomaly for all orbits
    M = np.where(P != 0, 2 * np.pi * (t - epp) / P, np.nan)
    
    # Initial guess for eccentric anomaly
    E = M + e * np.sin(M) + 0.5 * e**2 * np.sin(2*M)
    
    # Newton-Raphson iteration for all orbits simultaneously
    for _ in range(100):
        delta = E - e * np.sin(E) - M
        E_new = E - delta / (1 - e * np.cos(E))
        
        if np.all(np.abs(E_new - E) < kepler_tol):
            break
        E = E_new
    else:
        warnings.warn("Kepler solver did not converge to desired tolerance")
    
    # True anomaly using stable formula
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), 
                       np.sqrt(1 - e) * np.cos(E/2))
    
    # Orbital radius
    r = sma * (1 - e**2) / (1 + e * np.cos(nu))
    
    # Position in orbital plane
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)
    
    # Apply rotations and return only sky plane coordinates
    x_sky, y_sky = rotate_coordinates_vectorized(x_orb, y_orb, i_rad, aop_rad, pan_rad)
    
    return x_sky, y_sky

def solve_orbit_vectorized(params_array, t, distance_pc, kepler_tol=1e-4):
    """
    Vectorized version of solve_orbit that handles multiple parameter sets.
    Only computes separation and position angle.
    
    Parameters
    ----------
    params_array : ndarray
        Array of shape (n_orbits, 7) containing orbital parameters
        Each row: [sma, e, i, aop, pan, epp, P]
    t : float or array-like
        Time(s) of observation
    distance_pc : float
        Distance to system in parsecs
    kepler_tol : float, optional
        Convergence tolerance for Kepler solver
        
    Returns
    -------
    tuple of ndarrays
        (sep_angle_arcsec, position_angle_deg) for all orbits
    """
    if not isinstance(distance_pc, (int, float)) or distance_pc <= 0:
        raise ValueError("Distance must be a positive number")
    
    # Unpack parameters
    sma, e, i, aop, pan, epp, P = params_array.T
    
    # Calculate positions for all orbits
    x_sky, y_sky = _calculate_orbital_position_vectorized(
        sma, e, i, aop, pan, epp, P, t, kepler_tol
    )
    
    # Calculate separations and position angles
    r2D = np.sqrt(x_sky**2 + y_sky**2)
    sep_angle_arcsec = r2D / distance_pc
    
    position_angle_rad = np.arctan2(-x_sky, y_sky)
    position_angle_deg = (np.degrees(position_angle_rad) + 360) % 360
    
    return sep_angle_arcsec, position_angle_deg

def solve_all_epochs_vectorized(params_batch, obs_times, dStar):
    """
    Vectorized computation of orbital predictions for all epochs at once.
    
    Parameters
    ----------
    params_batch : ndarray
        Array of shape (n_orbits, 7) containing orbital parameters
        [sma, e, inc, aop, pan, epp, P]
    obs_times : ndarray
        Array of observation times
    dStar : float
        Distance to system in parsecs
        
    Returns
    -------
    tuple
        (sep_predictions, pa_predictions)
        Each array has shape (n_orbits, n_epochs)
    """
    n_orbits = len(params_batch)
    n_epochs = len(obs_times)
    
    # Unpack parameters and reshape for broadcasting
    sma = params_batch[:, 0][:, np.newaxis]  # Shape: (n_orbits, 1)
    e = params_batch[:, 1][:, np.newaxis]
    inc = np.radians(params_batch[:, 2])[:, np.newaxis]
    aop = np.radians(params_batch[:, 3])[:, np.newaxis]
    pan = np.radians(params_batch[:, 4])[:, np.newaxis]
    epp = params_batch[:, 5][:, np.newaxis]
    P = params_batch[:, 6][:, np.newaxis]
    
    # Pre-compute trig functions
    cos_inc = np.cos(inc)
    sin_inc = np.sin(inc)
    cos_aop = np.cos(aop)
    sin_aop = np.sin(aop)
    cos_pan = np.cos(pan)
    sin_pan = np.sin(pan)
    
    # Calculate mean anomaly for all orbits and epochs
    M = 2 * np.pi * (obs_times[np.newaxis, :] - epp) / P  # Shape: (n_orbits, n_epochs)
    
    # Solve Kepler's equation with optimized initial guess
    E = M + e * np.sin(M) + 0.5 * e**2 * np.sin(2*M)
    
    # Newton-Raphson iteration (vectorized over all orbits and epochs)
    for _ in range(10):  # Usually converges in 3-4 iterations
        delta = E - e * np.sin(E) - M
        E_new = E - delta / (1 - e * np.cos(E))
        if np.all(np.abs(E_new - E) < 1e-8):
            break
        E = E_new
    
    # True anomaly using vectorized computation
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2),
                       np.sqrt(1 - e) * np.cos(E/2))
    
    # Orbital radius
    r = sma * (1 - e**2) / (1 + e * np.cos(nu))
    
    # Position in orbital plane
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)
    
    # Apply rotations (vectorized over all orbits and epochs)
    # First rotation (argument of periastron)
    x1 = x_orb * cos_aop - y_orb * sin_aop
    y1 = x_orb * sin_aop + y_orb * cos_aop
    
    # Second rotation (inclination)
    x2 = x1
    y2 = y1 * cos_inc
    
    # Final rotation (position angle of nodes)
    x_sky = x2 * cos_pan - y2 * sin_pan
    y_sky = x2 * sin_pan + y2 * cos_pan
    
    # Calculate separations and position angles
    r2D = np.sqrt(x_sky**2 + y_sky**2)
    sep_predictions = r2D / dStar
    
    pa_predictions = np.degrees(np.arctan2(-x_sky, y_sky)) % 360
    
    return sep_predictions, pa_predictions

def solve_orbit_vectorized_full(params_batch, t, distance_pc):
    """
    Vectorized computation of orbital parameters including phase function.
    
    Parameters
    ----------
    params_batch : ndarray
        Array of shape (n_orbits, 7) containing orbital parameters
        [sma, e, inc, aop, pan, epp, P]
    t : float
        Observation time
    distance_pc : ndarray
        Array of distances to systems in parsecs
        
    Returns
    -------
    tuple
        (sep_angle_arcsec, position_angle_deg, r3d, phase_func)
    """
    
    # Unpack parameters and reshape for broadcasting
    sma = params_batch[:, 0]
    e = params_batch[:, 1]
    inc = np.radians(params_batch[:, 2])
    aop = np.radians(params_batch[:, 3])
    pan = np.radians(params_batch[:, 4])
    epp = params_batch[:, 5]
    P = params_batch[:, 6]
    
    # Pre-compute trig functions
    cos_inc = np.cos(inc)
    sin_inc = np.sin(inc)
    cos_aop = np.cos(aop)
    sin_aop = np.sin(aop)
    cos_pan = np.cos(pan)
    sin_pan = np.sin(pan)
    
    # Calculate mean anomaly
    M = 2 * np.pi * (t - epp) / P
    
    # Solve Kepler's equation with optimized initial guess
    E = M + e * np.sin(M) + 0.5 * e**2 * np.sin(2*M)
    
    # Newton-Raphson iteration
    for _ in range(20):
        delta = E - e * np.sin(E) - M
        E_new = E - delta / (1 - e * np.cos(E))
        if np.all(np.abs(E_new - E) < 1e-3):
            break
        E = E_new
    
    # True anomaly
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2),
                       np.sqrt(1 - e) * np.cos(E/2))
    
    # Orbital radius
    r = sma * (1 - e**2) / (1 + e * np.cos(nu))
    
    # Position in orbital plane
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)
    
    # Apply rotations
    # First rotation (argument of periastron)
    x1 = x_orb * cos_aop - y_orb * sin_aop
    y1 = x_orb * sin_aop + y_orb * cos_aop
    
    # Second rotation (inclination)
    x2 = x1
    y2 = y1 * cos_inc
    z2 = y1 * sin_inc
    
    # Final rotation (position angle of nodes)
    x_sky = x2 * cos_pan - y2 * sin_pan
    y_sky = x2 * sin_pan + y2 * cos_pan
    z_sky = z2
    
    # Calculate separations and position angles
    r2D = np.sqrt(x_sky**2 + y_sky**2)
    sep_angle_arcsec = r2D / distance_pc
    
    pa_deg = np.degrees(np.arctan2(-x_sky, y_sky)) % 360
    
    # Calculate 3D distance and phase function
    r3d = np.sqrt(x_sky**2 + y_sky**2 + z_sky**2)
    
    # Phase angle calculation
    r_vector = np.column_stack([y_sky, x_sky, z_sky])
    r_mag = np.linalg.norm(r_vector, axis=1)
    r_hat = r_vector / r_mag[:, np.newaxis]
    s_hat = np.array([0, 0, -1])
    
    cos_phangle = np.dot(r_hat, -s_hat)
    sin_phangle = np.sqrt(1 - cos_phangle**2)
    phangle = np.arctan2(sin_phangle, cos_phangle)
    phase_func = (sin_phangle + (np.pi - phangle) * cos_phangle) / np.pi
    
    return sep_angle_arcsec, pa_deg, r3d, phase_func

def solve_all_epochs_vectorized_full(params_batch, obs_times, dStar):
    """
    Fully vectorized orbital propagation for all posterior samples and epochs.

    Parameters
    ----------
    params_batch : (N, 7) array
        Columns: [sma_AU, e, inc_deg, aop_deg, pan_deg, epp_yr, P_yr]
    obs_times : (T,) array
        Observation times (years).
    dStar : float
        Distance to star in parsecs.

    Returns
    -------
    sep_arcsec : (N, T) array
        Apparent separation in arcseconds.
    pa_deg : (N, T) array
        Position angle in degrees (same convention as solve_orbit).
    r3D_AU : (N, T) array
        3D star–planet distance in AU.
    phase_func : (N, T) array
        Lambertian phase function f(alpha) = (sin α + (π−α) cos α)/π.
    """
    import numpy as np

    params_batch = np.asarray(params_batch, dtype=float)
    t = np.asarray(obs_times, dtype=float)[None, :]          # (1, T)
    N = params_batch.shape[0]
    if params_batch.shape[1] < 7:
        raise ValueError("params_batch must have 7 columns: [sma,e,inc,aop,pan,epp,P]")

    # Unpack, shape (N,1) so they broadcast across T
    sma = params_batch[:, 0][:, None]                        # AU
    e   = params_batch[:, 1][:, None]
    inc = np.radians(params_batch[:, 2])[:, None]
    aop = np.radians(params_batch[:, 3])[:, None]
    pan = np.radians(params_batch[:, 4])[:, None]
    epp = params_batch[:, 5][:, None]                        # yr
    P   = params_batch[:, 6][:, None]                        # yr

    # Precompute trig
    cos_inc, sin_inc = np.cos(inc), np.sin(inc)
    cos_aop, sin_aop = np.cos(aop), np.sin(aop)
    cos_pan, sin_pan = np.cos(pan), np.sin(pan)

    # Mean anomaly for all (N,T)
    M = 2.0 * np.pi * (t - epp) / P

    # Kepler solve (vectorized Newton across (N,T))
    E = M + e*np.sin(M) + 0.5*e*e*np.sin(2.0*M)              # good initial guess
    for _ in range(20):
        f  = E - e*np.sin(E) - M
        fp = 1.0 - e*np.cos(E)
        dE = f / fp
        E  = E - dE
        if np.max(np.abs(dE)) < 1e-10:
            break

    # True anomaly and radius
    sqrt1pe = np.sqrt(1.0 + e)
    sqrt1me = np.sqrt(1.0 - e)
    nu = 2.0 * np.arctan2(sqrt1pe * np.sin(E/2.0), sqrt1me * np.cos(E/2.0))
    r  = sma * (1.0 - e*e) / (1.0 + e*np.cos(nu))            # AU, shape (N,T)

    # Orbital plane coords
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)

    # Rotate by argument of periastron
    x1 = x_orb * cos_aop - y_orb * sin_aop
    y1 = x_orb * sin_aop + y_orb * cos_aop

    # Inclination
    x2 = x1
    y2 = y1 * cos_inc
    z2 = y1 * sin_inc

    # Rotate by longitude of ascending node (position angle of nodes)
    x_sky = x2 * cos_pan - y2 * sin_pan
    y_sky = x2 * sin_pan + y2 * cos_pan
    z_sky = z2

    # Apparent separation (AU -> arcsec via AU/pc)
    r2D = np.hypot(x_sky, y_sky)
    sep_arcsec = r2D / float(dStar)

    # PA convention consistent with your single-epoch function
    pa_deg = (np.degrees(np.arctan2(-x_sky, y_sky)) % 360.0)

    # 3D distance
    r3D_AU = np.sqrt(x_sky*x_sky + y_sky*y_sky + z_sky*z_sky)

    # Lambertian phase function; star->observer along -z, so cos(alpha)=z/r
    # (because -s_hat = +z-hat)
    cos_alpha = np.clip(z_sky / r3D_AU, -1.0, 1.0)
    sin_alpha = np.sqrt(1.0 - cos_alpha*cos_alpha)
    alpha = np.arctan2(sin_alpha, cos_alpha)
    phase_func = (sin_alpha + (np.pi - alpha) * cos_alpha) / np.pi

    return sep_arcsec, pa_deg, r3D_AU, phase_func