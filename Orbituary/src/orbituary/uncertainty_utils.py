import numpy as np

def calculate_circular_uncertainty(angles):
    """
    Calculate uncertainty for circular quantities (angles).
    
    Parameters
    ----------
    angles : array-like
        Array of angles in degrees
        
    Returns
    -------
    float
        Uncertainty in degrees based on circular statistics
    """
    # Convert to radians
    angles_rad = np.radians(angles)
    
    # Calculate mean direction
    x = np.mean(np.cos(angles_rad))
    y = np.mean(np.sin(angles_rad))
    
    # Calculate circular standard deviation
    R = np.sqrt(x**2 + y**2)  # Mean resultant length

    # Add handling for complete dispersion
    if R < 1e-10:  # If R is effectively zero
        return 180.0  # Maximum possible uncertainty
    
    circular_std = np.sqrt(-2 * np.log(R))  # In radians
    
    return np.degrees(circular_std)

def calculate_parameter_uncertainties(samples):
    """
    Calculate uncertainties for all parameters, handling angles appropriately.
    
    Parameters
    ----------
    samples : ndarray
        Array of shape (n_samples, n_params) containing parameter values
    param_names : list
        List of parameter names corresponding to columns in samples
        
    Returns
    -------
    ndarray
        Array of uncertainties for each parameter
    """
    uncertainties = np.zeros(samples.shape[1])

    param_names = ['sma', 'ecc', 'inc', 'aop', 'pan', 'epp', 'P']
    
    for i, param in enumerate(param_names):
        if param in ['aop', 'pan']:  # Angle parameters
            uncertainties[i] = calculate_circular_uncertainty(samples[:, i])
        else:  # Non-angle parameters
            p16, p50, p84 = np.percentile(samples[:, i], [16, 50, 84])
            uncertainties[i] = np.mean([p50 - p16, p84 - p50])
            
    return uncertainties
