import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from orbituary.ofti_algorithms import fit_single_epoch, fit_two_epochs
from orbituary.solve_orbit import solve_orbit_XY
from orbituary.mcmc_solver import MCMC
from orbituary.uncertainty_utils import calculate_parameter_uncertainties

def fit_orbit(
    times: np.ndarray,
    separations: np.ndarray,
    position_angles: np.ndarray,
    stellar_mass: float,
    distance_parsec: float,
    *,  # Force remaining arguments to be keyword-only
    sep_uncertainty: float = 0.0005,
    pa_uncertainty: float = 0.1,
    num_orbits: int = 5000,
    max_ofti_time: float = 300,  # Maximum time for OFTI in seconds
    mcmc_walkers: int = 50,     # Number of walkers for MCMC
    mcmc_steps: int = 30000,      # Number of MCMC steps
    mcmc_burnin: int = 5000,     # Number of burn-in steps to discard
    all_epochs: bool = False,
    progress_bar: bool = True     # Whether to show progress updates
):
    
    """
    Fit Keplerian orbits to astrometric observations using OFTI/MCMC methods.
    
    Parameters
    ----------
    times : np.ndarray
        Observation times in decimal years
    separations : np.ndarray
        Projected separations in arcseconds
    position_angles : np.ndarray
        Position angles in degrees (0-360°, measured East of North)
    stellar_mass : float
        Mass of the total system (or the central star) in solar masses
    distance_parsec : float
        Distance to the system in parsecs
    sep_uncertainty : float, optional
        Uncertainty in separation measurements (arcseconds)
        Default is 0.003 arcsec
    pa_uncertainty : float, optional
        Uncertainty in position angle measurements (degrees)
        Default is 0.1 degrees
    num_orbits : int, optional
        Number of orbital solutions to generate
        Default is 1000
    max_ofti_time : float, optional
        Maximum computation time for OFTI algorithm before we switch to MCMC in seconds
        This is applicable when fitting two (or more) epochs of data
        Default is 300 seconds (5 minutes)
    mcmc_steps : int, optional
        Number of steps for MCMC chain
        Only applies when using MCMC (>2 epochs)
        Default is 8000
    mcmc_burnin : int, optional
        Number of initial MCMC steps to discard as burn-in
        Only applies when using MCMC (>2 epochs)
        Default is 2000
    output_dir : str or Path, optional
        Directory to save the output CSV file
        If None, current working directory is used
    output_filename : str, optional
        Name of the output CSV file
        If None, a default name based on timestamps will be used
    
    progress_bar : bool, optional
        Whether to display progress updates during fitting
        Default is True
        
    Returns
    -------
    orbits : pd.DataFrame
        DataFrame containing the fitted orbital parameters
        Columns: semi-major axis (AU), eccentricity, inclination (deg),
                argument of periastron (deg), position angle of nodes (deg),
                epoch of periastron passage (year), period (year), and 7 columns 
                of associated uncertainties
        
    Raises
    ------
    ValueError
        If input arrays have inconsistent shapes
        If physical parameters are outside valid ranges
        If file paths are invalid
    TypeError
        If inputs have incorrect data types
    
    Notes
    -----
    The algorithm automatically switches between OFTI and MCMC methods
    depending on the number of epochs:
    - For 1 epoch: Uses OFTI to rescale sma and pan, ensuring all orbits 
      pass through the 1 datapoint
    - For 2 epochs: Uses OFTI with rejection sampling
    - For >2 epochs: Switches to MCMC
    
    For two-epoch OFTI, the algorithm will switch to MCMC if no valid
    orbits are found within max_ofti_time seconds.
    
    Examples
    --------
    >>> # Basic usage with minimum parameters
    >>> times = np.array([2020.0, 2021.0])
    >>> seps = np.array([1.5, 1.6])
    >>> pas = np.array([45.0, 46.0])
    >>> mStar = 1.0 (Mass of the system in solar masses)
    >>> dist = 10.0 (Distance to the system in pc)
    >>> orbits, stats = fit_orbit(times, seps, pas, mStar, dist)

    
    >>> # Custom uncertainties and output location
    >>> orbits, stats = fit_orbit(
    ...     times, seps, pas, 1.0, 10.0,
    ...     sep_uncertainty=0.005,
    ...     pa_uncertainty=0.2,
    ...     output_dir='my_results'
    ... )

    Note that all the non-required arguments have to be specified with a keyword
    when calling the function. See the example above for uncertainties.
    """

    lp_array = None

    # Run validations
    _validate_input_types(times, separations, position_angles, stellar_mass, distance_parsec)
    _validate_physical_ranges(times, separations, position_angles, stellar_mass, distance_parsec, sep_uncertainty, pa_uncertainty)

    n_epochs = len(times)

    switch_mcmc = False

    all_epoch_results = []

    # Initialize progress tracking if enabled
    progress = OrbitProgress(n_epochs) if progress_bar else None

    try:
        for i in range(n_epochs):
            t_obs_curr = times[:i+1]
            sep_obs_curr = separations[:i+1]
            pa_obs_curr = position_angles[:i+1]

            success = True  # Track success for this epoch
            try:
                if i < 2:
                    if len(t_obs_curr) == 1:
                        if progress:
                            progress.start_computation(5000000)  # Typical number for single epoch
                        accepted_orbits = fit_single_epoch(5000000, t_obs_curr[0], 
                                                         sep_obs_curr[0], pa_obs_curr[0], 
                                                         stellar_mass, distance_parsec)
                    else:
                        if progress:
                            progress.start_computation(num_orbits)
                        accepted_orbits, switch_mcmc = fit_two_epochs(
                            num_orbits, 1000, t_obs_curr, sep_obs_curr, pa_obs_curr,
                            stellar_mass, distance_parsec, old_orbits,
                            max_ofti_time=max_ofti_time
                        )

                if i >= 2 or switch_mcmc:
                    if progress:
                        progress.start_computation(mcmc_steps)

                    accepted_orbits, orbit_uncertainties = MCMC(
                        num_orbits, sep_obs_curr, pa_obs_curr, t_obs_curr,
                        distance_parsec, stellar_mass, accepted_orbits,
                        nwalkers=mcmc_walkers, max_steps=mcmc_steps,
                        burnin=mcmc_burnin, progress=progress
                        )

                accepted_orbits = np.array(accepted_orbits)
                uncertainties = calculate_parameter_uncertainties(accepted_orbits)
                orbit_uncertainties = np.tile(uncertainties, (len(accepted_orbits), 1))

                # Create DataFrame for the current epoch
                param_names = ['sma', 'ecc', 'inc', 'aop', 'pan', 'epp', 'p']
                current_df = pd.DataFrame(accepted_orbits, columns=param_names)

                # Add uncertainties
                for j, param in enumerate(param_names):
                    current_df[f'{param}_err'] = orbit_uncertainties[:, j]

                if i == 0:
                    all_epoch_results.append(current_df.iloc[:num_orbits])
                else:
                    all_epoch_results.append(current_df)

                old_orbits = accepted_orbits

            except Exception as e:
                success = False
                print(f"Error in epoch {i+1}: {str(e)}")
                raise

            finally:
                if progress:
                    progress.finish_epoch(success=success)
    finally:
        if progress:
            progress.finish()

        accepted_orbits = np.array(accepted_orbits)

        # Create DataFrame for the current epoch
        param_names = ['sma', 'ecc', 'inc', 'aop', 'pan', 'epp', 'p']
        current_df = pd.DataFrame(accepted_orbits, columns=param_names)

        # Add the uncertainties for the current epoch
        for j, param in enumerate(param_names):
            current_df[f'{param}_err'] = orbit_uncertainties[:, j]

        old_orbits = accepted_orbits

    if lp_array is not None:
        return all_epoch_results if all_epochs else all_epoch_results[-1], lp_array
    
    return all_epoch_results if all_epochs else all_epoch_results[-1]

def setup_plotting_style():
    """Set up the plotting style to match the example."""
    plt.rcParams['xtick.color'] = "323034"
    plt.rcParams['ytick.color'] = "323034"
    plt.rcParams['text.color'] = "323034"
    plt.rcParams['lines.markeredgecolor'] = "black"
    plt.rcParams['patch.facecolor'] = "bc80bd"
    plt.rcParams['patch.force_edgecolor'] = True
    plt.rcParams['patch.linewidth'] = 0.8
    plt.rcParams['scatter.edgecolors'] = "black"
    plt.rcParams['grid.color'] = "b1afb5"
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.title_fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.prop_cycle'] = "(cycler('color', ['1f77b4', 'fdb462', 'b3de69', 'fb8072', 'bc80bd', 'fccde5', '8dd3c7', 'ffed6f', 'bebada', '80b1d3', 'ccebc5', 'd9d9d9']))"
    plt.rcParams['mathtext.fontset'] = "stix"
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = ['Calibri']
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.fontsize'] = 13
    plt.rcParams['legend.edgecolor'] = "black"
    plt.rcParams['legend.borderpad'] = 0.2
    plt.rcParams['legend.columnspacing'] = 1.5
    plt.rcParams['legend.labelspacing'] = 0.4
    plt.rcParams['text.usetex'] = False
    plt.rcParams['axes.labelsize'] = 17
    plt.rcParams['axes.titlelocation'] = "center"
    plt.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams['axes.autolimit_mode'] = "round_numbers"
    plt.rcParams['axes.labelpad'] = 3
    plt.rcParams['axes.formatter.limits'] = (-4, 4)
    plt.rcParams['axes.labelcolor'] = "black"
    plt.rcParams['axes.edgecolor'] = "black"
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.spines.right'] = True
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.top'] = True
    plt.rcParams['figure.titlesize'] = 18
    plt.rcParams['figure.dpi'] = 300

def plot_complete_fit(orbit_df, observation_times, sep_obs, pa_obs, dStar, L,
                     n_orbits=100, time_span=None, save_path=None):
    """
    Plot a single figure showing orbital fits using all epochs at once.
    
    Parameters
    ----------
    orbit_df : pd.DataFrame
        DataFrame containing orbital parameters
    observation_times : array-like
        Array of observation times
    sep_obs : array-like
        Array of observed separations
    pa_obs : array-like
        Array of observed position angles
    dStar : float
        Distance to the system in parsecs
    n_orbits : int, optional
        Number of orbits to plot (default: 100)
    time_span : tuple, optional
        (start_year, end_year) for orbit plotting. If None, uses default span
    save_path : str, optional
        If provided, saves the figure to this path
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot
    """
    setup_plotting_style()
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    # Set up time array for plotting orbits
    if time_span is None:
        time_start = min(observation_times) - 1
        time_end = max(observation_times) + 1
    else:
        time_start, time_end = time_span
    time = np.linspace(time_start, time_end, 1000)
    
    # Plot sample of orbits
    orbit_sample = orbit_df.iloc[:n_orbits, :7].values
    for orbit in orbit_sample:
        x_sky, y_sky = solve_orbit_XY(*orbit, time)
        ax.plot(x_sky, y_sky, color='grey', lw=0.5, alpha=0.5)
    
    # Plot observed positions
    for t, sep, pa in zip(observation_times, sep_obs, pa_obs):
        x_obs = -sep * np.sin(np.radians(pa)) * dStar
        y_obs = sep * np.cos(np.radians(pa)) * dStar
        ax.scatter(x_obs, y_obs, color='red', zorder=3, edgecolor='black')
    
    # Set plot limits and labels
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_xlabel("$x_{sky}$ (AU)")
    ax.set_ylabel("$y_{sky}$ (AU)")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

def plot_progressive_fits(orbit_dfs, observation_times, sep_obs, pa_obs, dStar,
                         n_orbits=100, time_span=None, save_path=None):
    """
    Plot the progression of orbital fits as epochs are added.
    
    Parameters
    ----------
    orbit_dfs : list of pd.DataFrame
        List of DataFrames containing orbital parameters for each epoch
    observation_times : array-like
        Array of observation times
    sep_obs : array-like
        Array of observed separations
    pa_obs : array-like
        Array of observed position angles
    dStar : float
        Distance to the system in parsecs
    n_orbits : int, optional
        Number of orbits to plot per subplot (default: 100)
    time_span : tuple, optional
        (start_year, end_year) for orbit plotting. If None, uses default span
    save_path : str, optional
        If provided, saves the figure to this path
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the subplots
    """
    setup_plotting_style()
    
    n_epochs = len(orbit_dfs)
    n_rows = 2
    n_cols = (n_epochs + 1) // 2  # Rounds up for odd numbers
    
    # Create figure with appropriate size
    fig = plt.figure(figsize=(6*n_cols, 10))
    
    # Set up time array for plotting orbits
    if time_span is None:
        time_start = min(observation_times) - 1
        time_end = max(observation_times) + 1
    else:
        time_start, time_end = time_span
    time = np.linspace(time_start, time_end, 1000)
    
    # Create subplots
    for i in range(n_epochs):
        # Calculate subplot position
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        
        # Plot sample of orbits for current epoch
        orbit_sample = orbit_dfs[i].iloc[:n_orbits,:7].values
        for orbit in orbit_sample:

            x_sky, y_sky = solve_orbit_XY(*orbit, time)
            ax.plot(x_sky, y_sky, color='grey', lw=0.5, alpha=0.5)
        
        # Plot observed positions up to current epoch
        x_obs = -sep_obs[:i+1] * np.sin(np.radians(pa_obs[:i+1])) * dStar
        y_obs = sep_obs[:i+1] * np.cos(np.radians(pa_obs[:i+1])) * dStar
        ax.scatter(x_obs, y_obs, color='red', zorder=3, edgecolor='black')

        if i == 0:
            ax.scatter(x_obs, y_obs, color='blue', zorder=3, edgecolor='black')
        else:
            ax.scatter(x_obs[-1], y_obs[-1], color='blue', zorder=3, edgecolor='black')
        
        # Set plot limits and labels
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_xlabel("$x_{sky}$ (AU)")
        ax.set_ylabel("$y_{sky}$ (AU)")
        ax.set_title(f"Epoch {i+1}")
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_progressive_fits_with_HZ(orbit_dfs, observation_times, sep_obs, pa_obs, dStar, L,
                         n_orbits=100, time_span=None, save_path=None):
    """
    Plot the progression of orbital fits as epochs are added.
    
    Parameters
    ----------
    orbit_dfs : list of pd.DataFrame
        List of DataFrames containing orbital parameters for each epoch
    observation_times : array-like
        Array of observation times
    sep_obs : array-like
        Array of observed separations
    pa_obs : array-like
        Array of observed position angles
    dStar : float
        Distance to the system in parsecs
    n_orbits : int, optional
        Number of orbits to plot per subplot (default: 100)
    time_span : tuple, optional
        (start_year, end_year) for orbit plotting. If None, uses default span
    save_path : str, optional
        If provided, saves the figure to this path
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the subplots
    """
    setup_plotting_style()

    hz_inner = np.sqrt(L / 1.78)
    hz_outer = np.sqrt(L / 0.32)
    
    n_epochs = len(orbit_dfs)
    n_rows = 2
    n_cols = (n_epochs + 1) // 2  # Rounds up for odd numbers
    
    # Create figure with appropriate size
    fig = plt.figure(figsize=(6*n_cols, 10))
    
    # Set up time array for plotting orbits
    if time_span is None:
        time_start = min(observation_times) - 1
        time_end = max(observation_times) + 1
    else:
        time_start, time_end = time_span
    time = np.linspace(time_start, time_end, 1000)
    
    # Create subplots
    for i in range(n_epochs):
        # Calculate subplot position
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        
        # Plot sample of orbits for current epoch
        orbit_sample = orbit_dfs[i].iloc[:n_orbits,:7].values
        for orbit in orbit_sample:
            sma = orbit[0]
            ecc = orbit[1]

            peri = sma * (1-ecc)
            ap = sma * (1+ecc)

            x_sky, y_sky = solve_orbit_XY(*orbit, time)

            if peri > hz_inner and ap < hz_outer:
                ax.plot(x_sky, y_sky, color='#67C700', lw=0.5, alpha=0.5, zorder = 2)

            else:
                ax.plot(x_sky, y_sky, color='grey', lw=0.5, alpha=0.5, zorder = 2)
        
        # Plot observed positions up to current epoch
        x_obs = -sep_obs[:i+1] * np.sin(np.radians(pa_obs[:i+1])) * dStar
        y_obs = sep_obs[:i+1] * np.cos(np.radians(pa_obs[:i+1])) * dStar

        if i == 0:
            ax.scatter(x_obs, y_obs, color='blue', zorder=3, edgecolor='black')
        else:
            ax.scatter(x_obs[:-1], y_obs[:-1], color='red', zorder=3, edgecolor='black')
            ax.scatter(x_obs[-1], y_obs[-1], color='blue', zorder=3, edgecolor='black')

        
        # Set plot limits and labels
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_xlabel("$x_{sky}$ (AU)")
        ax.set_ylabel("$y_{sky}$ (AU)")
        ax.set_title(f"Epoch {i+1}")
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def _validate_input_types(times, separations, position_angles, stellar_mass, distance_parsec):
    """Validate input data types."""

    # Check array lengths match
    n_epochs = len(times)
    if not (len(separations) == len(position_angles) == n_epochs):
        raise ValueError(
            f"Input arrays must have same length. Got: times({len(times)}), "
            f"separations({len(separations)}), position_angles({len(position_angles)})")
    
    # Check if the inputs are numpy arrays, and they only contain numbers
    for name, arr in [
        ("times", times),
        ("separations", separations),
        ("position_angles", position_angles)
    ]:
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"{name} must be a numpy array, got {type(arr)}"
            )
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(
                f"{name} must contain numerical values, got {arr.dtype}"
            )
    
    # Check the scalar float inputs (mass and distance) are either integers or floats
    for name, val in [
        ("stellar_mass", stellar_mass),
        ("distance_parsec", distance_parsec)
    ]:
        if not isinstance(val, (int, float)):
            raise TypeError(
                f"{name} must be a number, got {type(val)}"
            )
        
def _validate_physical_ranges(times, separations, position_angles, stellar_mass, distance_parsec, sep_uncertainty, pa_uncertainty):
    """Validate that all inputs are within physically meaningful ranges."""
    
    # Check for NaN/inf values in the input arrays
    for name, arr in [
        ("times", times),
        ("separations", separations),
        ("position_angles", position_angles)
    ]:
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or infinite values")
    
    # Validate ranges
    if np.any(separations < 0):
        raise ValueError("Separations must be non-negative")
        
    if np.any((position_angles < 0) | (position_angles >= 360)):
        raise ValueError("Position angles must be in range [0, 360)")
        
    if stellar_mass <= 0:
        raise ValueError("Stellar mass must be positive")
        
    if distance_parsec <= 0:
        raise ValueError("Distance must be positive")
        
    if sep_uncertainty <= 0:
        raise ValueError("Separation uncertainty must be positive")
        
    if pa_uncertainty <= 0:
        raise ValueError("Position angle uncertainty must be positive")
    
# Progress tracking class
class OrbitProgress:
    def __init__(self, n_epochs):
        """
        Simple progress tracking for orbit fitting.
        """
        self.n_epochs = n_epochs
        self.current_epoch = 0
        self.progress_bar = None
        self.min_update_interval = 0.1  # seconds
        self.is_mcmc = False

    def start_computation(self, n_steps):
        """Start progress bar for current computation."""
        self.current_epoch += 1
        if self.progress_bar:
            self.progress_bar.close()
        
        # Detect if this is an MCMC epoch
        self.is_mcmc = n_steps == 8000  # Standard MCMC steps
        
        if not self.is_mcmc:
            # For OFTI epochs, scale down for responsiveness
            if n_steps > 10000:  # Single epoch OFTI
                display_steps = 100
            else:  # Two epoch OFTI
                display_steps = n_steps
        else:
            # For MCMC, use actual steps
            display_steps = n_steps
            
        self.progress_bar = tqdm(total=display_steps,
                               desc=f"Epoch {self.current_epoch}/{self.n_epochs}",
                               bar_format='{desc:<15}: {percentage:3.0f}%|{bar:30}{r_bar}',
                               mininterval=self.min_update_interval)
        
        # Store the scale factor for updates
        self.update_scale = n_steps / display_steps if not self.is_mcmc else 1
    
    def update_computation(self, n=1):
        """Update current computation progress."""
        if self.progress_bar:
            if not self.is_mcmc:
                # Scale down the update for OFTI
                scaled_update = max(1, int(n / self.update_scale))
                self.progress_bar.update(scaled_update)
            else:
                # Direct update for MCMC
                self.progress_bar.update(n)
    
    def finish_epoch(self, success=True):
        """Complete current epoch."""
        if self.progress_bar:
            # Ensure the progress bar shows completion
            self.progress_bar.n = self.progress_bar.total
            self.progress_bar.refresh()
            self.progress_bar.close()
            self.progress_bar = None
            
        # Print status on new line
        if success:
            tqdm.write(f"Epoch {self.current_epoch}/{self.n_epochs} completed successfully")
        else:
            tqdm.write(f"Epoch {self.current_epoch}/{self.n_epochs} failed to converge")
    
    def finish(self):
        """Clean up progress tracking."""
        if self.progress_bar:
            self.progress_bar.close()
      
"""
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
obs_log_path = os.path.join(current_dir, 'data', 'observing_log.csv')
stars_data_path = os.path.join(current_dir, 'data', 'stars_with_planets.csv')

dfObs = pd.read_csv(obs_log_path)
observation_times = np.array(dfObs['LastObs'])  # Example observation times (years)
sep_obs = np.array(dfObs['Sep'])  # Example separations (arcsec)
pa_obs = np.array(dfObs['PA'])  # Example position angles (degrees)

print(len(dfObs))

dfData = pd.read_csv(stars_data_path)
Mstar = np.array(dfData['M(sol)'])[0]  # Stellar mass (solar masses)
dStar = np.array(dfData['d(pc)'])[0] 

fit = fit_orbit(observation_times, sep_obs, pa_obs, Mstar, dStar, all_epochs=True)

fig1 = plot_progressive_fits(fit, observation_times, sep_obs, pa_obs, dStar)
fig1.savefig('D:\Exoplanet-Stuff\Orbituary\src\orbituary\my_plot.png', dpi=300, bbox_inches='tight')

fig1 = plot_complete_fit(fit[5], observation_times, sep_obs, pa_obs, dStar)
fig1.savefig('D:\Exoplanet-Stuff\Orbituary\src\orbituary\my_plot2.png', dpi=300, bbox_inches='tight')
"""