import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

from .solve_orbit import calculate_sky_position

def setup_plotting_style():
    """Set up the plotting style to match the example."""
    plt.rcParams['xtick.color'] = "#323034"
    plt.rcParams['ytick.color'] = "#323034"
    plt.rcParams['text.color'] = "#323034"
    plt.rcParams['lines.markeredgecolor'] = "black"
    plt.rcParams['patch.facecolor'] = "#bc80bd"
    plt.rcParams['patch.force_edgecolor'] = True
    plt.rcParams['patch.linewidth'] = 0.8
    plt.rcParams['scatter.edgecolors'] = "black"
    plt.rcParams['grid.color'] = "#b1afb5"
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.title_fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.prop_cycle'] = cycler(color=[
        '#1f77b4', '#fdb462', '#b3de69', '#fb8072', '#bc80bd', '#fccde5',
        '#8dd3c7', '#ffed6f', '#bebada', '#80b1d3', '#ccebc5', '#d9d9d9'
    ])
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
        x_sky, y_sky, _, _ = calculate_sky_position(*orbit, time)
        ax.plot(x_sky, y_sky, color='grey', lw=0.5, alpha=0.5)
    
    # Plot observed positions
    for t, sep, pa in zip(observation_times, sep_obs, pa_obs):
        x_obs = sep * np.sin(np.radians(pa)) * dStar
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

            x_sky, y_sky, _, _ = calculate_sky_position(*orbit, time)
            ax.plot(x_sky, y_sky, color='grey', lw=0.5, alpha=0.5)
        
        # Plot observed positions up to current epoch
        x_obs = sep_obs[:i+1] * np.sin(np.radians(pa_obs[:i+1])) * dStar
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

            x_sky, y_sky, _, _ = calculate_sky_position(*orbit, time)

            if peri > hz_inner and ap < hz_outer:
                ax.plot(x_sky, y_sky, color='#67C700', lw=0.5, alpha=0.5, zorder = 2)

            else:
                ax.plot(x_sky, y_sky, color='grey', lw=0.5, alpha=0.5, zorder = 2)
        
        # Plot observed positions up to current epoch
        x_obs = sep_obs[:i+1] * np.sin(np.radians(pa_obs[:i+1])) * dStar
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
