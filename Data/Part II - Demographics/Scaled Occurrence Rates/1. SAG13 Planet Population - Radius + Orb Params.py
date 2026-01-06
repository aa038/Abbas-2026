import pandas as pd
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import rv_continuous

def classify_sag13_type(mass):
    if mass > 1.15:
        return 'F'
    elif 0.9 <= mass <= 1.15:
        return 'G'
    elif 0.6 <= mass < 0.9:
        return 'K'
    else:
        return 'M'
    
def occurrence_rate_for_Rp_a(Rp, a, coeffs, Mstar, smaknee = 30):
    
    # Pull out SAG13 coefficients
    Gamma = coeffs['Gamma']
    alpha = coeffs['alpha']
    beta  = coeffs['beta']
    Rplim = coeffs['Rplim']
    
    # Ensure the function works for both float and array inputs
    Rp = np.atleast_1d(Rp)
    a = np.atleast_1d(a)
    
    # Empty array to be filled with probability densities
    f = np.zeros(Rp.shape)
    
    # Divide the planets into two groups based on Rplim
    small_planet_mask = Rp < Rplim

    f[small_planet_mask] = 1.5 * Gamma[0] * Rp[small_planet_mask] ** (alpha[0] - 1) * a[small_planet_mask] ** ((3*beta[0] - 2)/2) * Mstar ** (-0.5*beta[0]) * np.exp(-(a[small_planet_mask] / smaknee)**3)
    
    # Find the giant planets
    giant_planet_mask = Rp >= Rplim
    
    f[giant_planet_mask] = 1.5 * Gamma[1] * Rp[giant_planet_mask] ** (alpha[1] - 1) * a[giant_planet_mask] ** ((3*beta[1] - 2)/2) * Mstar ** (-0.5*beta[1]) * np.exp(-(a[giant_planet_mask] / smaknee)**3)
    
    return f * 15

def make_Rp_a_grid(Rprange, arange, n_rp=200, n_a=200):
    
    # Generate a uniform grid of planet mass and SMA
    Rp_vals = np.linspace(Rprange[0], Rprange[1], n_rp)
    a_vals = np.linspace(arange[0], arange[1], n_a)
    
    # Convert it into a 2D meshgrid
    Rp_grid, a_grid = np.meshgrid(Rp_vals, a_vals, indexing='ij')
    
    return Rp_vals, a_vals, Rp_grid, a_grid

def get_occurrence_rate(Rp_grid, a_grid, coeffs, Mstar, Rprange, arange):
    
    # Compute the occurrence rate for each specific planet radius and SMA
    f_vals = occurrence_rate_for_Rp_a(Rp_grid, a_grid, coeffs, Mstar)
    
    # Compute the bin width in the planet radius and SMA bins
    dRp = (Rprange[1] - Rprange[0]) / (Rp_grid.shape[0] - 1)
    da  = (arange[1] - arange[0]) / (a_grid.shape[1] - 1)
    
    # Compute the occurrence rate across the planet radius and SMA range for the given star
    freq = np.trapz(np.trapz(f_vals, dx=da, axis=1), dx=dRp, axis=0)
    
    return f_vals, freq

def occurrence_rate_per_star(Mstar, coeffs, Rp_range, a_range, n_rp=200, n_a=200):
    
    # Set up the meshgrid over planet radius and SMA
    Rp_vals, a_vals, Rp_grid, a_grid = make_Rp_a_grid(Rp_range, a_range, n_rp, n_a)
    
    # Get the occurrence rate for the current star
    f_vals, freq = get_occurrence_rate(Rp_grid, a_grid, coeffs, Mstar, Rp_range, a_range)
    
    return f_vals, freq

def make_inverse_sampler_1d(x_vals, pdf_vals):
    
    # Compute bin widths
    dx = np.diff(x_vals)
    dx = np.append(dx, dx[-1])  # Assume last bin width = previous

    # Compute cumulative sum to get the CDF
    cdf = np.cumsum(pdf_vals * dx)
    cdf /= cdf[-1]  # Normalize to [0, 1]
    
    return interp1d(cdf, x_vals, bounds_error=False, fill_value=(x_vals[0], x_vals[-1]))

def sample_from_joint_pdf(p_grid, Rp_vals, a_vals, n_samples):
    
    # Step 1: marginalize over a to get f(Rp)
    f_rp = np.trapz(p_grid, x=a_vals, axis=1)
    Rp_sampler = make_inverse_sampler_1d(Rp_vals, f_rp)
    
    # Inverse sample the Rp CDF to get back an Rp
    sampled_Rp = Rp_sampler(np.random.uniform(size=n_samples))

    # Step 2: for each Rp, get p(a | Rp)
    sampled_a = np.zeros(n_samples)

    for i, Rp in enumerate(sampled_Rp):

        # Find the grid value in the radius axis closest to Rp
        Rp_idx = np.abs(Rp_vals - Rp).argmin()

        # Find the condition probability p(a | Rp)
        pa_given_Rp = p_grid[Rp_idx, :]

        # Normalise the conditional PDF
        pa_given_Rp = pa_given_Rp / np.trapz(pa_given_Rp, x = a_vals)

        # Construct the inverse CDF, now over p(a | Rp)
        a_sampler = make_inverse_sampler_1d(a_vals, pa_given_Rp)

        # Use the inverse CDF to get SMA
        sampled_a[i] = a_sampler(np.random.uniform())

    return sampled_Rp, sampled_a


# Directory Management
curr_dir = Path(__file__).resolve().parent
data_dir = curr_dir / "Data"

# Load the star dataframe
stars = pd.read_csv(data_dir/"HWO Stars.csv")

#---------------------------  PLANET PARAM RANGE  ----------------------------#

# Compute the most appropriate SAG13 spectral type based on mass
stars['Base_SpecType'] = stars['M'].apply(classify_sag13_type)

# Define the SAG13 power law coefficients
SAG13_coeffs = {
    'Gamma': [0.38, 0.73],    # Occurrence Rate in each radius regime: Gamma[0] for R < Rplim, Gamma[1] for R > Rplim
    'alpha': [-0.19, -1.18],  # Power law slope for radius
    'beta': [0.26, 0.59],     # Power law slope for SMA
    'Rplim': 3.4              # Planet radius breakpoint between the two regimes, in Earth radii 
}

# Define a reasonable SMA and planet radius range
Rp_range = [0.5, 3.4]  # Radius in Earth radii
a_range = [0.1, 10]   # SMA in AU
#------------------------------------------------------------------------------#


#----------------------------  PLANET GENERATION  -----------------------------#
# Empty list to store planet details
all_planets = []

for i, row in stars.iterrows():
    Mstar = row['M']
    coeffs = SAG13_coeffs

    # Step 1: Make Rp vs SMA grid and normalize PDF
    Rp_vals, a_vals, Rp_grid, a_grid = make_Rp_a_grid(Rp_range, a_range)
    f_vals, freq = occurrence_rate_per_star(Mstar, coeffs, Rp_range, a_range)
    p_grid = f_vals / freq

    # Step 2: Sample number of planets
    n_planets = np.random.poisson(freq)

    # Step 3: Sample Rp and a
    Rp, sma = sample_from_joint_pdf(p_grid, Rp_vals, a_vals, n_planets)

    # Step 4: Generate other orbital parameters
    mean_anomaly = np.random.uniform(0, 2*np.pi, size=n_planets)
    aop = np.random.uniform(0, 2*np.pi, size=n_planets)
    pan = np.random.uniform(0, 2*np.pi, size=n_planets)
    cosi = np.random.uniform(0, 1, size=n_planets)
    inc = np.arccos(cosi)
    ecc = np.random.rayleigh(scale=0.25, size=n_planets)
    ecc = np.clip(ecc, 0, 0.95)

    # Step 5: Compute the period
    P = np.sqrt(sma**3 / Mstar)

    # Step 6: Convert mean anomaly to epp
    epp = 2035 - mean_anomaly / (2 * np.pi) * P

    # Step 6: Package into list of dicts
    hip_id = row["HDName"]
    for j in range(n_planets):
        planet = {
            "PlanetID": f"{hip_id}_{j}",
            "HDName": hip_id,
            "Spec": row['SpecType'],
            "d_pc": row["Dist"],
            "L_sol": row["L"],
            "M_sol": row["M"],
            "Rp_Rearth": Rp[j],
            "sma_AU": sma[j],
            "P_yr": P[j],
            "ecc": ecc[j],
            "inc_rad": inc[j],
            "aop_rad": aop[j],
            "pan_rad": pan[j],
            "epp_yr": epp[j]
        }
        all_planets.append(planet)

# Convert to dataframe
planet_df = pd.DataFrame(all_planets)

# Format to 4 significant figures and save
planet_df = planet_df.applymap(lambda x: f"{x:.4g}" if isinstance(x, float) else x)
planet_df.to_csv(data_dir / "SAG13 Planet Catalog.csv", index=False)
#------------------------------------------------------------------------------#


