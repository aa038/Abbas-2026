"""
SAG13 Planet Generator (Part 1 of 2)
------------------------------------
Generates a catalog of simulated planets around the HWO star list by sampling a
SAG13-style occurrence model over (Rp, P). Users can recreate the paper's results
and tune the planet population mainly by changing the limits below:

    Rp_range = [0.5, 3.4]   # Earth radii
    P_range  = [0.03, 10]   # years

Output:
    Data/Planet Generation/SAG13 Planet Catalog.csv

Notes:
- Occurrence model is a broken power law in Rp with separate slopes/normalizations.
- Eccentricities drawn from Rayleigh(scale=0.061) and clipped to [0, 0.95].
- Semi-major axis computed from sampled period via Kepler's 3rd law (AU, Msun).
"""

# ------------------------ Scientific Assumptions --------------------------- #
# - Occurrence model follows SAG13 small-planet regime (Rp < 3.4 R_earth)
# - Both period and planet radius drawn from SAG13 power laws
# - Mean number of planets per star = integrated SAG13 rate over Rp-P range
# - Sin i prior on inclination
# - Eccentricities drawn from Rayleigh(sigma=0.061), clipped at e=0.95 (see Abbas et. al. 2026 Fig 12)
# - No planet–planet correlations are modeled
# --------------------------------------------------------------------------- #

import pandas as pd
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d

# ----------------------------- Reproducibility ------------------------------ #
SEED = 42  # Set to None for stochastic runs; set to an int for reproducibility
rng = np.random.default_rng(SEED)
# --------------------------------------------------------------------------- #
    
# -------------------------- SAG13 Occurrence Model -------------------------- #
def make_Rp_P_grid(Rp_range, P_range, n_rp=200, n_P=200):
    """
    1. Build logarithmically spaced grids in Rp and P
    2. And a 2D meshgrid in Rp x P space

    """
    # Logarithmically spaced grid in planet radius and period
    Rp_vals = np.logspace(np.log10(Rp_range[0]), np.log10(Rp_range[1]), n_rp)
    P_vals  = np.logspace(np.log10(P_range[0]), np.log10(P_range[1]), n_P)

    # Create a 2D meshgrid of planet radius and period
    # Both Rp_grid and P_grid have shapes (n_rp x n_P)
    # Each row in Rp_grid corresponds to a given Rp
    # Each column in P_grid corresponds to a given P
    Rp_grid, P_grid = np.meshgrid(Rp_vals, P_vals, indexing='ij')

    return Rp_vals, P_vals, Rp_grid, P_grid

def occurrence_rate_for_Rp_P(Rp, P, coeffs):
    """
    Differential occurrence density f(Rp, P) = d^2N / (dRp dP)
    evaluated at (Rp, P) points (scalars, arrays, or meshgrids).

    SAG13-style model:
        d^2N / (dlnRp dlnP) = Gamma_k * Rp^alpha_k * P^beta_k
    which implies in linear variables:
        f(Rp,P) = d^2N / (dRp dP) = Gamma_k * Rp^(alpha_k - 1) * P^(beta_k - 1),
    with k=0 for Rp <= Rplim and k=1 for Rp > Rplim.
    """
    # SAG13 parameters for both the small and large radius regimes (Rplim = 3.4 R_E)
    Gamma = coeffs['Gamma']
    alpha = coeffs['alpha']
    beta  = coeffs['beta']
    Rplim = coeffs['Rplim']

    # Both Rp and P are 2D meshgrids of shape (n_rp x n_P; defined in the previous function)
    Rp = np.atleast_1d(Rp)
    P  = np.atleast_1d(P)

    # f will be computed for each point in Rp x P space. Generate an array with the same shape as either meshgrid
    f  = np.zeros_like(Rp, dtype=float)

    # Compute the SAG13 OR for both the small and large radius regimes evaluated in linear Rp and P space
    small = (Rp <= Rplim)
    if np.any(small):
        f[small] = Gamma[0] * Rp[small]**(alpha[0] - 1) * P[small]**(beta[0] - 1)

    giant = ~small
    if np.any(giant):
        f[giant] = Gamma[1] * Rp[giant]**(alpha[1] - 1) * P[giant]**(beta[1] - 1)

    return f


def get_occurrence_rate(Rp_grid, P_grid, coeffs, Rp_vals, P_vals):
    """
    Compute f(Rp,P) on the grid and integrate to get total frequency over the box.
    
    """
    # Compute the differential occurrence rate 
    f_vals = occurrence_rate_for_Rp_P(Rp_grid, P_grid, coeffs)
    # Integrate over P (axis=1), then over Rp (axis=0)
    freq = np.trapz(np.trapz(f_vals, x=P_vals, axis=1), x=Rp_vals, axis=0)
    print(freq)

    return f_vals, freq
# --------------------------------------------------------------------------- #

# ------------------------------- Samplers ---------------------------------- #
def inverse_sampler_1d(x_vals, pdf_vals):
    """
    An inverse-CDF sampler for a 1D PDF
    
    """
    # CDF = Sum over bins of (PDF * dx) normalized to 0-1
    # Compute bin width dx for each bin
    dx = np.diff(x_vals)
    dx = np.append(dx, dx[-1])  # Assume last bin width equals previous

    # Compute the CDF
    cdf = np.cumsum(pdf_vals * dx)
    cdf /= cdf[-1]

    # Returns a function that computes the inverse CDF 
    # CDF(x) = u, u = [0, 1];   CDF^{-1}(u) = x
    # For a CDF in Rp, the inverse CDF gives Rp for a random deviate u, allowing us to sample Rp
    return interp1d(cdf, x_vals, bounds_error=False, fill_value=(x_vals[0], x_vals[-1]))


def sample_from_joint_pdf(pdf_grid, Rp_vals, P_vals, n_planets):
    """
    Sample (Rp, P) pairs for each planet using the joint Rp-P PDF

    """
    # ----- II. Sampling Rp ----- #
    # Marginalise the PDF over period to get p(Rp)
    p_Rp = np.trapz(pdf_grid, x=P_vals, axis=1)

    # Build an inverse CDF sampler using this marginalised PDF
    Rp_sampler = inverse_sampler_1d(Rp_vals, p_Rp)

    # Use the inverse CDF sampler to draw radii for all the planets
    sampled_Rp = Rp_sampler(rng.random(size=n_planets))

    # ----- III. Sampling P for a given Rp ----- #
    # Empty array to store the period (length = n_planets)
    sampled_P = np.empty(n_planets)

    # For each Rp, find p(P|Rp), compute the CDF, invert it, and draw P
    for i, Rp in enumerate(sampled_Rp):

        # Find the closest p(P|Rp) from our discretized PDF grid
        idx = np.abs(Rp_vals - Rp).argmin()
        pP_given_Rp = pdf_grid[idx, :]

        # pP_given_Rp is a slice of the joint PDF at a given Rp.
        # Normalize it so it integrates to 1
        pP_given_Rp = pP_given_Rp / np.trapz(pP_given_Rp, x=P_vals)

        # Build the inverse CDF sampler for p(P|Rp)
        P_sampler = inverse_sampler_1d(P_vals, pP_given_Rp)

        # Draw periods for each planet (ensured by the loop)
        sampled_P[i] = P_sampler(rng.random())

    return sampled_Rp, sampled_P
# --------------------------------------------------------------------------- #


# ----------------------------- I/O & Parameters ---------------------------- #
curr_dir  = Path(__file__).resolve().parent
stars = pd.read_csv(curr_dir / "HWO Stars.csv")

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> USER-TUNABLE LIMITS <<<<<<<<<<<<<<<<<<<<<<<<<< #
Rp_range = [0.5, 3.4]   # Planet radius in Earth radii 
P_range  = [0.03, 10.0] # Orbital period in years
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #

# SAG13 broken-power-law coefficients 
# The SAG13 OR is defined as a broken power law in planet radius and orbital period
# with a break at Rp = 3.4 R_E, with separate power law indices on either side of the break
# d^2N / dlogR dlogP = Gamma x R**alpha x P**beta (power law break at R = Rplim = 3.4 R_E)
SAG13_coeffs = {
    'Gamma': [0.38, 0.73],    # Normalization for Rp <= 3.4 R_E, and Rp > 3.4 R_E
    'alpha': [-0.19, -1.18],  # Power law slope in radius (either side of the break)
    'beta' : [0.26, 0.59],    # Power law slope in orbital period (either side of the break)
    'Rplim': 3.4              # Radius breakpoint (R_E)
}
# --------------------------------------------------------------------------- #


# ------------------------------ Planet Draws ------------------------------- #
all_planets = []

# The SAG13 OR is used for 3 things:
#   I. The occurrence rate (obviously) - Draw the number of planets around each star
# For each planet,
#   II. Planet Radius - Marginalize the SAG13 joint PDF over period to get p(Rp) and draw Rp
#   III. Orbital Period - Draw P from the conditional PDF p(P|Rp)

# ----- I. Computing the occurrence rate per star ----- #

# Build logarithmically spaced grids in radius and period space, and meshgrids in radius x period space
Rp_vals, P_vals, Rp_grid, P_grid = make_Rp_P_grid(Rp_range, P_range)

# Compute the differential occurrence rate: f_vals (f(Rp, P) = d^2N / dRp dP)
# and the total OR across the Rp-P range: freq
f_vals, freq = get_occurrence_rate(Rp_grid, P_grid, SAG13_coeffs, Rp_vals, P_vals)

# ----- II, III. Drawing planet radius and period ----- #

# Joint PDF p(Rp, P) discretized over the Rp x P grid defined earlier
# This can be marginalized over period (p(Rp)) to draw Rp
# Once Rp is drawn, period can be drawn from the conditional distribution p(P|Rp)
pdf_grid = f_vals / freq  

for _, row in stars.iterrows():
    
    # Mass of the star in Solar Masses
    Mstar = row['M']  

    # Using the total OR per star over the Rp-P range (freq) computed earlier, 
    # draw actual number of planets around each star using freq as the Poisson mean
    n_planets = rng.poisson(freq)

    # If there are no planets around this star, skip generating planet properties
    if n_planets == 0:
        continue

    # Using the joint PDF computed earlier, draw both planet radius Rp and period P for each planet
    Rp, P = sample_from_joint_pdf(pdf_grid, Rp_vals, P_vals, n_planets)

    # Draw the other orbital params for the planet 
    mean_anomaly = rng.uniform(0, 2*np.pi, size=n_planets)             # Mean anomaly drawn from a uniform prior in (0-2pi)
    aop = rng.uniform(0, 2*np.pi, size=n_planets)                      # Angle of periastron
    pan = rng.uniform(0, 2*np.pi, size=n_planets)                      # Position angle of nodes
    inc = np.arccos(rng.uniform(0, 1, size=n_planets))                 # Inclination - Drawn from a sin i distribution
    ecc = np.clip(rng.rayleigh(scale=0.061, size=n_planets), 0, 0.95)  # Eccentricity - Rayleigh Distribution(scale=0.061) and clipped to [0, 0.95] (see Abbas et. al. 2026 Fig. 12)

    # Kepler's Third Law to compute SMA
    sma = (P**2 * Mstar) ** (1/3)

    # Epoch of periastron passage from randomly drawn mean anomaly (Arbitrary reference year: 2035)
    epp = 2035.0 - mean_anomaly / (2 * np.pi) * P

    # Package records 
    # **** Optional user input: Any additional columns can be added on here to be saved in the data product **** #
    hd_name = row["HDName"]
    for j in range(n_planets):
        all_planets.append({
            "PlanetID":  f"{hd_name}_{j}",
            "HDName":    hd_name,
            "Spec":      row['SpecType'],
            "d_pc":      row["Dist"],
            "L_sol":     row["L"],
            "M_sol":     row["M"],
            "Rp_Rearth": Rp[j],
            "sma_AU":    sma[j],
            "P_yr":      P[j],
            "ecc":       ecc[j],
            "inc_rad":   inc[j],
            "aop_rad":   aop[j],
            "pan_rad":   pan[j],
            "epp_yr":    epp[j],
        })

# Save as a .csv
planet_df = pd.DataFrame(all_planets)
out_path = curr_dir / "SAG13 Planet Catalog.csv"
planet_df.to_csv(out_path, index=False, float_format="%.3f")
# --------------------------------------------------------------------------- #

