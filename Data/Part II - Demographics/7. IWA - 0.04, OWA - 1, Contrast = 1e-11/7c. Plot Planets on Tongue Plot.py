import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.stats import beta
import pandas as pd

from PlotStyle import plotStyle
plotStyle()

# Directory Management
curr_dir = Path(__file__).resolve().parent
parent_dir = curr_dir.parent.parent
data_dir = parent_dir / "Data"

# Load the saved file
data = np.load(curr_dir / '7b. 4D Tongue Plot.npz', allow_pickle=True)

completeness_4d = data['completeness']      # Shape: (n_mass, n_sma, n_ecc, n_stars)
rad_centers = data['rad_centers']
per_centers = data['per_centers']
ecc_centers = data['ecc_centers']

rad_edges = data['rad_edges']
per_edges = data['per_edges']
ecc_edges = data['ecc_edges']

# Marginalising over eccentricities
beta_weights = beta.pdf(ecc_centers, a=0.867, b=3.03)
beta_weights /= np.sum(beta_weights)

completeness_3d  = np.average(completeness_4d, axis=2, weights=beta_weights)  # over ecc
completeness_2d = np.sum(completeness_3d , axis=2)  # over stars → now shape (n_mass, n_sma)

fig, ax = plt.subplots(figsize=(10, 8))

colors = ["white", "yellow", "red", "blue"]
ylrdblu_cmap = LinearSegmentedColormap.from_list("YlRdBlu", colors, N=1024)

# Plot the image
im = ax.imshow(completeness_2d,
               origin='lower',
               aspect='auto',
               extent=[np.log10(per_edges[0]), np.log10(per_edges[-1]),
                       np.log10(rad_edges[0]), np.log10(rad_edges[-1])],
               cmap=ylrdblu_cmap)

# Optional: Smooth and contour
smoothed = gaussian_filter(completeness_2d, sigma=1.0)
levels = [1, 5, 10, 20, 40, 60, 80, 90, 95]
CS = ax.contour(np.log10(per_centers),
                np.log10(rad_centers),
                smoothed,
                levels=levels,
                colors='#cacfd2')

fmt = {1: '1 star'}
fmt.update({lvl: f'{lvl} stars' for lvl in levels[1:]})

ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=12)
#ax.clabel(CS, CS.levels, inline=True, fmt={lvl: f'{lvl:.0f} stars' for lvl in levels[1:]}, fontsize=12)

# Labels
cbar = plt.colorbar(im)
cbar.set_label('Number of Stars', rotation=270, labelpad=15)
ax.set_xlabel('Period (yr)')
ax.set_ylabel('Planet Radius (R$_\oplus$)')

# Log ticks
per_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40]
rad_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3]
ax.set_xticks(np.log10(per_ticks))
ax.set_xticklabels(per_ticks)
ax.set_yticks(np.log10(rad_ticks))
ax.set_yticklabels(rad_ticks)

ax.set_ylim([np.log10(0.01), np.log10(3.4)])

# Load planet simulation results
obs_df = pd.read_csv(curr_dir / '7a. Observing Log.csv') 

# Find the 20 closest stars
#closest_20_star_names = obs_df[['StarName', 'd_pc']].drop_duplicates().sort_values('d_pc', ascending = False).head(20)['StarName'].values
# Filter the observations to only include those stars
#obs_df_20 = obs_df[obs_df['StarName'].isin(closest_20_star_names)]

# Group by PlanetID and aggregate over all epochs
grouped = obs_df.groupby('PlanetID').agg({
    'Rp_REarth': 'first',
    'SMA_AU': 'first',
    'M_sol': 'first',
    'NDet': 'sum'  # Total number of detections over all epochs
}).reset_index()

grouped['P'] = np.sqrt(grouped['SMA_AU']**3 / grouped['M_sol'])

# Classify detections
detected = grouped[grouped['NDet'] >= 1]
undetected = grouped[grouped['NDet'] == 0]

# Plot detected planets (green)
ax.scatter(np.log10(detected['P']),
           np.log10(detected['Rp_REarth']),
           color='green', s=2, label='Detected', alpha=1, zorder = 3)

# Plot undetected planets (red)
ax.scatter(np.log10(undetected['P']),
           np.log10(undetected['Rp_REarth']),
           color='red', s=1, label='Undetected', alpha=0.5, zorder = 3)

ax.legend()

plt.savefig(curr_dir / '7c. Tongue Plot with Planets.png', dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()