import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from PlotStyle import plotStyle

plotStyle()

# Directory Management
curr_dir = Path(__file__).resolve().parent

planets_df = pd.read_csv(curr_dir / "1d. Detected Planets.csv")

plt.scatter(planets_df['Rp_REarth'], planets_df['Mp_MEarth'])
plt.xlabel("Radius ($R_{\\oplus}$)")
plt.ylabel("Mass ($M_{\\oplus}$)")
plt.xlim([0.1, 7])
plt.ylim([0.01, 50])
plt.xscale('log')
plt.yscale('log')
plt.savefig("1j. Mass vs Radius.png", dpi = 300, bbox_inches = 'tight')