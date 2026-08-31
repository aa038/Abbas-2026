"""
Export one final Orbituary period/eccentricity posterior per detected planet.

Inputs
------
5c. Orbit Fits.pkl
5d. Detected Planets.csv

Output
-------
5d. Planet Posterior Samples.csv

The demographics likelihood must count every PlanetID as one event. The many
rows for a PlanetID are posterior samples for that event, not extra planets.
"""

from pathlib import Path

import numpy as np
import pandas as pd


CURR_DIR = Path(__file__).resolve().parent
DETECTED_FILE = CURR_DIR / "5d. Detected Planets.csv"
FITS_FILE = CURR_DIR / "5c. Orbit Fits.pkl"
OUTPUT_FILE = CURR_DIR / "5d. Planet Posterior Samples.csv"


def is_valid_posterior(value):
    """Return True for a nonempty Orbituary posterior containing P and e."""
    return (
        isinstance(value, pd.DataFrame)
        and not value.empty
        and {"period", "ecc"}.issubset(value.columns)
    )


def main():
    detected = pd.read_csv(DETECTED_FILE)
    fits = pd.read_pickle(FITS_FILE)

    if isinstance(fits.index, pd.MultiIndex):
        fits = fits.reset_index()

    required_fit_columns = {"PlanetID", "EpochNum", "orbit_df"}
    missing = required_fit_columns.difference(fits.columns)
    if missing:
        raise ValueError(f"{FITS_FILE.name} is missing columns: {sorted(missing)}")

    required_planet_columns = {
        "PlanetID", "StarName", "Rp_REarth", "M_sol", "NDet"
    }
    missing = required_planet_columns.difference(detected.columns)
    if missing:
        raise ValueError(
            f"{DETECTED_FILE.name} is missing columns: {sorted(missing)}"
        )

    detected_lookup = detected.set_index("PlanetID")
    posterior_tables = []
    missing_planets = []

    for planet_id in detected["PlanetID"]:
        planet_fits = fits[fits["PlanetID"] == planet_id]
        valid = planet_fits[planet_fits["orbit_df"].map(is_valid_posterior)]

        if valid.empty:
            missing_planets.append(planet_id)
            continue

        # Use exactly one posterior for each catalog event: the last successful
        # orbit fit, including a carried-forward final fit when appropriate.
        final_row = valid.sort_values("EpochNum").iloc[-1]
        samples = final_row["orbit_df"].loc[:, ["period", "ecc"]].copy()
        samples = samples.replace([np.inf, -np.inf], np.nan).dropna()
        samples = samples[
            (samples["period"] > 0)
            & (samples["ecc"] > 0)
            & (samples["ecc"] < 1)
        ]

        if samples.empty:
            missing_planets.append(planet_id)
            continue

        planet = detected_lookup.loc[planet_id]
        samples.insert(0, "PlanetID", planet_id)
        samples["StarName"] = planet["StarName"]
        samples["Rp_REarth"] = float(planet["Rp_REarth"])
        samples["M_sol"] = float(planet["M_sol"])
        samples["FinalPosteriorEpoch"] = int(final_row["EpochNum"])
        samples["NDetectedEpochs"] = int(planet["NDet"])
        samples["InterimPrior"] = "Orbituary"
        posterior_tables.append(samples)

    if not posterior_tables:
        raise RuntimeError("No valid final Orbituary posteriors were found.")

    posterior_samples = pd.concat(posterior_tables, ignore_index=True)
    posterior_samples.to_csv(OUTPUT_FILE, index=False)

    n_exported = posterior_samples["PlanetID"].nunique()
    print(
        f"Saved {len(posterior_samples):,} posterior samples for "
        f"{n_exported}/{len(detected)} detected planets to {OUTPUT_FILE.name}."
    )
    if missing_planets:
        print(
            "No valid final posterior was available for "
            f"{len(missing_planets)} planet(s): {', '.join(missing_planets)}"
        )


if __name__ == "__main__":
    main()
