import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from orbituary_interface import fit_orbit
from plotting import plot_progressive_fits


def main():
    """Fit each observing epoch and plot how the posterior develops."""

    np.random.seed(7)

    observations = pd.read_csv("3. Observing Log.csv")
    catalog = pd.read_csv("2. Planet Catalog w Mass.csv")

    planet_id = observations["PlanetID"].iloc[0]
    observations = observations[observations["PlanetID"] == planet_id].sort_values("LastObs")
    system = catalog[catalog["PlanetID"] == planet_id].iloc[0]

    times = observations["LastObs"].to_numpy()
    separations = observations["Sep"].to_numpy()
    position_angles = observations["PA"].to_numpy()

    stellar_mass = system["M_sol"]
    distance = system["d_pc"]

    print(
        f"Fitting {planet_id}: true SMA={system['sma_AU']:.3f} AU, "
        f"true eccentricity={system['ecc']:.3f}"
    )

    orbit_fits = []
    prev_orbits = None

    for epoch in range(1, len(times) + 1):

        # For epoch 3, instead of using the smaller OFTI sample, use a larger new OFTI run
        if epoch <= 3:
            mcmc_initial_orbits = None
        else:
            mcmc_initial_orbits = prev_orbits

        orbits, diagnostics = fit_orbit(
            times[:epoch],
            separations[:epoch],
            position_angles[:epoch],
            stellar_mass,
            distance,
            sep_uncertainty=0.0005,
            pa_uncertainty=0.1,
            initial_orbits=mcmc_initial_orbits,
            sma_bounds=(0.1, 10.0),
            num_orbits=200,
            max_ofti_time=120,
            ofti_batch_size=10_000,
            mcmc_walkers=24,
            mcmc_steps=15_000,
            mcmc_burnin=3000
        )

        prev_orbits = orbits

        orbit_fits.append(orbits)
        print(
            f"Epoch {epoch}: {diagnostics['method']}, "
            f"median SMA={orbits['sma'].median():.3f} AU, "
            f"median eccentricity={orbits['ecc'].median():.3f}, "
            f"chi-square median/max="
            f"{np.median(diagnostics['chi_squared']):.1f}/"
            f"{np.max(diagnostics['chi_squared']):.1f}"
        )

    figure = plot_progressive_fits(
        orbit_fits,
        times,
        separations,
        position_angles,
        distance,
        n_orbits=100,
        save_path="progressive_orbit_fit.png",
    )

    plt.close(figure)


if __name__ == "__main__":
    main()