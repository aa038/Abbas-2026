import pandas as pd
import numpy as np
from scipy.optimize import minimize, dual_annealing
from pathlib import Path
import time

start_time = time.time()

from solve_orbit import solve_orbit

# Define habitable zone based on luminosity
def HZ(L):
    """
    Calculate inner and outer edges of the habitable zone.

    Parameters:
    L (float or np.array): Stellar luminosity in solar units.

    Returns:
    HZ_inner (float or np.array): Inner edge of the HZ (in AU).
    HZ_outer (float or np.array): Outer edge of the HZ (in AU).
    """
    HZ_inner = np.sqrt(L / 1.78)
    HZ_outer = np.sqrt(L / 0.32)
    return HZ_inner, HZ_outer




def RankStars(starsDF, weights = [-5,-10,5,10]):
    """
    Rank stars based on distance, brightness, and spectral type.
    
    Parameters:
    starsDF (pd.DataFrame): DataFrame containing star and planet properties.
    
    Returns:
    pd.DataFrame: Ranked DataFrame with a 'Score' column.
    """

    # Weights
    wDist = weights[0]   # Weight for the distance
    wLum = weights[1]    # Weight for luminosity
    wSpecType = weights[2]     # Weight for spectral type
    wExpTime = weights[3]     # Weight for exposure time (prefer shorter exposure times)

    # Spectral type factor mapping
    # We prefer F and G stars since the HZ will be larger, farther out and we have the potential for more SNR
    spectral_type_map = {'F': 3, 'G': 3, 'K': 2, 'M': 1} 

    # Add spectral type factor
    spectralTypeFactor = starsDF['Spec'].str[0].map(spectral_type_map).fillna(0)

    # Normalize the exposure times
    normExpTime = (starsDF['tExp(s)'] - starsDF['tExp(s)'].min()) / (starsDF['tExp(s)'].max() - starsDF['tExp(s)'].min())

    # Compute scores
    starsDF['Score'] = (
        wDist / starsDF['d_pc'] +  # Closer stars rank higher
        wLum * starsDF['L_sol'] +  # Brighter stars rank higher
        wSpecType * spectralTypeFactor +  # Adjust for spectral type
        wExpTime * normExpTime # Adjust for exposure time
    )

    # Sort stars by score
    ranked_stars = starsDF.sort_values(by='Score', ascending=False).reset_index(drop=True)
    return ranked_stars

def IntegrationTime(L, d, A, throughput = 0.27, contrast = 1e-11, SNR = 10, wavelength = 0.6e-6):

    # Convert distance from pc to metres
    d = d * PC_TO_M

    # Flux from the star
    starFlux = (L * SOLARLUM) / (4 * np.pi * d**2)

    # Convert to photon flux
    photon_energy = h * c / wavelength  # Energy of a photon (J)
    starPhotonFlux = starFlux / photon_energy  # Photon flux (photons/m^2/s)

    # Using the max contrast that can be achieved, we can get the minimum flux for the planet we can detect
    planetPhotonFlux = starPhotonFlux * contrast

    # Compute the exposure time, assuming the signal limited case
    tExp = (SNR ** 2) * (starPhotonFlux*contrast + planetPhotonFlux) / (planetPhotonFlux**2 * A * throughput)

    return tExp

def contrastCheck(DetParams, albedo, Rp, coronagraphContrast = 1e-11):

    # Compute the 3D distance 
    r3D = DetParams[2] * AU_TO_M

    # Contrast
    contrast = albedo * (Rp/r3D)**2 * DetParams[3]

    if contrast > coronagraphContrast:
        return True
    
    return False

def sepCheck(DetParams, IWA = 0.04, OWA = 1):

    # Check if the planet is within the working angle
    if IWA < DetParams[0] < OWA:
        return True
    
    return False

def SNRCheck(L, d, Rp, albedo, A, tExp, DetParams, wavelength = 0.6e-6, throughput = 0.27, contrast = 1e-11):

    # Flux from the star
    starFlux = (L * SOLARLUM) / (4 * np.pi * (d*PC_TO_M)**2)

    # Convert to photon flux
    photon_energy = h * c / wavelength  # Energy of a photon (J)
    starPhotonFlux = starFlux / photon_energy  # Photon flux (photons/m^2/s)

    # Compute planet flux (stellar flux * albedo * planet cross-sectional area * phase function)
    starFluxAtPlanet = (L * SOLARLUM) / (4 * (DetParams[2]*AU_TO_M)**2)
    planetLum = starFluxAtPlanet * albedo * Rp**2 * DetParams[3]
    planetFlux = planetLum / (4 * np.pi * (d*PC_TO_M)**2)
    planetPhotonFlux = planetFlux / photon_energy

    # Compute the SNR assuming the coronagraph masks out much of the star's light
    SNR = planetPhotonFlux / (np.sqrt(planetPhotonFlux + starPhotonFlux*contrast)) * np.sqrt(A * tExp * throughput)

    if SNR > 3:
        return True
    
    return False
        

# Directory Management
curr_dir = Path(__file__).resolve().parent
parent_dir = curr_dir.parent.parent
data_dir = parent_dir / "Data"

# Load the list of stars with planets around them
planets_df = pd.read_csv(data_dir / "SAG13 Planet Catalog.csv")
planets_df = planets_df[planets_df['P_yr'] < 10]


# Empty array to keep track of how many times the planet has been detected
planets_df["NDet"] = np.zeros(len(planets_df))

# One row per star, for obs timing and exposure time
stars_df = planets_df.drop_duplicates("HDName")[["HDName", "Spec", "d_pc", "L_sol", "M_sol"]].copy()

# Diameter of the primary mirror (in m)
telAperture = 4

# Compute the area of the telescope
A = np.pi * (telAperture/2) ** 2

SOLARLUM = 3.83e26
EARTHRADII = 6.371e6
AU_TO_M = 1.5e11
PC_TO_M = 3.086e16 

h = 6.626e-34  # Planck's constant (J·s)
c = 3e8  # Speed of light (m/s)


# ----------------------------------  MISSION SPECS  ----------------------------------------- #
# Compute the exposure time required for each star, and rank the stars by exposure time
stars_df['tExp(s)'] = IntegrationTime(stars_df['L_sol'], stars_df['d_pc'], A)
stars_df = RankStars(stars_df)

# Empty arrays to keep track of:
# - How many times a star has been observed
# - How many times a planet has been detected
# - When the stars was last observed
stars_df["NObs"] = np.zeros(len(stars_df))
stars_df["LastObs"] = np.ones(len(stars_df)) * np.inf

# Mission start and end times
mission_clock = 2035
mission_end = 2040

# Max number of times a star will be observed
maxObs = 8
# Min wait time between observations
timeStep = 3/12
# -------------------------------------------------------------------------------------------- #

# Empty dataframe to hold the observing log
ObsLog = pd.DataFrame(columns = ['PlanetID', 'StarName', 'SMA_AU', 'ecc', 'Rp_REarth', 'Mp_MEarth', 'd_pc', 'M_sol', 'L_sol', 'Sep', 'PA', '3dDist', 'PhaseFunc', 'DetStatus', 'NDet', 'NObs', 'LastObs'])

obs_counter = 0
max_possible_obs = len(stars_df) * maxObs

while mission_clock < mission_end:

    # Creating our target list. We want stars that:
    # 1. Have not been observed more than NObs times
    # 2. Have been observed more than one timeStep (the minimum wait time between obs) ago
    eligible_stars = stars_df[(stars_df['NObs'] < maxObs) & (np.abs(stars_df['LastObs'] - mission_clock) >= timeStep)]

    # If the target list is empty, that could be due to two reasons
    # 1. All the stars have been observed the required number of times. In that case, we're done!
    # 2. It has not been long enough since we last observed all the stars. In that case, move time forward by a day and try again
    if eligible_stars.empty:
        if (stars_df['NObs'] == maxObs).all():
            break

        mission_clock += 1/365
        continue

    # Pick the top-ranked star
    star = eligible_stars.sort_values("Score", ascending=False).iloc[0]
    star_name = star["HDName"]
    star_mass = star["M_sol"]
    star_lum = star["L_sol"]
    star_planets = planets_df[planets_df['HDName'] == star_name]

    # Update star observation timing
    stars_df.loc[stars_df["HDName"] == star_name, 'LastObs'] = mission_clock
    stars_df.loc[stars_df["HDName"] == star_name, 'NObs'] += 1

    obs_counter += 1

    if obs_counter % len(stars_df) == 0:
        elapsed = time.time() - start_time
        avg_time = elapsed / obs_counter
        remaining = (max_possible_obs - obs_counter) * avg_time
        print(f"[{mission_clock:.2f}] {obs_counter} obs done | Est. {remaining:.1f} sec remaining")

    # Loop through all the planets around this star
    for _, planet in star_planets.iterrows():

        # Empty list to keep track of the observing log for one single observation
        # This will be appended to ObsLog at the end of the loop iteration
        Obs_row = []

        # ObsLog Cols 1. & 2. Planet ID and Star ID
        Obs_row.append(planet['PlanetID']) 
        Obs_row.append(star_name)

        # Planet orbital and physical parameters
        sma = planet['sma_AU']
        ecc = planet['ecc']
        inc = np.degrees(planet['inc_rad'])
        aop = np.degrees(planet['aop_rad'])
        pan = np.degrees(planet['pan_rad'])
        epp = planet['epp_yr']
        P = planet['P_yr']
        d = planet['d_pc']
        albedo = planet['albedo']
        Rp = planet['Rp_Rearth'] * EARTHRADII
        Mp = planet['Mp_Mearth']

        # ObsLog Cols 3-9. SMA (in AU), Rp (in R_E), M_p (in M_E)
        Obs_row.extend([sma, ecc, planet['Rp_Rearth'], Mp, d, star_mass, star_lum])

        # Compute sep, PA, 3D dist and phase function
        DetParams = solve_orbit(sma, ecc, inc, aop, pan, epp, P, mission_clock, d)

        # ObsLog Cols 10-13. 2D Sep, PA, 3D Dist, Phase Func
        Obs_row.extend(DetParams)

        # Run detection checks
        check2 = sepCheck(DetParams)
        check3 = contrastCheck(DetParams, albedo, Rp)

        # Check if the planet is detected
        if check2 and check3:
            planets_df.loc[planets_df['PlanetID'] == planet['PlanetID'], 'NDet'] += 1
            Obs_row.append(1)  # Detected, ObsLog Col. 14
        else:
            Obs_row.append(0)  # Not detected, ObsLog Col. 14

        # ObsLog Cols. 15-17: NDet, NObs (star), LastObs (star)
        Obs_row.append(int(planets_df.loc[planets_df['PlanetID'] == planet['PlanetID'], 'NDet'].iloc[0]))
        Obs_row.append(int(stars_df.loc[stars_df["HDName"] == star_name, 'NObs'].iloc[0]))
        Obs_row.append(mission_clock)

        ObsLog.loc[len(ObsLog)] = Obs_row

    # Advance clock by exposure time
    mission_clock += star['tExp(s)'] / (3600 * 24 * 365)


#print(HWOStarsWPlanets['NObs'].head(25))
ObsLog = ObsLog.round(3)
ObsLog.to_csv(curr_dir / "7a. Observing Log.csv", index=False)


    


