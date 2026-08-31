"""
OR Fitting (Part 5 of 9)
-------------------------------------------
This script uses the tongue plot and and an assumed OR model, to fit the list of detected planets.
The result is the full list of MCMC posteriors for all the OR parameters

Input:
    1b. 4D Tongue Plot.npz              # The 4D tongue plot stored a 4D NumPy array (From Part b)
    1d. Planet Posterior Samples.csv    # Final Orbituary P/e posterior samples (From Part d)


Output:
    1e. Fit, N = 1e4.csv                # The full list of MCMC posteriors for each parameter in the OR model
"""

using CSV
using NPZ
using Glob
using Optim
using Turing
using DataFrames
using Statistics
using MCMCChains
using Distributed
using Distributions
using SpecialFunctions

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> USER-TUNABLE LIMITS <<<<<<<<<<<<<<<<<<<<<<<<<< #
# Binning the tongue plot in stellar mass to speed up computation
STELLAR_MASS_LOWER   = 0.3             # Lower limit in solar masses
STELLAR_MASS_UPPER   = 1.5             # Upper limit in solar masses
N_BINS               = 10

# Multiprocessing
N_CORES              = 11              # Number of cores this script is run on. This is SOLELY dependent on how many cores you have on your machine
                                       # If you are unsure, set it equal to 1
                                       # DO NOT FUCK AROUND WITH THIS NUMBER. YOUR COMPUTER WILL FREEZE, AT BEST 

# Analysis boundaries (the regions over which the power‐law normalization and fitting are done)
# The first set of 4 variables is the rad x per x ecc region over which we normalize the power law
# This is region over which the OR applies

# The second set of 4 is the region over which the fitting is done
# These are the ranges used in the likelihood calculations
direct_imaging_analysis_regions = Dict(
    :power_law_rad_min  => 0.5,
    :power_law_rad_max  => 3.4,
    :power_law_per_min  => 0.03,
    :power_law_per_max  => 10.0,
    :power_law_ecc_min  => 0.0001,
    :power_law_ecc_max  => 0.99,

    :analysis_rad_min   => 0.5,
    :analysis_rad_max   => 3.4,
    :analysis_per_min   => 0.03,
    :analysis_per_max   => 10.0,
    :analysis_ecc_min   => 0.0001,
    :analysis_ecc_max   => 0.99
)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #

addprocs(N_CORES)

@everywhere using Turing, MCMCChains, DataFrames, CSV, Distributions
@everywhere using StatsBase: quantile
@everywhere using ForwardDiff

# Define type aliases for convenience (using the default Float64)
const VI   = Vector{Int}
const VF   = Vector{Float64}
const VStr = Vector{String}
const AF3  = Array{Float64,3}
const AF4  = Array{Float64,4}

# Define directory paths (assumes the data are in a "Data" folder next to this file)
const curr_dir     = @__DIR__
const parent_dir   = dirname(dirname(curr_dir))
const data_dir     = joinpath(parent_dir, "Planet Generation")


function load_tongue_plot()::Dict{Symbol,Any}
"""
Function to load the tongue plot data and stellar details
"""
    # Path to the tongue plot and stellar file
    tplot_file   = joinpath(curr_dir, "1b. 4D Tongue Plot.npz")
    stellar_file = joinpath(data_dir, "HWO Stars.csv")

    # Load up the star details 
    df             = CSV.read(stellar_file, DataFrame)
    stellar_masses = Float64.(df.M)
    stellar_names  = String.(df.HDName)

    # Read in the tongue plot 
    # This will be a 4D array of shape 60 x 100 x 20 x 98
    # The axes are planet_rad x per x ecc x star
    tplot_data = npzread(tplot_file)
    tplot      = tplot_data["completeness"]

    # Bin edges for the tongue plot
    rad_centers = tplot_data["rad_centers"]
    per_centers = tplot_data["per_centers"]
    ecc_centers = tplot_data["ecc_centers"]

    # Bin centers for the tongue plot
    rad_edges = tplot_data["rad_edges"]
    per_edges = tplot_data["per_edges"]
    ecc_edges = tplot_data["ecc_edges"]
 
    # The tongue plot, and the star details are returned as a dictionary
    return Dict(
        :tplot           => tplot,
        :stellar_masses  => stellar_masses,
        :stellar_names   => stellar_names,
        :rad_centers     => rad_centers,
        :per_centers     => per_centers,
        :ecc_centers     => ecc_centers,
        :rad_edges       => rad_edges,
        :per_edges       => per_edges,
        :ecc_edges       => ecc_edges
    )
end


function create_stellar_mass_bins(n_stellar_bins = N_BINS)
"""
Function to create stellar mass bins to bin the tongue plot
"""
    # Define the edges of the stellar mass bins
    stellar_mass_edges = range(STELLAR_MASS_LOWER, STELLAR_MASS_UPPER, length = n_stellar_bins+1)

    # Compute the bin centres
    stellar_mass_centres = (stellar_mass_edges[1:end-1] .+ stellar_mass_edges[2:end]) ./ 2

    return Dict(
        :edges => stellar_mass_edges,
        :centres => stellar_mass_centres
    )
end

function bin_tplot(tplot, stellar_masses, stellar_mass_edges)
"""
Function to bin the tongue plot in stellar masses using the bins created above
"""
    # Find the number of bins
    # We subtract 1 because 30 bins have 31 edges
    n_bins = length(stellar_mass_edges) - 1

    # Get the size of the first three dimensions from the original DI tplot
    n_rad, n_per, n_ecc, _ = size(tplot)

    # Create a new 4D array using the binned stellar masses
    binned_tplot = zeros(n_rad, n_per, n_ecc, n_bins)

    # Loop over each stellar mass bin
    # The new tongue plot is created by summing over the tongue plots of all stars in a mass bin
    for i in 1:n_bins

        # Find the lower and upper edges of the current mass bin
        lower = stellar_mass_edges[i]
        upper = stellar_mass_edges[i+1]

        # Find the indices of all the stars that fall within the current mass bin
        star_indices = findall(s -> (s >= lower && s < upper), stellar_masses)

        if !isempty(star_indices)
            # If there are stars in the bin, sum over the contributions
            binned_tplot[:,:,:,i] = sum(tplot[:,:,:,star_indices], dims = 4)
        else
            # If no stars fall in this bin, leave it as zeros
            binned_tplot[:,:,:,i] .= 0.0
        end
    end

    return binned_tplot

end

function crop_di_region(tplot_data, analysis_regions::Dict{Symbol,Float64}, stellar_mass_edges)
"""
Function to crop the tongue plot to match the analysis region defined below
This is the radius x period x ecc range over which the likelihood calculation is done
"""
    # Extract the tongue plot
    tplot = tplot_data[:tplot]
    
    # Compute the lower and upper bin edges for rad, per, ecc and stellar mass
    rad_lower          = tplot_data[:rad_edges][1:end-1]
    rad_upper          = tplot_data[:rad_edges][2:end]

    per_lower          = tplot_data[:per_edges][1:end-1]
    per_upper          = tplot_data[:per_edges][2:end]

    ecc_lower          = tplot_data[:ecc_edges][1:end-1]
    ecc_upper          = tplot_data[:ecc_edges][2:end]

    stellar_mass_lower = stellar_mass_edges[1:end-1]
    stellar_mass_upper = stellar_mass_edges[2:end]

    # Indices of bins that intersect the analysis region
    per_inds = findall(i -> per_upper[i] > analysis_regions[:analysis_per_min] && per_lower[i] < analysis_regions[:analysis_per_max], 1:length(per_lower))
    rad_inds = findall(i -> rad_upper[i] > analysis_regions[:analysis_rad_min] && rad_lower[i] < analysis_regions[:analysis_rad_max], 1:length(rad_lower))
    ecc_inds = findall(i -> ecc_upper[i] > analysis_regions[:analysis_ecc_min] && ecc_lower[i] < analysis_regions[:analysis_ecc_max], 1:length(ecc_lower))

    # Crop the 4D tplot (radius x per x ecc x stellar mass)
    cropped_tplot = tplot[rad_inds, per_inds, ecc_inds, :]

    # Cropped bin edges for use in modeling
    cropped_bin_edges = Dict(
        :per             => (per_lower[per_inds], per_upper[per_inds]),
        :rad             => (rad_lower[rad_inds], rad_upper[rad_inds]),
        :ecc             => (ecc_lower[ecc_inds], ecc_upper[ecc_inds]),
        :stellar_mass    => (stellar_mass_lower, stellar_mass_upper)
    )

    return cropped_tplot, per_inds, rad_inds, ecc_inds, cropped_bin_edges
end

function find_analysis_bin(value, lower_edges, upper_edges)
"""
Return a one-based bin index, or zero when value is outside the grid.
"""

    idx = searchsortedlast(lower_edges, value)
    
    # Fallback when the value is lower than the lowest bin
    if idx < 1 
        return 0
    end

    # Check if the value is in the largest bin
    # If so, ensure the value is within the bin edges
    if idx == lastindex(lower_edges) && value > upper_edges[idx]
        return 0
    end

    return idx
end

function create_posterior_event_weights(posterior_samples, bin_edges)
"""
Convert every planet posterior to sparse bin weights.

Orbituary samples with a prior uniform in log(sma), hence uniform in log(P),
and uses Beta(0.867, 3.03) for eccentricity. Dividing each posterior bin
probability by that interim-prior mass prevents the population fit from
double-counting the prior.
"""
    rad_lower, rad_upper   = bin_edges[:rad]
    per_lower, per_upper   = bin_edges[:per]
    ecc_lower, ecc_upper   = bin_edges[:ecc]
    star_lower, star_upper = bin_edges[:stellar_mass]

    n_rad = length(rad_lower)
    n_per = length(per_lower)
    n_ecc = length(ecc_lower)

    ecc_prior = Beta(0.867, 3.03)

    planet_bin_indices = Vector{Vector{Int}}()
    planet_bin_weights = Vector{Vector{Float64}}()
    planet_ids         = String[]

    for planet_samples in groupby(posterior_samples, :PlanetID)

        # Extract the planet ID
        planet_id = string(first(planet_samples.PlanetID))

        # Find the radius and stellar mass bin
        rad_bin  = find_analysis_bin(Float64(first(planet_samples.Rp_REarth)), rad_lower, rad_upper)
        star_bin = find_analysis_bin(Float64(first(planet_samples.M_sol)), star_lower, star_upper)

        if rad_bin == 0 || star_bin == 0
            println("Skipping $(planet_id): radius or stellar mass is outside the analysis grid")
            continue
        end

        # Dictionary to count posterior samples in each occupied (period, ecc) bin
        bin_counts = Dict{Tuple{Int,Int},Int}()
        n_inside = 0

        for sample in eachrow(planet_samples)

            # Find the (period, ecc) cell for each posterior sample
            per_bin = find_analysis_bin(Float64(sample.period), per_lower, per_upper)
            ecc_bin = find_analysis_bin(Float64(sample.ecc), ecc_lower, ecc_upper)

            if per_bin == 0 || ecc_bin == 0
                continue
            end

            # Increment the counts in the bin by 1
            key = (per_bin, ecc_bin)
            bin_counts[key] = get(bin_counts, key, 0) + 1

            n_inside += 1
        end

        if n_inside == 0
            println("Skipping $(planet_id): no posterior samples are inside the analysis grid")
            continue
        end

        indices = Int[]
        weights = Float64[]

        for ((per_bin, ecc_bin), count) in sort(collect(bin_counts))
            
            # Convert the 4D bin into a flat index
            flat_index = (
                rad_bin
                + (per_bin - 1) * n_rad
                + (ecc_bin - 1) * n_rad * n_per
                + (star_bin - 1) * n_rad * n_per * n_ecc
            )

            # Prior probability in this P/e bin
            period_prior_mass = log(per_upper[per_bin] / per_lower[per_bin])
            ecc_prior_mass    = cdf(ecc_prior, ecc_upper[ecc_bin]) - cdf(ecc_prior, ecc_lower[ecc_bin])
            
            # Combined P+e prior probability
            total_prior = period_prior_mass * ecc_prior_mass

            # Posterior probability from sample counts
            posterior_bin_probability = count / n_inside

            push!(indices, flat_index)
            push!(weights, posterior_bin_probability / total_prior)
        end

        push!(planet_bin_indices, indices)
        push!(planet_bin_weights, weights)
        push!(planet_ids, planet_id)
    end

    return planet_bin_indices, planet_bin_weights, planet_ids
end


@everywhere function power_law_smooth(model_params, analysis_regions, bin_edges)
"""
OR model defined as a power law in radius, period, stellar mass and beta function in ecc
"""
    # Unpack parameters:
    alpha   = model_params[:alpha]
    beta    = model_params[:beta]
    freq    = model_params[:freq]
    gamma   = model_params[:gamma]
    e_alpha = model_params[:e_alpha]
    e_beta  = model_params[:e_beta]

    # Extract the per, rad and ecc bin edges
    per_lower = bin_edges[:per][1]
    per_upper = bin_edges[:per][2]

    rad_lower = bin_edges[:rad][1]
    rad_upper = bin_edges[:rad][2]

    ecc_lower = bin_edges[:ecc][1]
    ecc_upper = bin_edges[:ecc][2]

    stellar_mass_lower = bin_edges[:stellar_mass][1]
    stellar_mass_upper = bin_edges[:stellar_mass][2]
    stellar_mass_centres = 0.5 .* (stellar_mass_lower .+ stellar_mass_upper)

    # ---- Radius term ---- #
    if abs(alpha+1) < 1e-3
        rad_terms = log.(rad_upper ./ rad_lower)
        rad_norm  = log(analysis_regions[:power_law_rad_max]/analysis_regions[:power_law_rad_min])
    else
        rad_terms = (rad_upper.^(alpha+1) .- rad_lower.^(alpha+1)) ./ (alpha+1)
        rad_norm  = (analysis_regions[:power_law_rad_max]^(alpha+1) - analysis_regions[:power_law_rad_min]^(alpha+1)) / (alpha+1)
    end
    # --------------------- #


    # ---- Period term ---- #
    if abs(beta + 1) < 1e-3
        per_terms = log.(per_upper ./ per_lower) 
        per_norm  = log(analysis_regions[:power_law_per_max] / analysis_regions[:power_law_per_min]) 
    else
        per_terms = (per_upper.^(beta+1) .- per_lower.^(beta+1)) ./ (beta + 1)
        per_norm  = (analysis_regions[:power_law_per_max]^(beta+1) - analysis_regions[:power_law_per_min]^(beta+1)) / (beta + 1)
    end
    # --------------------- #
    

    # ---- Eccentricity term ---- #
    ecc_centres = 0.5 .* (ecc_lower .+ ecc_upper)
    ecc_widths  = ecc_upper .- ecc_lower
    ecc_dist    = Beta(e_alpha, e_beta)
    ecc_pdf     = pdf.(ecc_dist, ecc_centres)

    # Normalize over the analysis range
    ecc_grid_fine = range(analysis_regions[:power_law_ecc_min], analysis_regions[:power_law_ecc_max], length=1000)
    ecc_norm = sum(pdf.(ecc_dist, ecc_grid_fine)) * step(ecc_grid_fine)

    ecc_terms = (ecc_pdf .* ecc_widths) ./ ecc_norm
    # --------------------------- #
    

    # ---- Build the model cube ---- #
    model = freq .* (rad_terms ./ rad_norm) .* (per_terms ./ per_norm)'

    model = reshape(model, size(model,1), size(model,2), 1) .* reshape(ecc_terms, 1, 1, :)

    stellar_term = (stellar_mass_centres ./ 1.0).^gamma  # OR normalised to 1 solar mass, hence the division by 1.0
    model = model .* reshape(stellar_term, 1, 1, 1, :)
    # ------------------------------ #

    return model
end

# === End function definitions ===

# Record start time
start_time = time()

# === Direct Imaging Set up === 

# Load the direct imaging tongue plot and related data
# direct_imaging_data is a dictionary that contains:
#   1. The tongue plot under :tplot (Size - 41 x 101 x 41 x n_stars)
#   2. The names of all the DI stars under :stellar_names (Size - n_stars)
#   3. The masses of all the DI stars under :stellar_masses (Size - n_stars)
#   4. Bin centres of all 3 tplot dimensions
#   5. Bin edges of all 3 tplot dimensions
direct_imaging_data = load_tongue_plot()

# Define stellar mass bins (to cut down dimension from a giant n_stars to a more manageable 10)
stellar_mass_bins = create_stellar_mass_bins()

# Bin the stellar masses in the tongue plot
direct_imaging_data[:tplot] = bin_tplot(direct_imaging_data[:tplot], direct_imaging_data[:stellar_masses], stellar_mass_bins[:edges])

# The tongue plot is defined over a larger region than needed
# Zero out regions not included in the fitting/likelihood calculations
di_tplot_masked, per_inds, rad_inds, ecc_inds, di_bin_edges = crop_di_region(direct_imaging_data, direct_imaging_analysis_regions, stellar_mass_bins[:edges])
di_tplot_masked .= max.(di_tplot_masked, 1e-10)

# Load one final Orbituary P/e posterior sample set per detected planet.
posterior_samples_file = joinpath(curr_dir, "1d. Planet Posterior Samples.csv")
posterior_samples = CSV.read(posterior_samples_file, DataFrame)

# Distribute each planet's posterior probability across its occupied P/e bins,
# correcting each bin for the priors used to generate the posterior.
planet_bin_indices, planet_bin_weights, planet_ids = create_posterior_event_weights(posterior_samples, di_bin_edges)

println("Loaded posterior likelihood contributions for $(length(planet_ids)) planets")


# ----------------------------  MCMC Fitting  ---------------------------------- #
@everywhere function compute_loglike_posterior(lam, planet_bin_indices, planet_bin_weights)
"""
Marginalized inhomogeneous-Poisson likelihood for uncertain planet properties.
"""

    lam_flat = vec(lam)

    # Usual Poisson expected-number penalty
    loglike = -sum(lam_flat)

    @inbounds for planet_idx in eachindex(planet_bin_indices)
        
        # Extract the probability for each bin
        indices = planet_bin_indices[planet_idx]
        weights = planet_bin_weights[planet_idx]

        marginalized_planet_prob = zero(eltype(lam_flat))

        for sample_bin_idx in eachindex(indices)
            marginalized_planet_prob += weights[sample_bin_idx] * lam_flat[indices[sample_bin_idx]]
        end

        if !(marginalized_planet_prob > zero(marginalized_planet_prob)) || !isfinite(marginalized_planet_prob)
            return -Inf
        end

        loglike += log(marginalized_planet_prob)
    end

    return loglike
end

@everywhere @model function power_law_model(planet_bin_indices, planet_bin_weights, di_tplot, di_stellar_masses, di_analysis_regions, di_bin_edges)
"""
MCMC Driver 
"""
    # Priors for broken power law parameters
    alpha    ~ Uniform(-10, 10)
    beta     ~ Uniform(-10, 10)
    gamma    ~ Uniform(-10, 10)
    freq     ~ Uniform(1e-4, 20)
    e_alpha  ~ Uniform(0.5, 10.0)
    e_beta   ~ Uniform(0.5, 200.0)

    # Modify freq using Jeffrey's prior, which makes lower values of freq more likely
    Turing.@addlogprob!(-0.5 * log(freq))
    
    # Pack parameters into a dictionary
    model_params = Dict(
        :alpha => alpha,
        :beta => beta,
        :gamma => gamma,
        :freq => freq,
        :e_alpha => e_alpha,
        :e_beta => e_beta
    )

    # Compute the model normalised by the tongue plot for the given set of OR parameters
    di_predictions = power_law_smooth(model_params, di_analysis_regions, di_bin_edges)
    di_predictions = di_predictions .* di_tplot

    # Marginalize each detected planet over its full Orbituary posterior.
    di_loglike = compute_loglike_posterior(di_predictions, planet_bin_indices, planet_bin_weights)

    # Add total log-likelihood to model
    Turing.@addlogprob!(di_loglike)
end

# Run the sampler using Turing’s NUTS sampler
model_instance = power_law_model(planet_bin_indices, planet_bin_weights, di_tplot_masked, direct_imaging_data[:stellar_masses],
    direct_imaging_analysis_regions, di_bin_edges)
# ------------------------------------------------------------------------------ #

# ------------------------------  Multi-threading  ----------------------------- #
# Set the number of chains
n_chains = N_CORES

@everywhere function run_single_chain!(model_instance; n_samples = 1000)

    chain = sample(
        model_instance,
        NUTS(0.65),
        n_samples; 
        warmup       = 1000,
        progress     = true      # Turn off progress bar, or keep it if you like
    )
    return chain
end

# Distribute the chain-running work across workers
chains = pmap(i -> run_single_chain!(model_instance), 1:n_chains)

chain = chainscat(chains...)
# ------------------------------------------------------------------------------ #

# Calculate and print the elapsed time
elapsed_time = time() - start_time
println("Elapsed time: $(elapsed_time) seconds")

# OR parameters to save
parameters = ["alpha", "beta", "gamma", "freq", "e_alpha", "e_beta"]

println("Best-fit parameters with 1σ uncertainties:")
for p in parameters
    # Extract all samples for each of the parameters
    samples = vec(chain[Symbol(p)])
    
    # Compute the 16th, 50th, and 84th percentiles
    q16, q50, q84 = quantile(samples, [0.16, 0.5, 0.84])
    
    # Calculate the lower and upper uncertainties
    lower_err = q50 - q16
    upper_err = q84 - q50
    
    println(" $(p) = $(round(q50, digits=4)) +$(round(upper_err, digits=4)) / -$(round(lower_err, digits=4))")
end

# Convert the chain to a DataFrame
df = DataFrame(chain)

# Save the DataFrame to a CSV file.
CSV.write(joinpath(curr_dir, "1e. Fit, N = 1e4.csv"), df)
