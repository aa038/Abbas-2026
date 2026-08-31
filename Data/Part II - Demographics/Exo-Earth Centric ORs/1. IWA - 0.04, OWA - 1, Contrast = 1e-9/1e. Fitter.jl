"""
OR Fitting (Part 5 of 9)
-------------------------------------------
This script uses the tongue plot and and an assumed OR model, to fit the list of detected planets.
The result is the full list of MCMC posteriors for all the OR parameters

Input:
    5b. 4D Tongue Plot.npz      # The 4D tongue plot stored a 4D NumPy array (From Part 2)
    5d. Planet Posterior Samples.csv
                                # Final Orbituary P/e posterior samples (From Part 4)


Output:
    5e. Fit, N = 1e4.csv        # The full list of MCMC posteriors for each parameter in the OR model
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

# Multiprocessing
N_CORES              = 5              # Number of cores this script is run on. This is SOLELY dependent on how many cores you have on your machine
                                       # If you are unsure, set it equal to 1
                                       # DO NOT FUCK AROUND WITH THIS NUMBER. YOUR COMPUTER WILL FREEZE, AT BEST 

# Analysis boundaries (the regions over which the power‐law normalization and fitting are done)
# Radius is truncated to the exo-Earth interval. The tongue plot is already
# zero outside each star's fully-in-HZ period/eccentricity region.
# The first set of variables is the rad x per x ecc region over which we normalize the power law
# This is region over which the OR applies

# The second set of 4 is the region over which the fitting is done
# These are the ranges used in the likelihood calculations
direct_imaging_analysis_regions = Dict(
    :power_law_rad_min  => 0.8,
    :power_law_rad_max  => 1.4,
    :power_law_per_min  => 0.03,
    :power_law_per_max  => 6.0,
    :power_law_ecc_min  => 0.0001,
    :power_law_ecc_max  => 0.99,

    :analysis_rad_min   => 0.8,
    :analysis_rad_max   => 1.4,
    :analysis_per_min   => 0.03,
    :analysis_per_max   => 6.0,
    :analysis_ecc_min   => 0.0001,
    :analysis_ecc_max   => 0.99
)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #

addprocs(N_CORES)

@everywhere using Turing, MCMCChains, DataFrames, CSV, Distributions
@everywhere using StatsBase: quantile
@everywhere using ForwardDiff
@everywhere using LinearAlgebra

# Define type aliases for convenience (using the default Float64)
const VI   = Vector{Int}
const VF   = Vector{Float64}
const VStr = Vector{String}
const AF3  = Array{Float64,3}
const AF4  = Array{Float64,4}

# Define directory paths (assumes the data are in a "Data" folder next to this file)
const curr_dir     = @__DIR__
const parent_dir   = dirname(dirname(dirname(curr_dir)))
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

function crop_di_region(tplot_data, analysis_regions::Dict{Symbol,Float64})
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
        :ecc             => (ecc_lower[ecc_inds], ecc_upper[ecc_inds])
    )

    return cropped_tplot, per_inds, rad_inds, ecc_inds, cropped_bin_edges
end


function load_planet_posterior_samples(filepath::String)::DataFrame
"""
Load the final Orbituary period/eccentricity posterior samples exported by
5d. List of detected planets.py.
"""
    df = CSV.read(filepath, DataFrame)
    required = [:PlanetID, :StarName, :period, :ecc, :Rp_REarth, :M_sol]
    missing_columns = setdiff(required, propertynames(df))

    if !isempty(missing_columns)
        error("Posterior sample file is missing columns: $(missing_columns)")
    end

    return df
end


function find_analysis_bin(value, lower_edges, upper_edges)
"""Return a one-based bin index, or zero when value is outside the grid."""
    if !isfinite(value)
        return 0
    end

    idx = searchsortedlast(lower_edges, value)
    if idx < 1 || idx > length(lower_edges)
        return 0
    end

    is_last_bin = idx == length(lower_edges)
    inside = value < upper_edges[idx] || (is_last_bin && value <= upper_edges[idx])
    return inside ? idx : 0
end


function create_posterior_event_weights(posterior_samples, bin_edges, stellar_names)
"""
Convert every planet posterior to sparse, importance-corrected bin weights.

Each planet remains one catalog event. Its possible period/eccentricity bin
assignments are marginalized in the likelihood instead of being converted to
fractional Poisson counts.

Orbituary samples with a prior uniform in log(sma), hence uniform in log(P),
and uses Beta(0.867, 3.03) for eccentricity. Dividing each posterior bin
probability by that interim-prior mass prevents the population fit from
double-counting the Orbituary prior.
"""
    rad_lower, rad_upper = bin_edges[:rad]
    per_lower, per_upper = bin_edges[:per]
    ecc_lower, ecc_upper = bin_edges[:ecc]

    n_rad = length(rad_lower)
    n_per = length(per_lower)
    n_ecc = length(ecc_lower)

    ecc_prior = Beta(0.867, 3.03)

    star_lookup = Dict(name => i for (i, name) in enumerate(stellar_names))

    event_bin_indices = Vector{Vector{Int}}()
    event_bin_weights = Vector{Vector{Float64}}()
    event_planet_ids = String[]

    for planet_samples in groupby(posterior_samples, :PlanetID)
        planet_id = string(first(planet_samples.PlanetID))
        rad_bin   = find_analysis_bin(Float64(first(planet_samples.Rp_REarth)), rad_lower, rad_upper)
        star_name = String(first(planet_samples.StarName))
        star_bin  = get(star_lookup, star_name, 0)

        if rad_bin == 0 || star_bin == 0
            println("Skipping $(planet_id): radius or host star is outside the analysis grid")
            continue
        end

        # Count posterior samples in each occupied (period, eccentricity) bin.
        bin_counts = Dict{Tuple{Int,Int},Int}()
        n_inside = 0

        for sample in eachrow(planet_samples)
            per_bin = find_analysis_bin(Float64(sample.period), per_lower, per_upper)
            ecc_bin = find_analysis_bin(Float64(sample.ecc), ecc_lower, ecc_upper)

            if per_bin == 0 || ecc_bin == 0
                continue
            end

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
            # Column-major flat index of [radius, period, eccentricity, star].
            flat_index = (
                rad_bin
                + (per_bin - 1) * n_rad
                + (ecc_bin - 1) * n_rad * n_per
                + (star_bin - 1) * n_rad * n_per * n_ecc
            )

            # Interim prior probability mass in this P/e bin, up to a global
            # normalization constant that is independent of population params.
            log_period_prior_mass = log(per_upper[per_bin] / per_lower[per_bin])
            ecc_prior_mass = (cdf(ecc_prior, ecc_upper[ecc_bin]) - cdf(ecc_prior, ecc_lower[ecc_bin]))
            interim_prior_mass = log_period_prior_mass * ecc_prior_mass

            if interim_prior_mass <= 0 || !isfinite(interim_prior_mass)
                continue
            end

            posterior_bin_probability = count / n_inside
            push!(indices, flat_index)
            push!(weights, posterior_bin_probability / interim_prior_mass)
        end

        if isempty(indices)
            println("Skipping $(planet_id): no bins have finite interim-prior mass")
            continue
        end

        push!(event_bin_indices, indices)
        push!(event_bin_weights, weights)
        push!(event_planet_ids, planet_id)
    end

    return event_bin_indices, event_bin_weights, event_planet_ids
end


@everywhere function power_law_smooth(model_params, analysis_regions, bin_edges)
"""
OR model defined as a power law in radius, period, stellar mass and beta function in ecc
"""
    # Unpack parameters:
    alpha   = model_params[:alpha]
    beta    = model_params[:beta]
    freq    = model_params[:freq]
    e_alpha = model_params[:e_alpha]
    e_beta  = model_params[:e_beta]

    # Extract the per, rad and ecc bin edges
    per_lower = bin_edges[:per][1]
    per_upper = bin_edges[:per][2]

    rad_lower = bin_edges[:rad][1]
    rad_upper = bin_edges[:rad][2]

    ecc_lower = bin_edges[:ecc][1]
    ecc_upper = bin_edges[:ecc][2]

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
    # ------------------------------ #

    return model
end

# === End function definitions ===

# Record start time
start_time = time()

# === Direct Imaging Set up === 

# Load the direct imaging tongue plot and related data
# direct_imaging_data is a dictionary that contains:
#   1. The tongue plot under :tplot (Size - 61 x 101 x 41 x n_stars)
#   2. The names of all the DI stars under :stellar_names (Size - n_stars)
#   3. The masses of all the DI stars under :stellar_masses (Size - n_stars)
#   4. Bin centres of all 3 tplot dimensions
#   5. Bin edges of all 3 tplot dimensions
direct_imaging_data = load_tongue_plot()

# The tongue plot is defined over a larger region than needed
# Zero out regions not included in the fitting/likelihood calculations
di_tplot_masked, per_inds, rad_inds, ecc_inds, di_bin_edges = crop_di_region(direct_imaging_data, direct_imaging_analysis_regions)

n_stars = size(di_tplot_masked, 4)

di_tplot_3d = reshape(di_tplot_masked, :, n_stars)

# Load one final Orbituary P/e posterior sample set per detected planet.
posterior_samples_file = joinpath(curr_dir, "1d. Planet Posterior Samples.csv")
posterior_samples = load_planet_posterior_samples(posterior_samples_file)

# Precompute sparse posterior-bin probabilities and divide out Orbituary's
# interim P/e prior. These arrays do not depend on the population parameters.
posterior_bin_indices, posterior_bin_weights, posterior_planet_ids = create_posterior_event_weights(posterior_samples, di_bin_edges, direct_imaging_data[:stellar_names])

println("Loaded posterior likelihood contributions for $(length(posterior_planet_ids)) planets")


# ----------------------------  MCMC Fitting  ---------------------------------- #
@everywhere function compute_loglike_posterior(base_model,completeness_matrix,stellar_masses,gamma,event_bin_indices,event_bin_weights)
    # base_model is radius × period × eccentricity
    model_flat = vec(base_model)
    n_cells = length(model_flat)

    # One occurrence-rate scaling per star
    star_weights = stellar_masses .^ gamma

    # completeness_matrix has shape:
    # (radius × period × eccentricity) × star
    weighted_completeness = completeness_matrix * star_weights

    # Expected number of detected planets
    loglike = -dot(model_flat, weighted_completeness)

    # Linear indexing of completeness_matrix matches the original flattened
    # radius × period × eccentricity × star tongue plot
    completeness_flat = vec(completeness_matrix)

    for planet_idx in eachindex(event_bin_indices)
        indices = event_bin_indices[planet_idx]
        weights = event_bin_weights[planet_idx]

        event_rate = zero(loglike)

        for sample_idx in eachindex(indices)
            flat_index = indices[sample_idx]

            # Convert the original 4D flat index into its 3D cell and star.
            cell_index = mod1(flat_index, n_cells)
            star_index = div(flat_index - 1, n_cells) + 1

            event_rate += (weights[sample_idx]
                * model_flat[cell_index]
                * star_weights[star_index]
                * completeness_flat[flat_index]
            )
        end

        if !(event_rate > zero(event_rate)) || !isfinite(event_rate)
            return -Inf
        end

        loglike += log(event_rate)
    end

    return loglike
end

@everywhere @model function power_law_model(event_bin_indices, event_bin_weights, di_tplot_3d, di_stellar_masses, di_analysis_regions, di_bin_edges)
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

    base_model = power_law_smooth(model_params, di_analysis_regions, di_bin_edges)

    di_loglike = compute_loglike_posterior(base_model, di_tplot_3d, di_stellar_masses, gamma, event_bin_indices, event_bin_weights)

    # Add total log-likelihood to model
    Turing.@addlogprob!(di_loglike)
end

# Run the sampler using Turing’s NUTS sampler
model_instance = power_law_model(
    posterior_bin_indices,
    posterior_bin_weights,
    di_tplot_3d,
    direct_imaging_data[:stellar_masses],
    direct_imaging_analysis_regions,
    di_bin_edges
)
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

# Convert the chain to a DataFrame.
df = DataFrame(chain)

# Save the DataFrame to a CSV file.
CSV.write(joinpath(curr_dir, "1e. Fit, N = 1e4.csv"), df)
