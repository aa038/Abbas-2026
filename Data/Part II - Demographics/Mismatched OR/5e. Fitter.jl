using CSV
using DataFrames
using NPZ
using Statistics
using Optim
using Distributions
using SpecialFunctions
using Turing
using MCMCChains
using Distributed
using Glob

addprocs(11)

@everywhere using Turing, MCMCChains, DataFrames, CSV
@everywhere using StatsBase: quantile
@everywhere using StatsFuns: logfactorial
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
const data_dir     = joinpath(parent_dir, "Data")

# === Function definitions ===

function load_direct_imaging_tongue_plots()::Dict{Symbol,Any}
    
    # Load the tongue plot and star list
    tplot_file  = joinpath(curr_dir, "5b. 4D Tongue Plot.npz")
    stellar_file = joinpath(data_dir, "HWO Stars.csv")

    # Load up the star details 
    # We need the name and mass in solar masses
    df = CSV.read(stellar_file, DataFrame)
    stellar_masses = Float64.(df.M)
    stellar_names = String.(df.HDName)

    # Then we read in the tongue plot for the direct imaging data
    # This will be a 4D array of shape 60 x 100 x 20 x 98
    # The axes are planet_mass x sma x ecc x star
    tplot_data = npzread(tplot_file)
    tplot = tplot_data["completeness"]

    # Tplot grids and centres
    mass_centers = tplot_data["mass_centers"]
    sma_centers  = tplot_data["sma_centers"]
    ecc_centers  = tplot_data["ecc_centers"]

    mass_edges = tplot_data["mass_edges"]
    sma_edges  = tplot_data["sma_edges"]
    ecc_edges  = tplot_data["ecc_edges"]
 
    # Find indices of high-mass stars
    high_mass_indices = findall(m -> m >= 0.3 && m <= 1.5, stellar_masses)

    # Filter the tongue plot to only include high-mass stars
    tplot = tplot[:,:,:,high_mass_indices]
    
    # Filter the stellar masses and names
    stellar_masses = stellar_masses[high_mass_indices]
    stellar_names = stellar_names[high_mass_indices]

    # The tongue plot, and the star details are returned as a dictionary
    return Dict(
        :tplot           => tplot,
        :stellar_masses  => stellar_masses,
        :stellar_names   => stellar_names,
        :mass_centers    => mass_centers,
        :sma_centers     => sma_centers,
        :ecc_centers     => ecc_centers,
        :mass_edges      => mass_edges,
        :sma_edges       => sma_edges,
        :ecc_edges       => ecc_edges
    )
end


function create_stellar_mass_bins(n_stellar_bins = 30)

    # Define the edges of the stellar mass bins
    stellar_mass_edges = range(0.3, 1.5, length = n_stellar_bins+1)

    # Compute the bin centres
    stellar_mass_centres = (stellar_mass_edges[1:end-1] .+ stellar_mass_edges[2:end]) ./ 2

    return Dict(
        :edges => stellar_mass_edges,
        :centres => stellar_mass_centres
    )
end

function bin_DI_tplot(di_tplot, stellar_masses, stellar_mass_edges)
    # Find the number of bins
    # We subtract 1 because 30 bins have 31 edges
    n_bins = length(stellar_mass_edges) - 1

    # Get the size of the first three dimensions from the original DI tplot
    n_mass, n_sma, n_ecc, _ = size(di_tplot)

    # Create a new array for binned tplot data
    binned_tplot = zeros(n_mass, n_sma, n_ecc, n_bins)

    # Loop over each stellar mass bin
    for i in 1:n_bins

        # Find the lower and upper edges of the current mass bin
        lower = stellar_mass_edges[i]
        upper = stellar_mass_edges[i+1]

        # Find the indices of all the stars that fall within the current mass bin
        star_indices = findall(s -> (s >= lower && s < upper), stellar_masses)

        if !isempty(star_indices)
            # If there are stars in the bin, sum over the contributions
            binned_tplot[:,:,:,i] = sum(di_tplot[:,:,:,star_indices], dims = 4)
        else
            # If no stars fall in this bin, leave it as zeros
            binned_tplot[:,:,:,i] .= 0.0
        end
    end

    return binned_tplot

end

function crop_di_region(tplot_data, analysis_regions::Dict{Symbol,Float64}, stellar_mass_edges)

    # Extract the tongue plot
    tplot = tplot_data[:tplot]
    
    # Now compute the lower and upper bin edges for mass, sma, ecc and stellar mass
    mass_lower = tplot_data[:mass_edges][1:end-1]
    mass_upper = tplot_data[:mass_edges][2:end]

    sma_lower  = tplot_data[:sma_edges][1:end-1]
    sma_upper  = tplot_data[:sma_edges][2:end]

    ecc_lower  = tplot_data[:ecc_edges][1:end-1]
    ecc_upper  = tplot_data[:ecc_edges][2:end]

    stellar_mass_lower = stellar_mass_edges[1:end-1]
    stellar_mass_upper = stellar_mass_edges[2:end]

    # Indices of bins that intersect the analysis region
    sma_inds = findall(i -> sma_upper[i] > analysis_regions[:analysis_sma_min] && sma_lower[i] < analysis_regions[:analysis_sma_max], 1:length(sma_lower))
    mass_inds = findall(i -> mass_upper[i] > analysis_regions[:analysis_mass_min] && mass_lower[i] < analysis_regions[:analysis_mass_max], 1:length(mass_lower))
    ecc_inds = findall(i -> ecc_upper[i] > analysis_regions[:analysis_ecc_min] && ecc_lower[i] < analysis_regions[:analysis_ecc_max], 1:length(ecc_lower))

    # Crop the 3D tplot (sep x sma x mass)
    cropped_tplot = tplot[mass_inds, sma_inds, ecc_inds, :]

    # Cropped bin edges for use in modeling
    cropped_bin_edges = Dict(
        :semi_major_axis => (sma_lower[sma_inds], sma_upper[sma_inds]),
        :mass            => (mass_lower[mass_inds], mass_upper[mass_inds]),
        :ecc             => (ecc_lower[ecc_inds], ecc_upper[ecc_inds]),
        :stellar_mass    => (stellar_mass_lower, stellar_mass_upper)
    )

    return cropped_tplot, sma_inds, mass_inds, ecc_inds, cropped_bin_edges
end

function load_detected_planets(filepath::String)::Dict{Symbol, Vector}
    df = CSV.read(filepath, DataFrame)

    return Dict(
        :names         => String.(df.StarName),
        :sma           => Float64.(df.SMA_AU),       
        :mass          => Float64.(df.Mp_MEarth),    
        :ecc           => Float64.(df.ecc),
        :stellar_mass  => Float64.(df.M_sol)
    )
end

function place_directly_imaged_planets!(N_obsdata, planet_properties, bin_edges)

    # Extract bin lower edges
    mass_lower, _          = bin_edges[:mass]
    sma_lower, _           = bin_edges[:semi_major_axis]
    ecc_lower, _           = bin_edges[:ecc]
    stellar_mass_lower, _  = bin_edges[:stellar_mass]

    # Find the right bins the planets go into.
    # searchsortedlast() finds the last index in the sorted sep_lower array,
    # where an element is less than or equal to planet_properties.separation.
    # This ensures the observed planet is placed into the right grid in separation.
    # Find bin indices
    m_bin = searchsortedlast(mass_lower, planet_properties.mass)
    sma_bin = searchsortedlast(sma_lower, planet_properties.sma)
    ecc_bin = searchsortedlast(ecc_lower, planet_properties.ecc)
    star_bin = searchsortedlast(stellar_mass_lower, planet_properties.stellar_mass)

    n_m        = length(mass_lower)
    n_sma      = length(sma_lower)
    n_ecc      = length(ecc_lower)
    n_star     = length(stellar_mass_lower)

    # Only increment if all bins are valid (i.e., inside the edges)
    if 1 ≤ m_bin ≤ n_m &&
        1 ≤ sma_bin ≤ n_sma &&
        1 ≤ ecc_bin ≤ n_ecc &&
        1 ≤ star_bin ≤ n_star
         N_obsdata[m_bin, sma_bin, ecc_bin, star_bin] += 1
     end
end


function create_di_planets_grid(direct_imaging_planets, bin_edges, n_stellar_bins)

    # Find the length of the separation and mass grids to create a 3D array to place observed planets in.
    # The mass, separation and stars grids have the same size as the tongue plot
    n_mass_bins = length(bin_edges[:mass][1])
    n_sma_bins  = length(bin_edges[:semi_major_axis][1])
    n_ecc_bins  = length(bin_edges[:ecc][1])

    N_obsdata = zeros(n_mass_bins, n_sma_bins, n_ecc_bins, n_stellar_bins)

    for i in 1:length(direct_imaging_planets[:names])
        props = (
            mass         = direct_imaging_planets[:mass][i],
            sma          = direct_imaging_planets[:sma][i],
            ecc          = direct_imaging_planets[:ecc][i],
            stellar_mass = direct_imaging_planets[:stellar_mass][i]
        )
        place_directly_imaged_planets!(N_obsdata, props, bin_edges)
    end

    return N_obsdata
end


@everywhere function power_law_smooth(model_params, analysis_regions, bin_edges)

    # Unpack parameters:
    alpha   = model_params[:alpha]
    beta    = model_params[:beta]
    freq    = model_params[:freq]
    gamma   = model_params[:gamma]
    e_alpha = model_params[:e_alpha]
    e_beta  = model_params[:e_beta]

    # Extract the sma and mass bin edges
    sma_lower = bin_edges[:semi_major_axis][1]
    sma_upper = bin_edges[:semi_major_axis][2]

    mass_lower = bin_edges[:mass][1]
    mass_upper = bin_edges[:mass][2]

    ecc_lower = bin_edges[:ecc][1]
    ecc_upper = bin_edges[:ecc][2]

    stellar_mass_lower = bin_edges[:stellar_mass][1]
    stellar_mass_upper = bin_edges[:stellar_mass][2]
    stellar_mass_centres = 0.5 .* (stellar_mass_lower .+ stellar_mass_upper)

    # === Mass term ===
    if abs(alpha+1) < 1e-3
        mass_terms = log.(mass_upper ./ mass_lower)
        mass_norm  = log(analysis_regions[:power_law_mass_max]/analysis_regions[:power_law_mass_min])
    else
        mass_terms = (mass_upper.^(alpha+1) .- mass_lower.^(alpha+1)) ./ (alpha+1)
        mass_norm  = (analysis_regions[:power_law_mass_max]^(alpha+1) - analysis_regions[:power_law_mass_min]^(alpha+1)) / (alpha+1)
    end

    # === SMA term ===
    if abs(beta + 1) < 1e-3
        sma_terms = log.(sma_upper ./ sma_lower) ./(beta + 1)
        sma_norm  = log(analysis_regions[:power_law_sma_max] / analysis_regions[:power_law_sma_min]) /(beta + 1)
    else
        sma_terms = (sma_upper.^(beta+1) .- sma_lower.^(beta+1)) 
        sma_norm  = (analysis_regions[:power_law_sma_max]^(beta+1) - analysis_regions[:power_law_sma_min]^(beta+1)) 
    end

    # === Eccentricity term (Beta distribution) ===
    ecc_centres = 0.5 .* (ecc_lower .+ ecc_upper)
    ecc_widths  = ecc_upper .- ecc_lower
    ecc_dist    = Beta(e_alpha, e_beta)
    ecc_pdf     = pdf.(ecc_dist, ecc_centres)

    # Normalize over the analysis range
    ecc_grid_fine = range(analysis_regions[:power_law_ecc_min], analysis_regions[:power_law_ecc_max], length=1000)
    ecc_norm = sum(pdf.(ecc_dist, ecc_grid_fine)) * step(ecc_grid_fine)

    ecc_terms = (ecc_pdf .* ecc_widths) ./ ecc_norm

    # === Build the model cube ===
    model = freq .* (mass_terms ./ mass_norm) .* (sma_terms ./ sma_norm)'

    model = reshape(model, size(model,1), size(model,2), 1) .* reshape(ecc_terms, 1, 1, :)

    stellar_term = (stellar_mass_centres ./ 1.0).^gamma
    model = model .* reshape(stellar_term, 1, 1, 1, :)

    return model
end

# === End function definitions ===

# Record start time
start_time = time()

# === Direct Imaging Set up === 

# Load the direct imaging tongue plot and related data
# direct_imaging_data is a dictionary that contains:
#   1. The tongue plot under :tplot (Size - 81 x 81 x 41 x n_stars)
#   2. The names of all the DI stars under :stellar_names (Size - n_stars)
#   3. The masses of all the DI stars under :stellar masses (Size - n_stars)
direct_imaging_data = load_direct_imaging_tongue_plots()

# Analysis boundaries (the regions over which the power‐law normalization and fitting are done)
# The first set of 4 variables is the mass x sma region over which we normalize the power law
# The second set of 4 is the region over which the fitting is done
direct_imaging_analysis_regions = Dict(
    :power_law_mass_min => 0.01,
    :power_law_mass_max => 40.0,
    :power_law_sma_min  => 0.09,
    :power_law_sma_max  => 4.9,
    :power_law_ecc_min  => 0.0001,
    :power_law_ecc_max  => 0.95,

    :analysis_mass_min  => 0.01,
    :analysis_mass_max  => 40.0,
    :analysis_sma_min   => 0.09,
    :analysis_sma_max   => 4.9,
    :analysis_ecc_min   => 0.0001,
    :analysis_ecc_max   => 0.95
)

# Define stellar mass bins (to cut down dimension from a giant n_stars to a more manageable 30)
stellar_mass_bins = create_stellar_mass_bins()

# Bin the stellar masses in the tongue plot
direct_imaging_data[:tplot] = bin_DI_tplot(direct_imaging_data[:tplot], direct_imaging_data[:stellar_masses], stellar_mass_bins[:edges])

# The tongue plot is defined over a larger region than needed; zero out unwanted regions
di_tplot_masked, sma_inds, mass_inds, ecc_inds, di_bin_edges = crop_di_region(direct_imaging_data, direct_imaging_analysis_regions, stellar_mass_bins[:edges])
di_tplot_masked .= max.(di_tplot_masked, 1e-10)

# Load the directly imaged planets and their properties
detected_planets_file = joinpath(curr_dir, "5d. Detected Planets.csv")
direct_imaging_planets = load_detected_planets(detected_planets_file)

# Place the observed planets into a 3D grid of sep vs planet mass vs stars
obs_di_planets_grid = create_di_planets_grid(direct_imaging_planets, di_bin_edges, length(stellar_mass_bins[:centres]))


# === Define the Turing model ===

# Helper function to compute log-likelihood
@everywhere function compute_loglike(obs, lam)

    # We'll accumulate log-likelihood in a single scalar
    loglike = 0.0


    @inbounds for i in 1:length(obs)
        # Poisson log-pdf => k*log(lam) - lam - logfactorial(k)
        if lam[i] > 0
            k = Int(obs[i])
            loglike += k * log(lam[i]) - lam[i] - logfactorial(k)
        else
            # If lam=0 but obs>0 => impossible
            if obs[i] > 0
                return -Inf
            end
            # else (lam=0, obs=0) => logprob=0
        end
    end
    return loglike
end

@everywhere @model function power_law_model(obs_di_grid, di_tplot, di_stellar_masses, di_analysis_regions, di_bin_edges)

    # Priors for broken power law parameters
    alpha    ~ Uniform(-10, 10)
    beta     ~ Uniform(-10, 10)
    gamma    ~ Uniform(-10, 10)
    freq     ~ Uniform(1, 20)
    e_alpha  ~ Uniform(0.5, 10.0)
    e_beta   ~ Uniform(0.5, 80.0)

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

    # Calculate log-likelihood for DI data
    di_predictions = power_law_smooth(model_params, di_analysis_regions, di_bin_edges)

    di_predictions_detected = di_predictions .* di_tplot

    # Compute log-likelihood
    di_loglike = compute_loglike(vec(obs_di_grid), vec(di_predictions_detected))

    # Add total log-likelihood to model
    Turing.@addlogprob!(di_loglike)
end


# Run the sampler using Turing’s NUTS sampler
model_instance = power_law_model(obs_di_planets_grid, di_tplot_masked, direct_imaging_data[:stellar_masses], direct_imaging_analysis_regions, di_bin_edges)

# Set the number of chains
n_chains = 10

@everywhere function run_single_chain!(
    model_instance;
    n_samples = 1000
)
    # We can do a single-chain sample here
    # - pass `NUTS(0.65)` or whichever sampler you prefer
    # - pass e.g. `MCMCSerial()` or no special argument so it doesn't do internal threading
    chain = sample(
        model_instance,
        NUTS(0.65),
        n_samples; 
        warmup       = 1000,
        progress     = true,      # turn off progress bar, or keep it if you like
    )
    return chain
end


# Distribute the chain-running work across workers.
# Note: We're mapping over indices (or directly over the init_theta values).
chains = pmap(i -> run_single_chain!(model_instance), 1:n_chains)

chain = chainscat(chains...)

# Calculate and print the elapsed time
elapsed_time = time() - start_time
println("Elapsed time: $(elapsed_time) seconds")

# List the parameters you want to report.
parameters = ["alpha", "beta", "gamma", "freq", "e_alpha", "e_beta"]

println("Best-fit parameters with 1σ uncertainties:")
for p in parameters
    # Correctly extract all samples for parameter `p` using the one-index shorthand.
    samples = vec(chain[Symbol(p)])
    
    # Compute the 16th, 50th, and 84th percentiles.
    q16, q50, q84 = quantile(samples, [0.16, 0.5, 0.84])
    
    # Calculate the lower and upper uncertainties.
    lower_err = q50 - q16
    upper_err = q84 - q50
    
    println(" $(p) = $(round(q50, digits=4)) +$(round(upper_err, digits=4)) / -$(round(lower_err, digits=4))")
end

# Convert the chain to a DataFrame.
df = DataFrame(chain)

# Save the DataFrame to a CSV file.
CSV.write(joinpath(curr_dir, "5e. Fit, N = 1e4.csv"), df)
