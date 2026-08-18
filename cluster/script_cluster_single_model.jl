############################################################
# Australian Electricity Demand:
# Expanding-Window Forecast Experiment
############################################################

### julia "C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST\cluster\script_cluster_single_model.jl"

using Pkg

# If script is e.g.
# TVSAR_FORECAST/experiments/electricity/expanding_window.jl
#Pkg.activate(abspath(joinpath(@__DIR__, "..")))
Pkg.activate(
    raw"C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST"
)

using Revise
using TVSAR_FORECAST

using CSV
using DataFrames
using Dates
using Statistics
using Random
using LinearAlgebra
using PDMats
using Distributions
using JLD2


############################################################
# 1. EXPERIMENT CONTROLS
#
# CHANGE THESE FIRST
############################################################


############################################################
# RUN OR COLLECT
############################################################

ACTION =
    Symbol(
        get(
            ENV,
            "TVSAR_ACTION",
            "run"
        )
    )

@assert ACTION in (:run, :collect)

# ----------------------------------------------------------
# Mini run / full run
# ----------------------------------------------------------

mini_run = true

if mini_run

    nOrigins     = 5
    nBurn        = 50
    nIter        = 50
    nPredPerIter = 2
    thinFactor   = 1

else

    nOrigins     = 20
    nBurn        = 2000
    nIter        = 2000
    nPredPerIter = 20
    thinFactor   = 1

end


# ----------------------------------------------------------
# Forecast horizons
#
# Examples:
#   collect(1:168)    -> every hour for one week
#   [1, 6, 12, 24]   -> selected short horizons
#   [1, 24, 72, 168] -> selected horizons up to one week
# ----------------------------------------------------------

forecastHorizons = collect(1:168)

maxHorizon = maximum(forecastHorizons)


# ----------------------------------------------------------
# Forecast origins
# ----------------------------------------------------------

# First observation to forecast.
# Training ends exactly one hour before this.
first_origin = DateTime(2005, 12, 14, 4)

# Distance between forecast origins, in observations/hours.
#
# Minimum for non-overlapping 168-hour test windows:
# origin_spacing = 168
#
# Recommended with nPerGroup = 720:
# origin_spacing = 720
#
# This gives non-overlapping tests AND advances estimation
# by exactly one complete state block.

origin_spacing = 24 * 30        # 720 hours


# ----------------------------------------------------------
# Forecast state grouping
# ----------------------------------------------------------

# 1:
# evolve forecast states hourly
#
# nPerGroup:
# preserve fitted 30-day grouping in forecasting

g_forecast = 1


# ----------------------------------------------------------
# Random seed
# ----------------------------------------------------------

base_seed = 1251222


############################################################
# 2. MODEL SETTINGS
############################################################

model_type = :SAR

obs_var_type   = :static
state_var_type = :DSP

INTERCEPT = true

intercept_dynamics =
    INTERCEPT ? :rw : nothing

startcol =
    INTERCEPT ?
    (intercept_dynamics == :ll ? 3 : 2) :
    1


# ----------------------------------------------------------
# Group size used for ESTIMATION
# ----------------------------------------------------------

nPerGroup = 24 * 30


# ----------------------------------------------------------
# SAR structure
# ----------------------------------------------------------

season = s1 = s2 = [1, 24, 24*7]
p      = p1 = p2 = [1, 1, 1]

pFit = sum(p1)

p_max = [
    sum(p1 .* s1),
    sum(p2 .* s2)
]

nLags =
    pFit +
    (INTERCEPT ?
        (intercept_dynamics == :ll ? 2 : 1) :
        0)


# ----------------------------------------------------------
# Other model options
# ----------------------------------------------------------

SAR_conditional = false
use_fourier     = false

ztrans          = "partials"
clipped_partials = true
p_threshold      = 0.99

iterated = false
num_iters = iterated ? 5 : 1
kf_method = :iekf


############################################################
# 3. DATA
############################################################

data_file = get(
    ENV,
    "TVSAR_DATA_FILE",
    raw"C:\Users\Anna Fagerberg\Desktop\PROJECTS IN JULIA\DATA\AustralianElecPriceDemand202512_hourly.csv"
)

df = CSV.read(
    data_file,
    DataFrame
)

# Ensure chronological ordering
sort!(df, :datetime)
dates = df.datetime
data  = df.demand_VIC


############################################################
# 4. LOCATE FORECAST ORIGINS
############################################################

first_origin_idx =
    findfirst(==(first_origin), dates)

isnothing(first_origin_idx) &&
    error("first_origin not found in data: $first_origin")


origin_idx = [
    first_origin_idx +
    (j - 1) * origin_spacing
    for j in 1:nOrigins
]


# ----------------------------------------------------------
# Basic checks
# ----------------------------------------------------------

@assert origin_spacing >= maxHorizon """
Forecast test windows overlap:
origin_spacing = $origin_spacing
maxHorizon     = $maxHorizon
"""

@assert origin_idx[end] + maxHorizon - 1 <= length(data) """
Not enough observations after final forecast origin.
"""


origin_dates = dates[origin_idx]

println()
println("============================================")
println("Forecast experiment")
println("============================================")
println("Number of origins : ", nOrigins)
println("Forecast horizons : ", forecastHorizons)
println("Max horizon       : ", maxHorizon)
println("Origin spacing    : ", origin_spacing, " hours")
println("First origin      : ", first(origin_dates))
println("Last origin       : ", last(origin_dates))
println("============================================")
println()


############################################################
# 5. SLURM / LOCAL EXECUTION
############################################################


on_slurm =
    haskey(ENV, "SLURM_ARRAY_TASK_ID")

if on_slurm

    slurm_id =
        parse(
            Int,
            ENV["SLURM_ARRAY_TASK_ID"]
        )

    @assert 1 <= slurm_id <= nOrigins

    # One forecast origin per worker
    origins_to_run = [slurm_id]

else

    # Optional:
    #
    # julia expanding_window.jl 3
    #
    # runs only origin 3 locally.

    if isempty(ARGS)

        origins_to_run =
            collect(1:nOrigins)

    else

        local_id =
            parse(Int, ARGS[1])

        @assert 1 <= local_id <= nOrigins

        origins_to_run =
            [local_id]

    end

end


println(
    "Origins handled by this worker: ",
    origins_to_run
)


############################################################
# 6. RESULT STORAGE
############################################################

nH = length(forecastHorizons)

LPSs = fill(NaN, nOrigins, nH)
MAEs = fill(NaN, nOrigins, nH)

elapsed_fit      = fill(NaN, nOrigins)
elapsed_forecast = fill(NaN, nOrigins)


resultsFolder = joinpath(
    @__DIR__,
    "results"
)

mkpath(resultsFolder)




############################################################
# COLLECT MODE
############################################################

if ACTION == :collect

    collect_results(
        resultsFolder,
        nOrigins,
        forecastHorizons
    )

    exit()

end


############################################################
# 7. EXPANDING-WINDOW LOOP
############################################################

for origin_id in origins_to_run

    println()
    println("################################################")
    println("Origin $origin_id / $nOrigins")
    println("Forecast starts: ", origin_dates[origin_id])
    println("################################################")


    # ========================================================
    # 7.1 Expanding training sample
    # ========================================================

    test_start_idx = origin_idx[origin_id]

    train_end_idx = test_start_idx - 1


    # Fixed beginning -> expanding end
    train_start_idx =findfirst(==(DateTime(2001, 1, 9, 2)),dates)

    isnothing(train_start_idx) && error("Training start date not found.")

   data_train = data[train_start_idx:train_end_idx]

    timestamp_train =dates[train_start_idx:train_end_idx]

    # ========================================================
    # 7.2 Test sample
    # ========================================================

    test_idx =test_start_idx:(test_start_idx + maxHorizon - 1)

    y_test_raw =data[test_idx]

    timestamp_test =dates[test_idx]


    # ========================================================
    # 7.3 Transformation
    # ========================================================

    xlog_train =log.(data_train)

    # Same centering rule for every expanding-window fit.
    #
    # If you want another rule, change ONLY this line.

    med_window = 30

    train_mean = median(xlog_train[1:med_window])

    x =xlog_train .-train_mean

    # True future observations on SAME scale as model
    y_test =log.(y_test_raw) .-train_mean


    # ========================================================
    # 7.4 Build SAR regressors
    # ========================================================

    init_y =fill(mean(x[1:50]),p_max[1])

    obs =vcat(init_y,x)

    activeLags_ar =FindActiveLagsMultiSAR(p1,s1)

    activeLags_ma =activeLags_ar

    Y, Z, T =SetupARReg_active(obs,activeLags_ar)


    # ========================================================
    # 7.5 Initial observation variance
    # ========================================================

    fitted_model = Arima(
        x[1:200],
        order = [2, 2, 5],
        seasonal = [5, 2, 5],
        include_mean = false,
        include_drift = false,
        include_constant = true
    )

    resid_variance =fitted_model[:sigma2]

    σ0 =sqrt(resid_variance)


    # ========================================================
    # 7.6 Group observations and regressors
    # ========================================================

    y_g =group_vector(Y,nPerGroup)

    Cargs =[Z[t, :] for t in 1:T]

    Cargs_g =group_vector_view(Cargs,nPerGroup)

    cache_ar =build_sarma_cache(p1,s1,activeLags_ar)

    cache_ma =build_sarma_cache(p2,s2,activeLags_ma)

    # ========================================================
    # 7.7 Priors
    # ========================================================

    alpha_sigma = 0.001
    beta_sigma  = 0.001

    alpha_sigma_hat =alpha_sigma +T / 2
    var_mat =fill( 0.3^2,nLags)

    if INTERCEPT
        var_mat[1] =1.0^2
        if intercept_dynamics == :ll
            var_mat[2] = 0.005^2
        end
    end


    Σ₀ =PDMat(Diagonal( var_mat))

    μ₀ =zeros(nLags)


    priorSettings = (

        # State evolution
        ϕ₀ = 0.5,
        κ₀ = 0.3,

        m₀ =
            -15.0 +
            log(nPerGroup),

        σ₀ = 3.0,

        ν₀ = 3.0,
        ψ₀ = 1.0,

        μ₀ = μ₀,
        Σ₀ = Σ₀,

        # Observation variance
        σₑ =
            fill(
                σ0,
                T
            ),

        alpha_sigma =
            alpha_sigma,

        beta_sigma =
            beta_sigma,

        alpha_sigma_hat =
            alpha_sigma_hat,

        # Filter
        α_ukf = 1e-3,
        β_ukf = 2.0,
        κ_ukf = 0
    )


    # ========================================================
    # 7.8 Algorithm settings
    # ========================================================

    algoSettings = (

        nBurn =
            nBurn,

        nIter =
            nIter +
            nBurn,

        INTERCEPT =
            INTERCEPT,

        resid_label =
            iterated,

        method_label =
            kf_method,

        model_type =
            model_type,

        SAR_conditional =
            SAR_conditional,

        obs_var_type =
            obs_var_type,

        state_var_type =
            state_var_type,

        ma_regressor_type =
            :median_freeze,

        clipped_partials =
            clipped_partials,

        p_threshold =
            p_threshold,

        presample_AR =
            :recursive,

        presample_MA =
            :simple
    )


    # ========================================================
    # 7.9 Model settings
    # ========================================================

    modelSettings = (

        nPerGroup =
            nPerGroup,

        s1 = s1,
        p1 = p1,

        s2 = s2,
        p2 = p2,

        p_max =
            p_max,

        nLags =
            nLags,

        iterations =
            num_iters,

        Cargs =
            Cargs_g,

        Z =
            Z,

        activeLags_ma =
            activeLags_ma,

        activeLags_ar =
            activeLags_ar,

        cache_ma =
            cache_ma,

        cache_ar =
            cache_ar,

        ztrans =
            ztrans,

        updateσₙ =
            false,

        nMixComp =
            10,

        α =
            0.5,

        β =
            0.5,

        ϕ̄₀ =
            0.5,

        κ̄₀ =
            0.3,

        m̄₀ =
            -15,

        σ̄₀ =
            3,

        ν̄₀ =
            3,

        ψ̄₀ =
            1.0,

        ϕ̄ =
            0.5,

        μ̄ =
            -15.0,

        σ̄²ₙ =
            1.0,

        intercept_dynamics =
            intercept_dynamics,

        T_use =
            2 * p_max[2]
    )


    # ========================================================
    # 7.10 Gibbs estimation
    # ========================================================

    Random.seed!(
        base_seed +
        origin_id
    )

    elapsed_fit[origin_id] =
        @elapsed begin

            SAR_res =
                GibbsSamplerTVSARMA_full(
                    y_g,
                    Y,
                    priorSettings,
                    modelSettings,
                    algoSettings
                )

        end

    println(
        "Fit time: ",
        round(
            elapsed_fit[origin_id] / 60,
            digits = 2
        ),
        " min"
    )


    # ========================================================
    # 7.11 Posterior draws used in prediction
    # ========================================================

    keep =
        1:thinFactor:size(SAR_res[1], 3)


    θₜpost =
        SAR_res[1][end, :, keep]


    if state_var_type == :DSP

        Hₜpost =
            SAR_res[2][end, :, keep] .-
            log(nPerGroup) .+
            log(g_forecast)

        μpost =
            SAR_res[5][:, keep] .-
            log(nPerGroup) .+
            log(g_forecast)

        ϕpost =
            SAR_res[4][:, keep]

    else

        Hₜpost =
            (SAR_res[7][:, keep] ./
             nPerGroup) .*
            g_forecast

        μpost = nothing
        ϕpost = nothing

    end


    # ========================================================
    # 7.12 Observation-variance posterior
    # ========================================================

    σₑₜpost =
        SAR_res[3][keep]


    σ²ₙpost = nothing
    ϕ̃post   = nothing
    μ̃post   = nothing
    h̃post   = nothing
    σ̄²ₙpost = nothing


    if obs_var_type == :SV

        μ̃post =
            SAR_res[6][1, keep]

        ϕ̃post =
            SAR_res[7][1, keep]

        σ̄²ₙpost =
            SAR_res[9][1, keep]


    elseif obs_var_type == :SVDSP

        μ̃post =
            SAR_res[6][1, keep]

        ϕ̃post =
            SAR_res[7][1, keep]

        h̃post =
            SAR_res[8][end, keep]

    end


    # ========================================================
    # 7.13 Initial lag history
    # ========================================================

    zₜall_orig =
        reverse(
            x[end - p_max[1] + 1:end]
        )

    # ========================================================
    # 7.14 Forecast
    # ========================================================

    Random.seed!(
        base_seed +
        100_000 +
        origin_id
    )

    elapsed_forecast[origin_id] =
        @elapsed begin

            yPred,
            LPS,
            AE =
                PredLocalMultiSAR_SV_gr(
                    nPredPerIter,
                    y_test,
                    zₜall_orig,
                    p1,
                    season,
                    ztrans,
                    θₜpost,
                    Hₜpost,
                    ϕpost,
                    σ²ₙpost,
                    μpost,
                    σₑₜpost,
                    ϕ̃post,
                    μ̃post,
                    h̃post,
                    σ̄²ₙpost,
                    forecastHorizons;
                    state_var_type =
                        state_var_type,
                    obs_var_type =
                        obs_var_type,
                    g_fc =
                        g_forecast,
                    INTERCEPT =
                        INTERCEPT,
                    H_freeze =
                        false,
                    STATE_fixed =
                        false,
                    use_fourier =
                        false,
                    intercept_dynamics =
                        intercept_dynamics,
                    startcol =
                        startcol,
                    p_threshold =
                        p_threshold
                )

        end


    # ========================================================
    # 7.15 Store horizon-specific scores
    # ========================================================

    LPSs[origin_id, :] .= LPS
    MAEs[origin_id, :] .= AE

    println(
        "Forecast time: ",
        round(
            elapsed_forecast[origin_id] / 60,
            digits = 2
        ),
        " min"
    )

    println(
        "Mean LPS: ",
        mean(LPS)
    )

    println(
        "Mean AE: ",
        mean(AE)
    )


    # ========================================================
    # 7.16 Save this origin immediately
    #
    # Important on cluster:
    # each worker writes a different file.
    # ========================================================

    ########################################################
    # Save this origin
    ########################################################

    origin_file =
        joinpath(
            resultsFolder,
            "origin_$(lpad(origin_id, 4, '0')).jld2"
        )


    # Posterior predictive median on model scale
    medianPred =
        vec(
            median(
                yPred,
                dims = 2
            )
        )


    JLD2.save(
        origin_file,
        Dict(

            "origin_id" =>
                origin_id,

            "origin_date" =>
                origin_dates[origin_id],

            "forecastHorizons" =>
                forecastHorizons,


            # ==============================================
            # Forecasts
            # ==============================================

            # All posterior predictive draws
            # dimensions:
            # maxHorizon × number of predictive draws
            "yPred" =>
                yPred,

            # Posterior predictive median
            "medianPred" =>
                medianPred,


            # ==============================================
            # Actual observations
            # ==============================================

            # Actual electricity demand
            "y_test_raw" =>
                y_test_raw,

            # Actual observations on model scale:
            # log(y) - train_mean
            "y_test" =>
                y_test,

            # Dates/times corresponding to test observations
            "timestamp_test" =>
                timestamp_test,


            # ==============================================
            # Forecast evaluation
            # ==============================================

            "LPS" =>
                LPS,

            # This is AE for this particular origin,
            # not yet MAE across origins
            "AE" =>
                AE,


            # ==============================================
            # Transformation
            # ==============================================

            "train_mean" =>
                train_mean,


            # ==============================================
            # Timing
            # ==============================================

            "elapsed_fit" =>
                elapsed_fit[origin_id],

            "elapsed_forecast" =>
                elapsed_forecast[origin_id],


            # ==============================================
            # Model information
            # ==============================================

            "p" =>
                p,

            "season" =>
                season,

            "nPerGroup" =>
                nPerGroup,

            "g_forecast" =>
                g_forecast
        )
    )

end

############################################################
# 8. COLLECT RESULTS ACROSS ALL ORIGINS
############################################################

function collect_results(
    resultsFolder,
    nOrigins,
    forecastHorizons
)

    nH = length(forecastHorizons)

    # One row per forecast origin,
    # one column per forecast horizon
    LPSs = fill(NaN, nOrigins, nH)
    AEs  = fill(NaN, nOrigins, nH)

    elapsed_fit =
        fill(NaN, nOrigins)

    elapsed_forecast =
        fill(NaN, nOrigins)

    origin_dates_saved =
        Vector{Union{Missing,DateTime}}(
            missing,
            nOrigins
        )


    ########################################################
    # Read every origin file
    ########################################################

    completed = Int[]

    for origin_id in 1:nOrigins

        origin_file =
            joinpath(
                resultsFolder,
                "origin_$(lpad(origin_id, 4, '0')).jld2"
            )


        if !isfile(origin_file)

            @warn(
                "Missing result file",
                origin_id = origin_id,
                file = origin_file
            )

            continue
        end


        d = JLD2.load(origin_file)


        ####################################################
        # Check that horizons agree
        ####################################################

        stored_horizons =
            Int.(d["forecastHorizons"])

        stored_horizons == forecastHorizons ||
            error(
                "Forecast horizons in $origin_file do not match."
            )


        ####################################################
        # Store scores
        ####################################################

        LPSs[origin_id, :] .=
            d["LPS"]


        # If you changed the saved key to "AE"
        if haskey(d, "AE")

            AEs[origin_id, :] .=
                d["AE"]

        # Allows your older files with key "MAE"
        elseif haskey(d, "MAE")

            AEs[origin_id, :] .=
                d["MAE"]

        else

            error(
                "Neither AE nor MAE found in $origin_file"
            )

        end


        ####################################################
        # Other saved information
        ####################################################

        origin_dates_saved[origin_id] =
            d["origin_date"]

        elapsed_fit[origin_id] =
            d["elapsed_fit"]

        elapsed_forecast[origin_id] =
            d["elapsed_forecast"]


        push!(
            completed,
            origin_id
        )

    end


    ########################################################
    # Check that something was found
    ########################################################

    isempty(completed) &&
        error(
            "No completed origin files found in $resultsFolder"
        )


    println()
    println(
        "Completed origins: ",
        length(completed),
        " / ",
        nOrigins
    )

    println(
        "Completed IDs: ",
        completed
    )


    ########################################################
    # Horizon-specific summary
    #
    # Average ACROSS forecast origins
    ########################################################

    LPS_mean =
        vec(
            mean(
                LPSs[completed, :],
                dims = 1
            )
        )


    MAE_mean =
        vec(
            mean(
                AEs[completed, :],
                dims = 1
            )
        )


    RMSE =
        sqrt.(
            vec(
                mean(
                    AEs[completed, :] .^ 2,
                    dims = 1
                )
            )
        )


    ########################################################
    # One overall number for this model
    #
    # Average across BOTH origins and horizons
    ########################################################

    overall_LPS =
        mean(
            LPSs[completed, :]
        )


    overall_MAE =
        mean(
            AEs[completed, :]
        )


    overall_RMSE =
        sqrt(
            mean(
                AEs[completed, :] .^ 2
            )
        )


    ########################################################
    # Horizon-specific table
    ########################################################

    summary =
        DataFrame(
            horizon =
                forecastHorizons,

            LPS =
                LPS_mean,

            MAE =
                MAE_mean,

            RMSE =
                RMSE
        )


    println()
    println("============================================")
    println("FORECAST SUMMARY")
    println("============================================")
    println(summary)

    println()
    println("Overall mean LPS : ", overall_LPS)
    println("Overall MAE      : ", overall_MAE)
    println("Overall RMSE     : ", overall_RMSE)
    println("============================================")
    println()


    ########################################################
    # Save horizon-specific summary as CSV
    ########################################################

    CSV.write(
        joinpath(
            resultsFolder,
            "forecast_summary.csv"
        ),
        summary
    )


    ########################################################
    # Save everything
    ########################################################

        JLD2.save(
            joinpath(
                resultsFolder,
                "forecast_summary.jld2"
            ),
            Dict(

                "completed_origins" =>
                    completed,

                "origin_dates" =>
                    origin_dates_saved[completed],

                "forecastHorizons" =>
                    forecastHorizons,

                # Full origin × horizon matrices
                "LPSs" =>
                    LPSs[completed, :],

                "AEs" =>
                    AEs[completed, :],

                # Horizon-specific averages
                "LPS_mean" =>
                    LPS_mean,

                "MAE_mean" =>
                    MAE_mean,

                "RMSE" =>
                    RMSE,


                # One number for model comparison
                "overall_LPS" =>
                    overall_LPS,

                "overall_MAE" =>
                    overall_MAE,

                "overall_RMSE" =>
                    overall_RMSE,


                # Timing
                "elapsed_fit" =>
                    elapsed_fit[completed],

                "elapsed_forecast" =>
                    elapsed_forecast[completed]
            )
        )


        return (
            summary      = summary,
            LPSs         = LPSs[completed, :],
            AEs          = AEs[completed, :],
            origin_dates = origin_dates_saved[completed],
            overall_LPS  = overall_LPS,
            overall_MAE  = overall_MAE,
            overall_RMSE = overall_RMSE
        )

    end