############################################################
# Australian Electricity Demand
# SAR Model Selection by Out-of-Sample Forecasting
#
# One job = one (model, forecast origin)
#julia choose_model_LPS.jl
############################################################

#cd "C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST\cluster\tetralith"
############################################################
# 0. PROJECT / ENVIRONMENT
############################################################

using Pkg

# Are we running under SLURM?
ON_SLURM = haskey(ENV, "SLURM_JOB_ID")

# Specifically an array task?
ON_SLURM_ARRAY =
    haskey(ENV, "SLURM_ARRAY_TASK_ID")


# Script is located at:
#
# TVSAR_FORECAST/
#     cluster/
#         tetralith/
#             choose_model_LPS.jl
#
# Therefore "..", ".." takes us back to TVSAR_FORECAST.

PROJECT_DIR =
    abspath(
        joinpath(
            @__DIR__,
            "..",
            ".."
        )
    )

println("Script directory  : ", @__DIR__)
println("Project directory : ", PROJECT_DIR)
println("Running on SLURM  : ", ON_SLURM)

# Safety check
@assert isfile(joinpath(PROJECT_DIR, "Project.toml")) """
Project.toml not found in:

$PROJECT_DIR

Check the location of choose_model_LPS.jl.
"""

Pkg.activate(PROJECT_DIR)


############################################################
# Packages
############################################################

using TVSAR_FORECAST
using CSV
using DataFrames
using Dates
using Statistics
using Random
using LinearAlgebra
using PDMats
using JLD2

############################################################
# 1. WHAT SHOULD THE SCRIPT DO?
############################################################

# :run     = estimate + forecast
# :collect = collect already-completed result files
#
# Can also be overridden with environment variable:
# TVSAR_ACTION=collect

ACTION = Symbol(
    get(ENV, "TVSAR_ACTION", "run")
)

@assert ACTION in (:run, :collect)


############################################################
# 2. MINI OR FULL RUN
############################################################

# Local default  -> mini
# SLURM default  -> full
#
# Can be overridden:
# TVSAR_RUN_SIZE=mini
# TVSAR_RUN_SIZE=full

RUN_SIZE = Symbol(
    get(
        ENV,
        "TVSAR_RUN_SIZE",
        ON_SLURM ? "full" : "mini"
    )
)

@assert RUN_SIZE in (:mini, :full)


if RUN_SIZE == :mini

    nBurn        = 500
    nIter        = 500
    nPredPerIter = 10
    thinFactor   = 1

else

    nBurn        = 5000
    nIter        = 3000
    nPredPerIter = 10

    # Forecast only every kth retained MCMC draw.
    # Set = 1 if you want every draw.
    thinFactor = 10

end


############################################################
# 3. CANDIDATE SAR MODELS
############################################################

# Orders:
#
#   p_regular = ordinary AR order
#   p_daily   = AR order at s = 24
#   p_weekly  = AR order at s = 168
#
# Current choice 1:2 gives:
#
# SAR111
# SAR112
# SAR121
# SAR122
# SAR211
# SAR212
# SAR221
# SAR222
#
# Change 1:2 -> 1:3 if later you want all 27 combinations.

regular_orders = 1:2
daily_orders   = 1:2
weekly_orders  = 1:2

models = [
    (
        name   = "SAR$(p)$(P24)$(P168)",
        p      = [p, P24, P168],
        season = [1, 24, 168]
    )
    for p    in regular_orders
    for P24  in daily_orders
    for P168 in weekly_orders
]

nModels = length(models)


############################################################
# Alternative: explicitly choose only certain models
############################################################

#=
candidate_orders = [
    [1, 1, 1],
    [2, 1, 1],
    [1, 2, 1],
    [1, 1, 2],
    [2, 2, 2]
]

models = [
    (
        name   = "SAR$(p[1])$(p[2])$(p[3])",
        p      = p,
        season = [1, 24, 168]
    )
    for p in candidate_orders
]

nModels = length(models)
=#


############################################################
# 4. FORECAST EXPERIMENT
############################################################

# ----------------------------------------------------------
# Horizons
# ----------------------------------------------------------

# Every hour for one week:
forecastHorizons = collect(1:168)

# Other examples:
# forecastHorizons = [1, 6, 12, 24, 48, 72, 168]
# forecastHorizons = [1, 24, 168]

# Keep sorted because the forecasting function stores LPS
# in increasing horizon order.
forecastHorizons = sort(unique(forecastHorizons))
maxHorizon = maximum(forecastHorizons)


# ----------------------------------------------------------
# Number of forecast origins
# ----------------------------------------------------------

nOrigins = 30


# ----------------------------------------------------------
# First forecast origin
#
# This means the first training sample ends:
# 2005-12-14 03:00
# ----------------------------------------------------------

#TRAIN_START  = DateTime(2001, 1, 9, 2)
#FIRST_ORIGIN = DateTime(2005, 12, 14, 4)

# ----------------------------------------------------------
# First forecast origin
#
# The first forecast week begins at the onset of the main
# COVID-19 shutdown period in Australia.
#
# Training sample ends:
# 2020-03-23 03:00
#
# First forecast:
# 2020-03-23 04:00
#
# The training span is kept identical to the previous setup:
# 1800 days + 2 hours.
# ----------------------------------------------------------

TRAIN_START  = DateTime(2015, 4, 19, 2)
FIRST_ORIGIN = DateTime(2020, 3, 23, 4)

# ----------------------------------------------------------
# Distance between origins, in hourly observations
#
# 168:
#   immediately after the previous 7-day test window
#
# 720:
#   one 30-day estimation block apart
#
# With maxHorizon=168, 720 gives clearly non-overlapping
# forecast periods.
# ----------------------------------------------------------

origin_spacing = 24 * 30       # 720 hours

@assert origin_spacing >= maxHorizon



############################################################
# 5. CURRENT MODEL SETTINGS
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


# Estimation grouping
nPerGroup = 24 * 30


# Forecast state evolution
#
# 1 = evolve states hourly in forecasting
g_forecast = 1


SAR_conditional = false
use_fourier     = false
scaled          = false


############################################################
# Stability transformation
############################################################

ztrans           = "partials"
clipped_partials = true
p_threshold      = 0.99


############################################################
# Filter
############################################################

iterated = false
num_iters =iterated ? 5 : 1
kf_method = :iekf


############################################################
# 6. DATA TRANSFORMATION
############################################################

log_transform = true
scale_factor  = 1.0

# Your earlier preprocessing:
#
# train_mean = median(x_train[1:30])
#
# Other available choices below:
# :first30median
# :mean
# :median
# :none

center_mode   = :first30median
center_window = 30


############################################################
# 7. FILE LOCATIONS
############################################################

if Sys.iswindows()

    default_data_file =
        raw"C:\Users\Anna Fagerberg\Desktop\PROJECTS IN JULIA\DATA\AustralianElecPriceDemand202512_hourly.csv"

else

    # On Teralith, supply this through TVSAR_DATA_FILE
    default_data_file =
        joinpath(
            @__DIR__,
            "AustralianElecPriceDemand202512_hourly.csv"
        )

end


data_file =
    get(
        ENV,
        "TVSAR_DATA_FILE",
        default_data_file
    )


@assert isfile(data_file) """
Data file not found:

$data_file

Set TVSAR_DATA_FILE to the location of the electricity CSV.
"""


EXPERIMENT_NAME =
    "AUS_ELEC_SAR_DSP_STATIC_MODEL_SELECTION"


results_root =
    get(
        ENV,
        "TVSAR_RESULTS_DIR",
        joinpath(@__DIR__, "results")
    )


results_dir =
    joinpath(
        results_root,
        EXPERIMENT_NAME
    )


mkpath(results_dir)


#println("Data file   : ", data_file)
#println("Results dir : ", results_dir)

############################################################
# 8. OTHER EXPERIMENT SETTINGS
############################################################

base_seed = 1251222

# Very useful if a cluster job is rerun:
SKIP_EXISTING = true

# Usually false because yPred is large.
SAVE_PREDICTIVE_DRAWS = false


############################################################
# 9. HELPER: RESULT FILE NAME
############################################################

function result_filename(
    results_dir,
    model_name,
    origin_id
)

    model_dir =
        joinpath(
            results_dir,
            model_name
        )

    mkpath(model_dir)

    file_name =
        "origin_" *
        lpad(string(origin_id), 4, '0') *
        ".jld2"

    return joinpath(
        model_dir,
        file_name
    )
end


############################################################
# 10. HELPER: PREPROCESS TRAIN / TEST
############################################################

function preprocess_data(
    train_raw,
    test_raw;
    log_transform::Bool,
    scale_factor::Float64,
    center_mode::Symbol,
    center_window::Int
)

    train_trans =
        Float64.(train_raw)

    test_trans =
        Float64.(test_raw)


    if log_transform

        any(x -> x <= 0.0, train_trans) &&
            error("Non-positive value in training data.")

        any(x -> x <= 0.0, test_trans) &&
            error("Non-positive value in test data.")

        train_trans =
            log.(train_trans)

        test_trans =
            log.(test_trans)

    end


    train_trans ./=
        scale_factor

    test_trans ./=
        scale_factor


    if center_mode == :first30median

        w =
            min(
                center_window,
                length(train_trans)
            )

        center_value =
            median(
                @view train_trans[1:w]
            )


    elseif center_mode == :mean

        center_value =
            mean(train_trans)


    elseif center_mode == :median

        center_value =
            median(train_trans)


    elseif center_mode == :none

        center_value = 0.0


    else

        error(
            "Unknown center_mode = $center_mode"
        )

    end

    x_train =train_trans .-center_value
    y_test = test_trans .-center_value

    return x_train,
           y_test,
           center_value

end


############################################################
# 11. HELPER: BACK TO ORIGINAL DEMAND SCALE
############################################################

function inverse_transform(
    x,
    center_value;
    log_transform::Bool,
    scale_factor::Float64
)

    z =(x .+ center_value) .*scale_factor

    if log_transform
        return exp.(z)
    else
        return z
    end

end


############################################################
# 12. FIT ONE SAR MODEL
############################################################

function fit_one_SAR(
    x,
    model;
    nPerGroup,
    nBurn,
    nIter,
    INTERCEPT,
    intercept_dynamics,
    SAR_conditional,
    obs_var_type,
    state_var_type,
    ztrans,
    clipped_partials,
    p_threshold,
    iterated,
    num_iters,
    kf_method,
    seed
)

    ########################################################
    # Model structure
    ########################################################

    model_type = :SAR

    season = model.season

    s1 = season
    s2 = season

    p1 = copy(model.p)
    p2 = copy(model.p)
    pFit = sum(p1)
    p_max = [sum(p1 .* s1),sum(p2 .* s2)]


    nLags =
        pFit +
        (
            INTERCEPT ?
            (
                intercept_dynamics == :ll ?
                2 :
                1
            ) :
            0
        )


    ########################################################
    # Initial observation variance
    ########################################################

    length(x) < 200 &&
        error("Training sample must contain at least 200 observations.")


    fitted_model = Arima(
        x[1:200],
        order = [2, 2, 5],
        seasonal = [5, 2, 5],
        include_mean = false,
        include_drift = false,
        include_constant = true
    )


    resid_variance =
        fitted_model[:sigma2]

    σ0_obs =
        sqrt(resid_variance)


    ########################################################
    # SAR regressors
    ########################################################

    if SAR_conditional
        obs = vcat(x[1:25],x[26:end])
    else
        init_y =fill(mean(x[1:50]),p_max[1])
        obs =vcat(init_y,x)
    end


    activeLags_ar =FindActiveLagsMultiSAR(p1,s1)
    activeLags_ma =activeLags_ar
    Y, Z, T =SetupARReg_active(obs,activeLags_ar)

    ########################################################
    # Static variance prior
    ########################################################

    alpha_sigma = 0.001
    beta_sigma  = 0.001
    alpha_sigma_hat =alpha_sigma +T / 2

    ########################################################
    # Grouping
    ########################################################

    y_g =group_vector(Y,nPerGroup)
    Cargs =[Z[t, :]for t in 1:T]
    Cargs_g =group_vector_view(Cargs, nPerGroup)

    cache_ar =
        build_sarma_cache(
            p1,
            s1,
            activeLags_ar
        )


    cache_ma =
        build_sarma_cache(
            p2,
            s2,
            activeLags_ma
        )


    ########################################################
    # Initial state prior
    ########################################################

    var_mat =fill(0.3^2,nLags)

    if INTERCEPT

        var_mat[1] =1.0^2
        if intercept_dynamics == :ll
            var_mat[2] =0.005^2
        end
    end


    Σ₀ =PDMat(Diagonal(var_mat))
    μ₀ =zeros(nLags)

    ########################################################
    # Prior settings
    ########################################################

    priorSettings = (

        # State evolution
        ϕ₀ = 0.5,
        κ₀ = 0.3,

        m₀ =-15.0 +log(nPerGroup),

        σ₀ = 3.0,

        ν₀ = 3.0,
        ψ₀ = 1.0,

        μ₀ = μ₀,
        Σ₀ = Σ₀,

        # Observation noise
        σₑ =fill(σ0_obs,T),

        alpha_sigma =alpha_sigma,

        beta_sigma =beta_sigma,

        alpha_sigma_hat =alpha_sigma_hat,

        # UKF / IEKF
        α_ukf = 1e-3,
        β_ukf = 2.0,
        κ_ukf = 0
    )


    ########################################################
    # Algorithm settings
    ########################################################

    algoSettings = (

        nBurn =nBurn,

        # Total sampler iterations
        nIter =nIter +nBurn,
        INTERCEPT =INTERCEPT,
        resid_label =iterated,
        method_label =kf_method,
        model_type =model_type,
        SAR_conditional =SAR_conditional,
        obs_var_type =obs_var_type,
        state_var_type =state_var_type,
        ma_regressor_type =:median_freeze,
        clipped_partials =clipped_partials,
        p_threshold =p_threshold,
        presample_AR =:recursive,
        presample_MA =:simple
    )


    ########################################################
    # Model settings
    ########################################################

    modelSettings = (

        nPerGroup =nPerGroup,
        s1 = s1,
        p1 = p1,
        s2 = s2,
        p2 = p2,
        p_max =p_max,
        nLags =nLags,
        iterations =num_iters,
        Cargs =Cargs_g,
        Z =Z,
        activeLags_ma =activeLags_ma,
        activeLags_ar =activeLags_ar,
        cache_ma =cache_ma,
        cache_ar =cache_ar,
        ztrans =ztrans,
        updateσₙ =false,
        nMixComp =10,
        α =0.5,
        β = 0.5,

        # SV settings
        ϕ̄₀ = 0.5,
        κ̄₀ = 0.3,

        m̄₀ = -15,
        σ̄₀ = 3,

        ν̄₀ = 3,
        ψ̄₀ = 1.0,

        ϕ̄ = 0.5,
        μ̄ = -15.0,
        σ̄²ₙ = 1.0,

        intercept_dynamics =intercept_dynamics,
        T_use =2 * p_max[2]
    )


    ########################################################
    # Gibbs
    ########################################################

    Random.seed!(seed)

    SAR_res = nothing


    elapsed =
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


    return (
        SAR_res      = SAR_res,
        p1           = p1,
        season       = season,
        pFit         = pFit,
        p_max        = p_max,
        nLags        = nLags,
        activeLags   = activeLags_ar,
        T            = T,
        elapsed      = elapsed
    )

end


############################################################
# 13. FORECAST FROM ONE FIT
############################################################

function forecast_one_SAR(
    fit,
    x_train,
    y_test,
    forecastHorizons;
    nPerGroup,
    g_forecast,
    nPredPerIter,
    thinFactor,
    state_var_type,
    obs_var_type,
    INTERCEPT,
    intercept_dynamics,
    startcol,
    ztrans,
    p_threshold,
    seed
)

    SAR_res =fit.SAR_res


    ########################################################
    # MCMC draws retained for forecasting
    ########################################################

    nSaved =
        size(
            SAR_res[1],
            3
        )


    keep =
        1:thinFactor:nSaved


    ########################################################
    # State at forecast origin
    ########################################################

    θₜpost =SAR_res[1][end,:, keep]


    ########################################################
    # DSP state-innovation parameters
    ########################################################

    # Current experiment is DSP.
    @assert state_var_type == :DSP


    Hₜpost =
        SAR_res[2][
            end,
            :,
            keep
        ] .-
        log(nPerGroup) .+
        log(g_forecast)


    ϕpost =
        SAR_res[4][
            :,
            keep
        ]


    μpost =
        SAR_res[5][
            :,
            keep
        ] .-
        log(nPerGroup) .+
        log(g_forecast)


    ########################################################
    # Static observation SD
    ########################################################

    @assert obs_var_type == :static

    σₑₜpost =
        SAR_res[3][keep]


    ########################################################
    # Not used for static observation variance
    ########################################################

    σ²ₙpost = nothing

    ϕ̃post   = nothing
    μ̃post   = nothing
    h̃post   = nothing
    σ̄²ₙpost = nothing


    ########################################################
    # Lag history at forecast origin
    ########################################################

    pmax =
        fit.p_max[1]


    length(x_train) < pmax &&
        error(
            "Training sample shorter than p_max=$pmax"
        )


    zₜall_orig =
        reverse(
            x_train[end - pmax + 1:end]
        )


    ########################################################
    # Forecast
    ########################################################

    Random.seed!(seed)


    yPred = nothing
    LPS   = nothing
    AE    = nothing


    elapsed =
        @elapsed begin

            yPred,
            LPS,
            AE =
                PredLocalMultiSAR_SV_gr(
                    nPredPerIter,
                    y_test,
                    zₜall_orig,
                    fit.p1,
                    fit.season,
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


    medianPred =
        vec(
            median(
                yPred,
                dims = 2
            )
        )


    return (
        yPred       = yPred,
        medianPred  = medianPred,
        LPS         = LPS,
        AE          = AE,
        elapsed     = elapsed,
        nPosterior  = length(keep)
    )

end


############################################################
# 14. COLLECT COMPLETED RESULTS
############################################################

function collect_results(
    models,
    nOrigins,
    forecastHorizons,
    results_dir
)

    nH =
        length(
            forecastHorizons
        )


    model_summary =
        DataFrame(
            model       = String[],
            p_regular   = Int[],
            p_daily     = Int[],
            p_weekly    = Int[],
            n_origins   = Int[],
            mean_LPS    = Float64[],
            MAE_model   = Float64[],
            RMSE_model  = Float64[],
            MAE_raw     = Float64[],
            RMSE_raw    = Float64[]
        )


    horizon_summary =
        DataFrame(
            model       = String[],
            horizon     = Int[],
            n_origins   = Int[],
            mean_LPS    = Float64[],
            MAE_model   = Float64[],
            RMSE_model  = Float64[],
            MAE_raw     = Float64[],
            RMSE_raw    = Float64[]
        )


    combined =
        Dict{String,Any}()


    for model in models

        LPSs =
            fill(
                NaN,
                nOrigins,
                nH
            )

        AEs_model =
            fill(
                NaN,
                nOrigins,
                nH
            )

        AEs_raw =
            fill(
                NaN,
                nOrigins,
                nH
            )

        found =
            falses(
                nOrigins
            )


        for origin_id in 1:nOrigins

            file =
                result_filename(
                    results_dir,
                    model.name,
                    origin_id
                )


            if !isfile(file)
                continue
            end


            d =
                JLD2.load(file)


            stored_h =
                Int.(d["forecastHorizons"])


            stored_h == forecastHorizons ||
                error(
                    "Different forecast horizons in $file"
                )


            LPSs[origin_id, :] .=
                d["LPS"]

            AEs_model[origin_id, :] .=
                d["AE_model"]

            AEs_raw[origin_id, :] .=
                d["AE_raw"]

            found[origin_id] =
                true

        end


        idx = findall(found)

        
        if isempty(idx)

            @warn(
                "No results found for $(model.name)"
            )

            continue
        end


        L =
            LPSs[idx, :]

        A_model =
            AEs_model[idx, :]

        A_raw =
            AEs_raw[idx, :]


        ####################################################
        # Model-level summary
        ####################################################

        push!(
            model_summary,
            (
                model =
                    model.name,

                p_regular =
                    model.p[1],

                p_daily =
                    model.p[2],

                p_weekly =
                    model.p[3],

                n_origins =
                    length(idx),

                mean_LPS =
                    mean(L),

                MAE_model =
                    mean(A_model),

                RMSE_model =
                    sqrt(
                        mean(
                            A_model .^ 2
                        )
                    ),

                MAE_raw =
                    mean(A_raw),

                RMSE_raw =
                    sqrt(
                        mean(
                            A_raw .^ 2
                        )
                    )
            )
        )


        ####################################################
        # Horizon-level summary
        ####################################################

        for j in 1:nH

            push!(
                horizon_summary,
                (
                    model =
                        model.name,

                    horizon =
                        forecastHorizons[j],

                    n_origins =
                        length(idx),

                    mean_LPS =
                        mean(
                            L[:, j]
                        ),

                    MAE_model =
                        mean(
                            A_model[:, j]
                        ),

                    RMSE_model =
                        sqrt(
                            mean(
                                A_model[:, j] .^ 2
                            )
                        ),

                    MAE_raw =
                        mean(
                            A_raw[:, j]
                        ),

                    RMSE_raw =
                        sqrt(
                            mean(
                                A_raw[:, j] .^ 2
                            )
                        )
                )
            )

        end


        combined[model.name] =
        Dict(
            "LPSs" =>
                LPSs[idx, :],

            "AEs_model" =>
                AEs_model[idx, :],

            "AEs_raw" =>
                AEs_raw[idx, :],

            "completed_origins" =>
                idx
        )

    end
    
    ########################################################
    # Rank models: HIGHER LPS is better
    ########################################################

    if nrow(model_summary) > 0

        sort!(
            model_summary,
            :mean_LPS,
            rev = true
        )

    end


    ########################################################
    # Save summaries
    ########################################################

    CSV.write(
        joinpath(
            results_dir,
            "model_summary.csv"
        ),
        model_summary
    )


    CSV.write(
        joinpath(
            results_dir,
            "horizon_summary.csv"
        ),
        horizon_summary
    )


    combined_file =
        joinpath(
            results_dir,
            "combined_results.jld2"
        )


    JLD2.jldopen(
        combined_file,
        "w"
    ) do f

        f["models"] =
            combined

        f["forecastHorizons"] =
            forecastHorizons

        f["model_summary"] =
            model_summary

        f["horizon_summary"] =
            horizon_summary

    end


    println()
    println("==============================================")
    println("MODEL RANKING — HIGHER MEAN LPS IS BETTER")
    println("==============================================")

    show(
        model_summary,
        allrows = true,
        allcols = true
    )

    println()
    println()


    return (
        model_summary,
        horizon_summary,
        combined
    )

end


############################################################
# 15. IF ONLY COLLECTING RESULTS, STOP HERE
############################################################

if ACTION == :collect

    collect_results(
        models,
        nOrigins,
        forecastHorizons,
        results_dir
    )

    exit()

end


############################################################
# 16. READ DATA
############################################################

df =
    CSV.read(
        data_file,
        DataFrame
    )

sort!(
    df,
    :datetime
)

dates =
    df.datetime

data =
    df.demand_VIC


############################################################
# 17. FIND TRAINING START AND FIRST FORECAST ORIGIN
############################################################

train_start_idx =
    findfirst(
        ==(TRAIN_START),
        dates
    )

isnothing(train_start_idx) &&
    error(
        "TRAIN_START not found: $TRAIN_START"
    )


first_origin_idx =
    findfirst(
        ==(FIRST_ORIGIN),
        dates
    )

isnothing(first_origin_idx) &&
    error(
        "FIRST_ORIGIN not found: $FIRST_ORIGIN"
    )


############################################################
# 18. CREATE FORECAST ORIGINS
############################################################

origin_idx = [
    first_origin_idx +
    (j - 1) * origin_spacing
    for j in 1:nOrigins
]


@assert (
    origin_idx[end] +
    maxHorizon -
    1
) <= length(data) """
Not enough observations for final forecast origin.
"""


origin_dates =
    dates[
        origin_idx
    ]


############################################################
# 19. CREATE MODEL × ORIGIN JOB GRID
############################################################

jobs = [
    (
        model_id  = model_id,
        origin_id = origin_id
    )
    for model_id in eachindex(models)
    for origin_id in 1:nOrigins
]


nJobs =
    length(jobs)


println()
println("==============================================")
println("EXPERIMENT")
println("==============================================")
println("Models             : ", nModels)
println("Forecast origins   : ", nOrigins)
println("Total jobs         : ", nJobs)
println("Forecast horizons  : ", forecastHorizons)
println("Max horizon        : ", maxHorizon)
println("Origin spacing     : ", origin_spacing)
println("First origin       : ", first(origin_dates))
println("Last origin        : ", last(origin_dates))
println("Run size           : ", RUN_SIZE)
println("==============================================")
println()

############################################################
# 20. WHICH JOBS DOES THIS PROCESS RUN?
############################################################

if ON_SLURM_ARRAY

    # Teralith SLURM array:
    # one array task = one model × one forecast origin

    slurm_id =
        parse(
            Int,
            ENV["SLURM_ARRAY_TASK_ID"]
        )

    @assert 1 <= slurm_id <= nJobs

    jobs_to_run =
        [slurm_id]


elseif !isempty(ARGS)

    # Local example:
    #
    # julia choose_model_LPS.jl 17
    #
    # runs global job 17 only.

    local_job =
        parse(
            Int,
            ARGS[1]
        )

    @assert 1 <= local_job <= nJobs

    jobs_to_run =
        [local_job]


elseif RUN_SIZE == :mini

    # Local mini experiment:
    #
    # ALL candidate models
    # ×
    # first 2 forecast origins
    #
    # 12 models × 2 origins = 24 fits

    mini_model_ids =
        1:nModels

    mini_origin_ids =
        1:min(2, nOrigins)


    jobs_to_run = [
        j
        for (j, job) in enumerate(jobs)
        if (
            job.model_id in mini_model_ids &&
            job.origin_id in mini_origin_ids
        )
    ]


else

    # Full local sequential run:
    # all 120 model × origin jobs

    jobs_to_run =
        collect(
            eachindex(jobs)
        )

end


println(
    "Jobs handled by this process: ",
    jobs_to_run
)


############################################################
# 21. RUN MODEL × ORIGIN JOBS
############################################################

for job_id in jobs_to_run

    job =
        jobs[job_id]

    model_id =
        job.model_id

    origin_id =
        job.origin_id

    model =
        models[model_id]


    ########################################################
    # Output file
    ########################################################

    output_file =
        result_filename(
            results_dir,
            model.name,
            origin_id
        )


    if SKIP_EXISTING &&
       isfile(output_file)

        println()
        println(
            "Already exists, skipping: ",
            output_file
        )

        continue

    end

    ########################################################
    # Total wall-clock time for this model × origin
    ########################################################

    origin_start_time = time()


    ########################################################
    # Current forecast origin
    ########################################################

    test_start_idx =
        origin_idx[origin_id]

    train_end_idx = test_start_idx -1
    test_end_idx =test_start_idx +maxHorizon -1
    ########################################################
    # Raw train / test data
    ########################################################

    data_train =data[train_start_idx:train_end_idx]
    timestamp_train =dates[train_start_idx:train_end_idx]
    y_test_raw =data[test_start_idx:test_end_idx]
    timestamp_test =dates[test_start_idx:test_end_idx]

    ########################################################
    # Check test length
    ########################################################

    @assert length(y_test_raw) == maxHorizon

    ########################################################
    # Transform data
    ########################################################

    x,
    y_test,
    center_value =
        preprocess_data(
            data_train,
            y_test_raw;
            log_transform =
                log_transform,
            scale_factor =
                scale_factor,
            center_mode =
                center_mode,
            center_window =
                center_window
        )


    ########################################################
    # Status
    ########################################################

    println()
    println("##############################################")
    println("JOB       : ", job_id, " / ", nJobs)
    println("MODEL     : ", model.name)
    println("p         : ", model.p)
    println("ORIGIN    : ", origin_id, " / ", nOrigins)
    println("DATE      : ", origin_dates[origin_id])
    println("TRAIN END : ", timestamp_train[end])
    println("TEST END  : ", timestamp_test[end])
    println("T train   : ", length(x))
    println("##############################################")
    println()


    ########################################################
    # Seeds
    ########################################################

    fit_seed =
        base_seed +
        10_000 * model_id +
        origin_id


    forecast_seed =
        base_seed +
        1_000_000 +
        10_000 * model_id +
        origin_id


    ########################################################
    # Estimate
    ########################################################

    fit =
        fit_one_SAR(
            x,
            model;
            nPerGroup =
                nPerGroup,
            nBurn =
                nBurn,
            nIter =
                nIter,
            INTERCEPT =
                INTERCEPT,
            intercept_dynamics =
                intercept_dynamics,
            SAR_conditional =
                SAR_conditional,
            obs_var_type =
                obs_var_type,
            state_var_type =
                state_var_type,
            ztrans =
                ztrans,
            clipped_partials =
                clipped_partials,
            p_threshold =
                p_threshold,
            iterated =
                iterated,
            num_iters =
                num_iters,
            kf_method =
                kf_method,
            seed =
                fit_seed
        )


    println(
        "Estimation: ",
        round(
            fit.elapsed / 60,
            digits = 2
        ),
        " minutes"
    )


    ########################################################
    # Forecast
    ########################################################

    fc =
        forecast_one_SAR(
            fit,
            x,
            y_test,
            forecastHorizons;
            nPerGroup =
                nPerGroup,
            g_forecast =
                g_forecast,
            nPredPerIter =
                nPredPerIter,
            thinFactor =
                thinFactor,
            state_var_type =
                state_var_type,
            obs_var_type =
                obs_var_type,
            INTERCEPT =
                INTERCEPT,
            intercept_dynamics =
                intercept_dynamics,
            startcol =
                startcol,
            ztrans =
                ztrans,
            p_threshold =
                p_threshold,
            seed =
                forecast_seed
        )


    println(
        "Forecast: ",
        round(
            fc.elapsed / 60,
            digits = 2
        ),
        " minutes"
    )


    ########################################################
    # Original-scale predictive median
    ########################################################

    medianPred_raw =
        inverse_transform(
            fc.medianPred,
            center_value;
            log_transform =
                log_transform,
            scale_factor =
                scale_factor
        )


    ########################################################
    # Absolute errors on ORIGINAL demand scale
    ########################################################

    AE_raw =
        abs.(
            medianPred_raw[
                forecastHorizons
            ] .-
            y_test_raw[
                forecastHorizons
            ]
        )


    ########################################################
    # Store result
    ########################################################

    output =
        Dict{String,Any}(

            # Experiment
            "experiment" =>
                EXPERIMENT_NAME,

            "job_id" =>
                job_id,

            # Model
            "model_id" =>
                model_id,

            "model_name" =>
                model.name,

            "p" =>
                model.p,

            "season" =>
                model.season,

            # Origin
            "origin_id" =>
                origin_id,

            "origin_date" =>
                origin_dates[origin_id],

            "train_start" =>
                timestamp_train[1],

            "train_end" =>
                timestamp_train[end],

            "test_start" =>
                timestamp_test[1],

            "test_end" =>
                timestamp_test[end],

            # Forecast settings
            "forecastHorizons" =>
                forecastHorizons,

            "nPredPerIter" =>
                nPredPerIter,

            "thinFactor" =>
                thinFactor,

            "nPosteriorForecast" =>
                fc.nPosterior,

            # Scores
            "LPS" =>
                fc.LPS,

            # This is exactly the AE returned by your
            # forecasting function on MODEL scale.
            "AE_model" =>
                fc.AE,

            # Absolute error on original electricity
            # demand scale.
            "AE_raw" =>
                AE_raw,

            # Predictive medians
            "medianPred_model" =>
                fc.medianPred,

            "medianPred_raw" =>
                medianPred_raw,

            # Truth
            "y_test_raw" =>
                y_test_raw,

            "timestamp_test" =>
                timestamp_test,

            # Transformation
            "center_value" =>
                center_value,

            "center_mode" =>
                center_mode,

            "log_transform" =>
                log_transform,

            "scale_factor" =>
                scale_factor,

            # Grouping
            "nPerGroup" =>
                nPerGroup,

            "g_forecast" =>
                g_forecast,

            # Timing
            "elapsed_fit" =>
                fit.elapsed,

            "elapsed_forecast" =>
                fc.elapsed,

            # Seeds
            "fit_seed" =>
                fit_seed,

            "forecast_seed" =>
                forecast_seed
        )


    if SAVE_PREDICTIVE_DRAWS

        output["yPred"] =
            fc.yPred

    end


    tmp_file =
    output_file * ".tmp"


    JLD2.jldopen(
        tmp_file,
        "w"
    ) do f

        for (key, value) in output
            f[key] = value
        end

    end


    mv(
        tmp_file,
        output_file;
        force = true
    )

    elapsed_origin =
    time() - origin_start_time

    println(
        "Saved: ",
        output_file
    )

    println(
        "Mean LPS = ",
        mean(fc.LPS)
    )

    println(
        "Mean AE model scale = ",
        mean(fc.AE)
    )

    println(
        "Mean AE raw scale = ",
        mean(AE_raw)
    )


    println(
        "Total time for origin = ",
        round(
            elapsed_origin / 60,
            digits = 2
        ),
        " minutes"
    )

end


############################################################
# 22. END
############################################################

println()
println("Finished.")

