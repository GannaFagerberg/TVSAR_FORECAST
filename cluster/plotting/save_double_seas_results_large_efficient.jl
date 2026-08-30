using CSV
using DataFrames
using JLD2
using Statistics
using Dates


# ============================================================
# PREPARE AND SAVE PLOT RESULTS FOR ANY TV-SAR EXPERIMENT
# ============================================================

function prepare_plot_results(
    results_dir;
    model_tag,
    comparison_dir,
    best_model = nothing
)

    mkpath(comparison_dir)


    # ========================================================
    # 1. Read model-selection summary
    # ========================================================

    summary_file =
        joinpath(
            results_dir,
            "model_summary.csv"
        )

    @assert isfile(summary_file) "Missing: $summary_file"

    model_summary =
        CSV.read(
            summary_file,
            DataFrame
        )


    # --------------------------------------------------------
    # Choose best specification according to mean LPS
    # --------------------------------------------------------

    if best_model === nothing

        best_idx =
            argmax(
                model_summary.mean_LPS
            )

        best_model =
            model_summary.model[best_idx]

    else

        best_idx =
            findfirst(
                ==(best_model),
                model_summary.model
            )

        isnothing(best_idx) &&
            error(
                "Model $best_model not found in model_summary.csv"
            )
    end


    best_LPS =
        model_summary.mean_LPS[best_idx]


    println()
    println("==============================================")
    println("MODEL RESULTS")
    println("==============================================")
    println("Tag        : ", model_tag)
    println("Best model : ", best_model)
    println("Mean LPS   : ", best_LPS)
    println("==============================================")
    println()


    # ========================================================
    # 2. Load combined results
    # ========================================================

    combined =
        JLD2.load(
            joinpath(
                results_dir,
                "combined_results.jld2"
            )
        )

    forecastHorizons =
        Int.(
            combined["forecastHorizons"]
        )

    results =
        combined["models"][best_model]

    LPSs =
        results["LPSs"]

    AEs_raw =
        results["AEs_raw"]

    completed_origins =
        Int.(
            results["completed_origins"]
        )


    @show size(LPSs)
    @show size(AEs_raw)
    @show completed_origins


    # ========================================================
    # 3. Performance by forecast horizon
    # ========================================================

    mean_LPS_h =
        vec(
            mean(
                LPSs,
                dims = 1
            )
        )

    MAE_h =
        vec(
            mean(
                AEs_raw,
                dims = 1
            )
        )

    RMSE_h =
        vec(
            sqrt.(
                mean(
                    AEs_raw .^ 2,
                    dims = 1
                )
            )
        )


    # ========================================================
    # 4. Load actual observations for the same origins
    # ========================================================

    nO =
        length(
            completed_origins
        )

    nH =
        length(
            forecastHorizons
        )

    actual_raw =
        fill(
            NaN,
            nO,
            nH
        )

    actual_time =
        Matrix{DateTime}(
            undef,
            nO,
            nH
        )


    for (ii, origin_id) in enumerate(completed_origins)

        file =
            joinpath(
                results_dir,
                best_model,
                "origin_" *
                lpad(
                    string(origin_id),
                    4,
                    '0'
                ) *
                ".jld2"
            )

        @assert isfile(file) "Missing origin file: $file"

        d =
            JLD2.load(file)

        y_origin =
            d["y_test_raw"]

        t_origin =
            d["timestamp_test"]

        actual_raw[ii, :] .=
            y_origin[forecastHorizons]

        actual_time[ii, :] .=
            t_origin[forecastHorizons]
    end


    @assert size(actual_raw) == size(AEs_raw)
    @assert size(actual_time) == size(AEs_raw)


    # ========================================================
    # 5. Performance by hour of day
    # ========================================================

    hours =
        collect(0:23)

    LPS_hour =
        zeros(24)

    MAE_hour =
        zeros(24)

    RMSE_hour =
        zeros(24)

    mean_load_hour =
        zeros(24)


    for hr in hours

        mask =
            hour.(actual_time) .== hr

        LPS_hour[hr + 1] =
            mean(
                LPSs[mask]
            )

        MAE_hour[hr + 1] =
            mean(
                AEs_raw[mask]
            )

        RMSE_hour[hr + 1] =
            sqrt(
                mean(
                    AEs_raw[mask] .^ 2
                )
            )

        mean_load_hour[hr + 1] =
            mean(
                actual_raw[mask]
            )
    end


    # ========================================================
    # 6. DataFrames used for plots
    # ========================================================

    horizon_results =
        DataFrame(
            Horizon = forecastHorizons,
            LPS     = mean_LPS_h,
            MAE     = MAE_h,
            RMSE    = RMSE_h
        )

    hour_results =
        DataFrame(
            Hour      = hours,
            LPS       = LPS_hour,
            MAE       = MAE_hour,
            RMSE      = RMSE_hour,
            Mean_load = mean_load_hour
        )


    # ========================================================
    # 7. Save CSVs
    # ========================================================

    horizon_csv =
        joinpath(
            comparison_dir,
            "$(model_tag)_horizon_results.csv"
        )

    hour_csv =
        joinpath(
            comparison_dir,
            "$(model_tag)_hour_results.csv"
        )

    CSV.write(
        horizon_csv,
        horizon_results
    )

    CSV.write(
        hour_csv,
        hour_results
    )


    # ========================================================
    # 8. Save everything together
    # ========================================================

    outfile =
        joinpath(
            comparison_dir,
            "$(model_tag)_plot_results.jld2"
        )


    JLD2.jldopen(
        outfile,
        "w"
    ) do f

        # Identification
        f["model_tag"] =
            model_tag

        f["best_model"] =
            best_model

        f["best_mean_LPS"] =
            best_LPS

        f["results_dir"] =
            results_dir


        # Evaluation setup
        f["completed_origins"] =
            completed_origins

        f["n_origins"] =
            nO

        f["forecastHorizons"] =
            forecastHorizons


        # Figure 1: performance by horizon
        f["mean_LPS_h"] =
            mean_LPS_h

        f["MAE_h"] =
            MAE_h

        f["RMSE_h"] =
            RMSE_h


        # Figure 2: performance by hour
        f["hours"] =
            hours

        f["LPS_hour"] =
            LPS_hour

        f["MAE_hour"] =
            MAE_hour

        f["RMSE_hour"] =
            RMSE_hour


        # Figure 3: actual demand
        f["mean_load_hour"] =
            mean_load_hour


        # Actual evaluation observations
        f["actual_raw"] =
            actual_raw

        f["actual_time"] =
            actual_time


        # Metadata
        f["error_scale"] =
            "original demand scale"

        f["LPS_scale"] =
            "model scale"
    end


    println("Saved:")
    println("  ", horizon_csv)
    println("  ", hour_csv)
    println("  ", outfile)


    # ========================================================
    # Return everything for immediate use
    # ========================================================

    return (
        model_summary =
            model_summary,

        best_model =
            best_model,

        best_LPS =
            best_LPS,

        completed_origins =
            completed_origins,

        forecastHorizons =
            forecastHorizons,

        LPSs =
            LPSs,

        AEs_raw =
            AEs_raw,

        mean_LPS_h =
            mean_LPS_h,

        MAE_h =
            MAE_h,

        RMSE_h =
            RMSE_h,

        hours =
            hours,

        LPS_hour =
            LPS_hour,

        MAE_hour =
            MAE_hour,

        RMSE_hour =
            RMSE_hour,

        mean_load_hour =
            mean_load_hour,

        actual_raw =
            actual_raw,

        actual_time =
            actual_time
    )
end


comparison_dir =
    raw"C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST\cluster\results\comparison_data"

double_large_results_dir =
    raw"C:\Users\Anna Fagerberg\Documents\AUS_ELEC_SAR_DSP_STATIC_MODEL_SELECTION_double_large"


double_large = prepare_plot_results(
    double_large_results_dir;
    model_tag = "double_large_TVSAR",
    comparison_dir = comparison_dir
)