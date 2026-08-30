using CSV
using DataFrames
using JLD2
using Statistics
using Dates

# ============================================================
# DOUBLE-SEASONAL TV-SAR
# ============================================================

double_results_dir = raw"C:\Users\Anna Fagerberg\Documents\AUS_ELEC_SAR_DSP_STATIC_MODEL_SELECTION_double"


# ============================================================
# 1. Find best double-seasonal model
# ============================================================

model_summary_double = CSV.read(
    joinpath(double_results_dir, "model_summary.csv"),
    DataFrame
)

display(model_summary_double)

best_idx_double =
    argmax(model_summary_double.mean_LPS)

best_model_double =
    model_summary_double.model[best_idx_double]

println("Best double-seasonal model: ", best_model_double)
println(
    "Mean LPS: ",
    model_summary_double.mean_LPS[best_idx_double]
)


# ============================================================
# 2. Load combined results for best model
# ============================================================

combined_double = JLD2.load(
    joinpath(
        double_results_dir,
        "combined_results.jld2"
    )
)

forecastHorizons_double =
    combined_double["forecastHorizons"]

results_double =
    combined_double["models"][best_model_double]

LPSs_double =
    results_double["LPSs"]

AEs_raw_double =
    results_double["AEs_raw"]

completed_origins_double =
    results_double["completed_origins"]

@show size(LPSs_double)
@show size(AEs_raw_double)
@show completed_origins_double


# ============================================================
# 3. Performance by forecast horizon
# ============================================================

mean_LPS_h_double =
    vec(
        mean(
            LPSs_double,
            dims = 1
        )
    )

MAE_h_double =
    vec(
        mean(
            AEs_raw_double,
            dims = 1
        )
    )

RMSE_h_double =
    vec(
        sqrt.(
            mean(
                AEs_raw_double .^ 2,
                dims = 1
            )
        )
    )


# ============================================================
# 4. Actual observations/timestamps for the same 30 origins
# ============================================================

nO_double =
    length(completed_origins_double)

nH_double =
    length(forecastHorizons_double)

actual_raw_double =
    fill(
        NaN,
        nO_double,
        nH_double
    )

actual_time_double =
    Matrix{DateTime}(
        undef,
        nO_double,
        nH_double
    )


for (ii, origin_id) in enumerate(completed_origins_double)

    file = joinpath(
        double_results_dir,
        best_model_double,
        "origin_" *
        lpad(string(origin_id), 4, '0') *
        ".jld2"
    )

    d = JLD2.load(file)

    y_origin =
        d["y_test_raw"]

    t_origin =
        d["timestamp_test"]

    actual_raw_double[ii, :] .=
        y_origin[forecastHorizons_double]

    actual_time_double[ii, :] .=
        t_origin[forecastHorizons_double]
end


@show size(actual_raw_double)
@show size(actual_time_double)


# ============================================================
# 5. Performance by hour of day
# ============================================================

hours = collect(0:23)

LPS_hour_double       = zeros(24)
MAE_hour_double       = zeros(24)
RMSE_hour_double      = zeros(24)
mean_load_hour_double = zeros(24)


for hr in hours

    mask =
        hour.(actual_time_double) .== hr

    LPS_hour_double[hr + 1] =
        mean(
            LPSs_double[mask]
        )

    MAE_hour_double[hr + 1] =
        mean(
            AEs_raw_double[mask]
        )

    RMSE_hour_double[hr + 1] =
        sqrt(
            mean(
                AEs_raw_double[mask] .^ 2
            )
        )

    mean_load_hour_double[hr + 1] =
        mean(
            actual_raw_double[mask]
        )
end


hour_summary_double = DataFrame(
    Hour      = hours,
    LPS       = LPS_hour_double,
    MAE       = MAE_hour_double,
    RMSE      = RMSE_hour_double,
    Mean_load = mean_load_hour_double
)

display(hour_summary_double)


# ============================================================
# 6. Save results for later overlay
# ============================================================

comparison_dir =
    raw"C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST\cluster\results\comparison_data"

mkpath(comparison_dir)

model_tag = "double_seasonal_TVSAR"


horizon_results_double = DataFrame(
    Horizon = forecastHorizons_double,
    LPS     = mean_LPS_h_double,
    MAE     = MAE_h_double,
    RMSE    = RMSE_h_double
)

CSV.write(
    joinpath(
        comparison_dir,
        "$(model_tag)_horizon_results.csv"
    ),
    horizon_results_double
)

CSV.write(
    joinpath(
        comparison_dir,
        "$(model_tag)_hour_results.csv"
    ),
    hour_summary_double
)


outfile_double = joinpath(
    comparison_dir,
    "$(model_tag)_plot_results.jld2"
)

JLD2.jldopen(
    outfile_double,
    "w"
) do f

    # Model information
    f["model_tag"] =
        model_tag

    f["best_model"] =
        best_model_double


    # Figure 1
    f["forecastHorizons"] =
        forecastHorizons_double

    f["mean_LPS_h"] =
        mean_LPS_h_double

    f["MAE_h"] =
        MAE_h_double

    f["RMSE_h"] =
        RMSE_h_double


    # Figure 2
    f["hours"] =
        hours

    f["LPS_hour"] =
        LPS_hour_double

    f["MAE_hour"] =
        MAE_hour_double

    f["RMSE_hour"] =
        RMSE_hour_double


    # Figure 3
    f["mean_load_hour"] =
        mean_load_hour_double


    # Metadata
    f["n_origins"] =
        size(LPSs_double, 1)

    f["error_scale"] =
        "original demand scale"

    f["LPS_scale"] =
        "model scale"
end


println("Saved double-seasonal results:")
println(outfile_double)

@assert forecastHorizons_single == forecastHorizons_double
@assert completed_origins_single == completed_origins_double

maximum(
    abs.(
        mean_load_hour_single .-
        mean_load_hour_double
    )
)