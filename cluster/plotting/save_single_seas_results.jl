using CSV
using DataFrames
using JLD2
using Statistics
using Dates

# ============================================================
# SINGLE-SEASONAL TV-SAR
# ============================================================

single_results_dir = raw"C:\Users\Anna Fagerberg\Documents\AUS_ELEC_SAR_DSP_STATIC_MODEL_SELECTION_single"


# ============================================================
# 1. Find best single-seasonal model
# ============================================================

model_summary_single = CSV.read(
    joinpath(single_results_dir, "model_summary.csv"),
    DataFrame
)

display(model_summary_single)

best_idx_single =
    argmax(model_summary_single.mean_LPS)

best_model_single =
    model_summary_single.model[best_idx_single]

println("Best single-seasonal model: ", best_model_single)
println(
    "Mean LPS: ",
    model_summary_single.mean_LPS[best_idx_single]
)


# ============================================================
# 2. Load combined results for best model
# ============================================================

combined_single = JLD2.load(
    joinpath(
        single_results_dir,
        "combined_results.jld2"
    )
)

forecastHorizons_single =
    combined_single["forecastHorizons"]

results_single =
    combined_single["models"][best_model_single]

LPSs_single =
    results_single["LPSs"]

AEs_raw_single =
    results_single["AEs_raw"]

completed_origins_single =
    results_single["completed_origins"]

@show size(LPSs_single)
@show size(AEs_raw_single)
@show completed_origins_single

# ============================================================
# 3. Performance by forecast horizon
# ============================================================

mean_LPS_h_single =
    vec(
        mean(
            LPSs_single,
            dims = 1
        )
    )

MAE_h_single =
    vec(
        mean(
            AEs_raw_single,
            dims = 1
        )
    )

RMSE_h_single =
    vec(
        sqrt.(
            mean(
                AEs_raw_single .^ 2,
                dims = 1
            )
        )
    )

    # ============================================================
# 4. Actual observations/timestamps for the same 30 origins
# ============================================================

nO_single =
    length(completed_origins_single)

nH_single =
    length(forecastHorizons_single)

actual_raw_single =
    fill(
        NaN,
        nO_single,
        nH_single
    )

actual_time_single =
    Matrix{DateTime}(
        undef,
        nO_single,
        nH_single
    )


for (ii, origin_id) in enumerate(completed_origins_single)

    file = joinpath(
        single_results_dir,
        best_model_single,
        "origin_" *
        lpad(string(origin_id), 4, '0') *
        ".jld2"
    )

    d = JLD2.load(file)

    y_origin =
        d["y_test_raw"]

    t_origin =
        d["timestamp_test"]

    actual_raw_single[ii, :] .=
        y_origin[forecastHorizons_single]

    actual_time_single[ii, :] .=
        t_origin[forecastHorizons_single]
end


@show size(actual_raw_single)
@show size(actual_time_single)

# ============================================================
# 5. Performance by hour of day
# ============================================================

hours = collect(0:23)

LPS_hour_single       = zeros(24)
MAE_hour_single       = zeros(24)
RMSE_hour_single      = zeros(24)
mean_load_hour_single = zeros(24)


for hr in hours

    mask =
        hour.(actual_time_single) .== hr

    LPS_hour_single[hr + 1] =
        mean(
            LPSs_single[mask]
        )

    MAE_hour_single[hr + 1] =
        mean(
            AEs_raw_single[mask]
        )

    RMSE_hour_single[hr + 1] =
        sqrt(
            mean(
                AEs_raw_single[mask] .^ 2
            )
        )

    mean_load_hour_single[hr + 1] =
        mean(
            actual_raw_single[mask]
        )
end


hour_summary_single = DataFrame(
    Hour      = hours,
    LPS       = LPS_hour_single,
    MAE       = MAE_hour_single,
    RMSE      = RMSE_hour_single,
    Mean_load = mean_load_hour_single
)

display(hour_summary_single)

maximum(
    abs.(
        mean_load_hour_single .-
        mean_load_hour
    )
)

# ============================================================
# 6. Save results for later overlay
# ============================================================

comparison_dir =
    raw"C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST\cluster\results\comparison_data"

mkpath(comparison_dir)

model_tag = "single_seasonal_TVSAR"

horizon_results_single = DataFrame(
    Horizon = forecastHorizons_single,
    LPS     = mean_LPS_h_single,
    MAE     = MAE_h_single,
    RMSE    = RMSE_h_single
)

CSV.write(
    joinpath(
        comparison_dir,
        "$(model_tag)_horizon_results.csv"
    ),
    horizon_results_single
)

CSV.write(
    joinpath(
        comparison_dir,
        "$(model_tag)_hour_results.csv"
    ),
    hour_summary_single
)

outfile_single = joinpath(
    comparison_dir,
    "$(model_tag)_plot_results.jld2"
)

JLD2.jldopen(
    outfile_single,
    "w"
) do f

    # Model information
    f["model_tag"] =
        model_tag

    f["best_model"] =
        best_model_single


    # Figure 1
    f["forecastHorizons"] =
        forecastHorizons_single

    f["mean_LPS_h"] =
        mean_LPS_h_single

    f["MAE_h"] =
        MAE_h_single

    f["RMSE_h"] =
        RMSE_h_single


    # Figure 2
    f["hours"] =
        hours

    f["LPS_hour"] =
        LPS_hour_single

    f["MAE_hour"] =
        MAE_hour_single

    f["RMSE_hour"] =
        RMSE_hour_single


    # Figure 3
    f["mean_load_hour"] =
        mean_load_hour_single


    # Metadata
    f["n_origins"] =
        size(LPSs_single, 1)

    f["error_scale"] =
        "original demand scale"

    f["LPS_scale"] =
        "model scale"
end

println("Saved single-seasonal results:")
println(outfile_single)