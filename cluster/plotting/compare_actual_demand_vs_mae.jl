using JLD2
using Dates
using Statistics
using Plots

# ============================================================
# Load actual observations from the SAME forecast-origin files
# used to construct AEs_raw and LPSs
# ============================================================

nO = length(completed_origins)
nH = length(forecastHorizons)

actual_raw  = fill(NaN, nO, nH)
actual_time = Matrix{DateTime}(undef, nO, nH)

for (ii, origin_id) in enumerate(completed_origins)

    file = joinpath(
        results_dir,
        best_model,
        "origin_" * lpad(string(origin_id), 4, '0') * ".jld2"
    )

    d = JLD2.load(file)

    y_origin = d["y_test_raw"]
    t_origin = d["timestamp_test"]

    actual_raw[ii, :]  .= y_origin[forecastHorizons]
    actual_time[ii, :] .= t_origin[forecastHorizons]
end

@show size(actual_raw)
@show size(actual_time)
@show size(AEs_raw)
@show size(LPSs)

hours = 0:23

mean_load_hour = zeros(24)
MAE_hour       = zeros(24)
RMSE_hour      = zeros(24)
LPS_hour       = zeros(24)

for hr in hours

    mask = hour.(actual_time) .== hr

    mean_load_hour[hr + 1] =
        mean(actual_raw[mask])

    MAE_hour[hr + 1] =
        mean(AEs_raw[mask])

    RMSE_hour[hr + 1] =
        sqrt(mean(AEs_raw[mask] .^ 2))

    LPS_hour[hr + 1] =
        mean(LPSs[mask])
end

p_load = plot(
    hours,
    mean_load_hour;
    ylabel = "Mean electricity demand",
    linewidth = 2,
    marker = :circle,
    label = false,
    xticks = false
)

p_mae = plot(
    hours,
    MAE_hour;
    xlabel = "Hour of day",
    ylabel = "MAE",
    linewidth = 2,
    marker = :circle,
    label = false,
    xticks = 0:2:23
)

p_compare = plot(
    p_load,
    p_mae;
    layout = (2, 1),
    link = :x,
    size = (800, 700)
)

display(p_compare)


outfile_pdf = joinpath(
    raw"C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST\cluster\results",
    "$(best_model)_hour_day_actual_mae.pdf"
)

savefig(p_compare, outfile_pdf)
