double = JLD2.load(
    joinpath(
        comparison_dir,
        "double_seasonal_TVSAR_plot_results.jld2"
    )
)

single = JLD2.load(
    joinpath(
        comparison_dir,
        "single_seasonal_TVSAR_plot_results.jld2"
    )
)
using JLD2
using Plots

comparison_dir =
    raw"C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST\cluster\results\comparison_data"

plot_dir =
    raw"C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST\cluster\results"

double = JLD2.load(
    joinpath(
        comparison_dir,
        "double_seasonal_TVSAR_plot_results.jld2"
    )
)

single = JLD2.load(
    joinpath(
        comparison_dir,
        "single_seasonal_TVSAR_plot_results.jld2"
    )
)

# ------------------------------------------------------------
# Variables
# ------------------------------------------------------------

horizons = double["forecastHorizons"]

@assert horizons == single["forecastHorizons"]

daily_lines = collect(24:24:168)

col_double = "#780000"
col_single = "#6C8EBF"

lab_double = "Double-seasonal TV-SAR"
lab_single = "Single-seasonal TV-SAR"


# ============================================================
# LPS
# ============================================================

p_lps = plot(
    horizons,
    double["mean_LPS_h"];
    color = col_double,
    lw = 2,
    label = lab_double,
    ylabel = "LPS",
    title = "(a) LPS",
    xticks = false,
    legend = :bottomleft
)

plot!(
    p_lps,
    horizons,
    single["mean_LPS_h"];
    color = col_single,
    lw = 2,
    label = lab_single
)

vline!(
    p_lps,
    daily_lines;
    color = :gray,
    linestyle = :dash,
    alpha = 0.3,
    lw = 0.8,
    label = false
)


# ============================================================
# MAE
# ============================================================

p_mae = plot(
    horizons,
    double["MAE_h"];
    color = col_double,
    lw = 2,
    label = false,
    ylabel = "MAE",
    title = "(b) MAE",
    xticks = false,
    legend = false
)

plot!(
    p_mae,
    horizons,
    single["MAE_h"];
    color = col_single,
    lw = 2,
    label = false
)

vline!(
    p_mae,
    daily_lines;
    color = :gray,
    linestyle = :dash,
    alpha = 0.3,
    lw = 0.8,
    label = false
)


# ============================================================
# RMSE
# ============================================================

p_rmse = plot(
    horizons,
    double["RMSE_h"];
    color = col_double,
    lw = 2,
    label = false,
    xlabel = "Forecast horizon (hours)",
    ylabel = "RMSE",
    title = "(c) RMSE",
    legend = false
)

plot!(
    p_rmse,
    horizons,
    single["RMSE_h"];
    color = col_single,
    lw = 2,
    label = false
)

vline!(
    p_rmse,
    daily_lines;
    color = :gray,
    linestyle = :dash,
    alpha = 0.3,
    lw = 0.8,
    label = false
)


# ============================================================
# Combined figure
# ============================================================

p_horizon_overlay = plot(
    p_lps,
    p_mae,
    p_rmse;
    layout = (3, 1),
    link = :x,
    size = (900, 900),
    margin = 5Plots.mm
)

display(p_horizon_overlay)

savefig(
    p_horizon_overlay,
    joinpath(
        plot_dir,
        "TVSAR_single_double_horizon_performance.pdf"
    )
)

#####################################
## Performance bu hour of the day
#####################################

hours = double["hours"]

@assert hours == single["hours"]


# ------------------------------------------------------------
# LPS
# ------------------------------------------------------------

p2_lps = plot(
    hours,
    double["LPS_hour"];
    color = col_double,
    lw = 2,
    label = lab_double,
    ylabel = "LPS",
    title = "(a) LPS",
    xticks = false,
    legend = :bottomleft
)

plot!(
    p2_lps,
    hours,
    single["LPS_hour"];
    color = col_single,
    lw = 2,
    label = lab_single
)


# ------------------------------------------------------------
# MAE
# ------------------------------------------------------------

p2_mae = plot(
    hours,
    double["MAE_hour"];
    color = col_double,
    lw = 2,
    label = false,
    ylabel = "MAE",
    title = "(b) MAE",
    xticks = false,
    legend = false
)

plot!(
    p2_mae,
    hours,
    single["MAE_hour"];
    color = col_single,
    lw = 2,
    label = false
)


# ------------------------------------------------------------
# RMSE
# ------------------------------------------------------------

p2_rmse = plot(
    hours,
    double["RMSE_hour"];
    color = col_double,
    lw = 2,
    label = false,
    xlabel = "Hour of day",
    ylabel = "RMSE",
    title = "(c) RMSE",
    xticks = 0:2:23,
    legend = false
)

plot!(
    p2_rmse,
    hours,
    single["RMSE_hour"];
    color = col_single,
    lw = 2,
    label = false
)


p_hour_overlay = plot(
    p2_lps,
    p2_mae,
    p2_rmse;
    layout = (3, 1),
    link = :x,
    size = (900, 900),
    margin = 5Plots.mm
)

display(p_hour_overlay)

savefig(
    p_hour_overlay,
    joinpath(
        plot_dir,
        "TVSAR_single_double_hour_performance.pdf"
    )
)

#####################################
## Figure 3: average actual demand versus MAE by hour
######################################

# Check that both experiments used the same actual observations
println(
    "Maximum difference in mean load = ",
    maximum(
        abs.(
            double["mean_load_hour"] .-
            single["mean_load_hour"]
        )
    )
)


# ------------------------------------------------------------
# Actual demand
# ------------------------------------------------------------

p3_load = plot(
    hours,
    double["mean_load_hour"];
    color = col_load,
    lw = 2,
    label = "Actual demand",
    ylabel = "Mean demand",
    title = "(a) Average electricity demand",
    xticks = false,
    legend = :topleft
)


# ------------------------------------------------------------
# MAE
# ------------------------------------------------------------

p3_mae = plot(
    hours,
    double["MAE_hour"];
    color = col_double,
    lw = 2,
    label = lab_double,
    xlabel = "Hour of day",
    ylabel = "MAE",
    title = "(b) Forecast error",
    xticks = 0:2:23,
    legend = :topleft
)

plot!(
    p3_mae,
    hours,
    single["MAE_hour"];
    color = col_single,
    lw = 2,
    label = lab_single
)


p_load_mae_overlay = plot(
    p3_load,
    p3_mae;
    layout = (2, 1),
    link = :x,
    size = (900, 650),
    margin = 5Plots.mm
)

display(p_load_mae_overlay)

savefig(
    p_load_mae_overlay,
    joinpath(
        plot_dir,
        "TVSAR_single_double_load_MAE.pdf"
    )
)