using JLD2
using Plots
using Statistics

comparison_dir =
    raw"C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST\cluster\results\comparison_data"

plot_dir =
    raw"C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST\cluster\results"


# ============================================================
# MODELS TO COMPARE
#
# To add another model, just add another entry here.
# ============================================================

model_specs = [
    (
        file  = "double_seasonal_TVSAR_plot_results.jld2",
        label = "Double-seasonal TV-SAR",
        color = "#780000"
    ),
    (
        file  = "single_seasonal_TVSAR_plot_results.jld2",
        label = "Single-seasonal TV-SAR",
        color = "#6C8EBF"
    ),
  
    (
        file  = "double_large_TVSAR_plot_results.jld2",
        label = "Double-seasonal (large) TV-SAR",
        color = "#3A7D44"
    )
]


# ============================================================
# LOAD EVERYTHING
# ============================================================

models = [
    (
        data  = JLD2.load(joinpath(comparison_dir, m.file)),
        label = m.label,
        color = m.color
    )
    for m in model_specs
]

println("Loaded $(length(models)) models")

for m in models
    println(
        m.label,
        "   [",
        m.data["best_model"],
        "]"
    )
end

### General plotting function
function overlay_metric(
    models,
    xkey,
    ykey;
    xlabel = "",
    ylabel = "",
    title = "",
    xticks = :auto,
    legend = false
)

    x = models[1].data[xkey]

    plt = plot(
        xlabel = xlabel,
        ylabel = ylabel,
        title = title,
        xticks = xticks,
        legend = legend
    )

    for m in models

        @assert m.data[xkey] == x

        plot!(
            plt,
            x,
            m.data[ykey];
            color = m.color,
            lw = 2,
            label = m.label
        )
    end

    return plt
end

# ============================================================
# FIGURE 1
# Performance over forecast horizon
# ============================================================

horizons = models[1].data["forecastHorizons"]

daily_lines = 24:24:maximum(horizons)


p1_lps = overlay_metric(
    models,
    "forecastHorizons",
    "mean_LPS_h";
    ylabel = "LPS",
    title = "(a) LPS",
    xticks = false,
    legend = :bottomleft
)

p1_mae = overlay_metric(
    models,
    "forecastHorizons",
    "MAE_h";
    ylabel = "MAE",
    title = "(b) MAE",
    xticks = false,
    legend = false
)

p1_rmse = overlay_metric(
    models,
    "forecastHorizons",
    "RMSE_h";
    xlabel = "Forecast horizon (hours)",
    ylabel = "RMSE",
    title = "(c) RMSE",
    legend = false
)


# Daily reference lines
for plt in (p1_lps, p1_mae, p1_rmse)

    vline!(
        plt,
        daily_lines;
        color = :gray,
        linestyle = :dash,
        alpha = 0.3,
        lw = 0.8,
        label = false
    )
end


p_horizon_overlay = plot(
    p1_lps,
    p1_mae,
    p1_rmse;
    layout = (3,1),
    link = :x,
    size = (900,900),
    margin = 5Plots.mm
)

display(p_horizon_overlay)

savefig(
    p_horizon_overlay,
    joinpath(
        plot_dir,
        "TVSAR_models_horizon_performance.pdf"
    )
)


# ============================================================
# FIGURE 2
# Performance by hour of day
# ============================================================

p2_lps = overlay_metric(
    models,
    "hours",
    "LPS_hour";
    ylabel = "LPS",
    title = "(a) LPS",
    xticks = false,
    legend = :bottomleft
)

p2_mae = overlay_metric(
    models,
    "hours",
    "MAE_hour";
    ylabel = "MAE",
    title = "(b) MAE",
    xticks = false,
    legend = false
)

p2_rmse = overlay_metric(
    models,
    "hours",
    "RMSE_hour";
    xlabel = "Hour of day",
    ylabel = "RMSE",
    title = "(c) RMSE",
    xticks = 0:2:23,
    legend = false
)


p_hour_overlay = plot(
    p2_lps,
    p2_mae,
    p2_rmse;
    layout = (3,1),
    link = :x,
    size = (900,900),
    margin = 5Plots.mm
)

display(p_hour_overlay)

savefig(
    p_hour_overlay,
    joinpath(
        plot_dir,
        "TVSAR_models_hour_performance.pdf"
    )
)

# ============================================================
# Verify same evaluation data
# ============================================================

reference_load =
    models[1].data["mean_load_hour"]

for m in models[2:end]

    maxdiff = maximum(
        abs.(
            reference_load .-
            m.data["mean_load_hour"]
        )
    )

    println(
        "Maximum load difference: ",
        m.label,
        " = ",
        maxdiff
    )

    @assert maxdiff < 1e-8
end

# ============================================================
# FIGURE 3
# Actual demand versus MAE
# ============================================================

hours =
    models[1].data["hours"]

col_load =
    "#c0a34d"


# Actual demand
p3_load = plot(
    hours,
    reference_load;
    color = col_load,
    lw = 2,
    label = "Actual demand",
    ylabel = "Mean demand",
    title = "(a) Average electricity demand",
    xticks = false,
    legend = :topleft
)


# MAE of every model
p3_mae = overlay_metric(
    models,
    "hours",
    "MAE_hour";
    xlabel = "Hour of day",
    ylabel = "MAE",
    title = "(b) Forecast error",
    xticks = 0:2:23,
    legend = :topleft
)


p_load_mae_overlay = plot(
    p3_load,
    p3_mae;
    layout = (2,1),
    link = :x,
    size = (900,650),
    margin = 5Plots.mm
)

display(p_load_mae_overlay)

savefig(
    p_load_mae_overlay,
    joinpath(
        plot_dir,
        "TVSAR_models_load_MAE.pdf"
    )
)