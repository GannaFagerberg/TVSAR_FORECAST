using JLD2
using Statistics
using StatsBase
using Plots
using Dates

# ============================================================
# DOUBLE-SEASONAL TV-SAR
# ============================================================

double_results_dir = raw"C:\Users\Anna Fagerberg\Documents\AUS_ELEC_SAR_DSP_STATIC_MODEL_SELECTION_double"

# These should already have been defined when loading
# the double-seasonal results:
#
# best_model_double
# completed_origins_double


default(
    background_color = :white,
    background_color_subplot = :white,
    framestyle = :axes,
    grid = false
)

# Same colors as summarize_and_plot_t
col_true   = "#c0a34d"
col_median = "#780000"
col_band   = "#6C8EBF"


# ============================================================
# Create forecast plot for every origin
# ============================================================

origin_plots_double = Plots.Plot[]


for (k, origin_id) in enumerate(completed_origins_double)

    file = joinpath(
        double_results_dir,
        best_model_double,
        "origin_" *
        lpad(string(origin_id), 4, '0') *
        ".jld2"
    )

    @assert isfile(file) "File not found: $file"

    d = JLD2.load(file)


    # --------------------------------------------------------
    # Saved forecasts and truth
    # --------------------------------------------------------

    yPred_model =
        d["yPred"]

    y_test_raw =
        d["y_test_raw"]

    timestamp_test =
        d["timestamp_test"]

    center_value =
        d["center_value"]

    scale_factor =
        d["scale_factor"]

    log_transform =
        d["log_transform"]


    # --------------------------------------------------------
    # Back-transform predictive draws to original demand scale
    # --------------------------------------------------------

    zPred =
        (yPred_model .+ center_value) .* scale_factor

    yPred_raw =
        log_transform ?
        exp.(zPred) :
        zPred


    H =
        size(yPred_raw, 1)


    # --------------------------------------------------------
    # Predictive summaries
    # --------------------------------------------------------

    medianPred =
        zeros(H)

    lowerPred =
        zeros(H)

    upperPred =
        zeros(H)


    for h in 1:H

        samples =
            view(yPred_raw, h, :)

        medianPred[h] =
            median(samples)

        lowerPred[h] =
            quantile(samples, 0.025)

        upperPred[h] =
            quantile(samples, 0.975)

    end


    # --------------------------------------------------------
    # Panel title
    # --------------------------------------------------------

    panel_title =
        string(Date(timestamp_test[1]))


    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt = plot(
        timestamp_test,
        medianPred;
        color = col_median,
        lw = 2,
        label = k == 1 ? "Median" : "",
        title = panel_title,
        titlefont = font(10),
        legend = (k == 1),
        legendfontsize = 7
    )


    # Lower predictive interval
    plot!(
        plt,
        timestamp_test,
        lowerPred;
        color = col_band,
        lw = 1.5,
        label = k == 1 ? "95% interval" : ""
    )


    # Upper predictive interval
    plot!(
        plt,
        timestamp_test,
        upperPred;
        color = col_band,
        lw = 1.5,
        label = ""
    )


    # Actual observations
    plot!(
        plt,
        timestamp_test,
        y_test_raw;
        color = col_true,
        lw = 2,
        linestyle = :dot,
        label = k == 1 ? "Observed" : ""
    )


    push!(
        origin_plots_double,
        plt
    )

end


# ============================================================
# Three figures: 10 forecast origins each
# ============================================================

p1_double = plot(
    origin_plots_double[1:10]...;
    layout = (5, 2),
    size = (1200, 1400),
    margin = 3Plots.mm
)

p2_double = plot(
    origin_plots_double[11:20]...;
    layout = (5, 2),
    size = (1200, 1400),
    margin = 3Plots.mm
)

p3_double = plot(
    origin_plots_double[21:30]...;
    layout = (5, 2),
    size = (1200, 1400),
    margin = 3Plots.mm
)


display(p1_double)
display(p2_double)
display(p3_double)


# ============================================================
# Save
# ============================================================

save_dir =
    raw"C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST\cluster\results"


savefig(
    p1_double,
    joinpath(
        save_dir,
        "$(best_model_double)_double_origins_01_10.pdf"
    )
)

savefig(
    p2_double,
    joinpath(
        save_dir,
        "$(best_model_double)_double_origins_11_20.pdf"
    )
)

savefig(
    p3_double,
    joinpath(
        save_dir,
        "$(best_model_double)_double_origins_21_30.pdf"
    )
)