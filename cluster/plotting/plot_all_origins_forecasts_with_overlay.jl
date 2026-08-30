using CSV
using DataFrames
using JLD2
using Statistics
using Plots
using Dates

# ============================================================
# SETTINGS
# ============================================================

save_dir =
    raw"C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST\cluster\results"

default(
    background_color = :white,
    background_color_subplot = :white,
    framestyle = :axes,
    grid = false
)

# ------------------------------------------------------------
# Models to compare
#
# Add/remove entries here only.
# ------------------------------------------------------------

model_specs = [
    (
        results_dir =
            raw"C:\Users\Anna Fagerberg\Documents\AUS_ELEC_SAR_DSP_STATIC_MODEL_SELECTION_double",
        label = "Double-seasonal TV-SAR",
        color = "#780000"
    ),

    (
        results_dir =
            raw"C:\Users\Anna Fagerberg\Documents\AUS_ELEC_SAR_DSP_STATIC_MODEL_SELECTION_single",
        label = "Single-seasonal TV-SAR",
        color = "#6C8EBF"
    ),

    # Example:
   
    (
        results_dir =
            raw"C:\Users\Anna Fagerberg\Documents\AUS_ELEC_SAR_DSP_STATIC_MODEL_SELECTION_double_large",
        label = "Double-seasonal TV-SAR (larger)",
        color = "#3A7D44"
    )
]

col_true = "#c0a34d"


# ============================================================
# HELPER 1
# Find best specification for a model class
# ============================================================

function load_model_info(spec)

    summary_file =
        joinpath(
            spec.results_dir,
            "model_summary.csv"
        )

    @assert isfile(summary_file) """
    model_summary.csv not found:

    $summary_file
    """

    summary =
        CSV.read(
            summary_file,
            DataFrame
        )

    best_idx =
        argmax(
            summary.mean_LPS
        )

    best_model =
        summary.model[best_idx]

    combined =
        JLD2.load(
            joinpath(
                spec.results_dir,
                "combined_results.jld2"
            )
        )

    results =
        combined["models"][best_model]

    completed_origins =
        Int.(
            results["completed_origins"]
        )

    println(
        spec.label,
        ": best model = ",
        best_model,
        ", mean LPS = ",
        round(
            summary.mean_LPS[best_idx],
            digits = 4
        )
    )

    return (
        results_dir =
            spec.results_dir,

        label =
            spec.label,

        color =
            spec.color,

        best_model =
            best_model,

        completed_origins =
            completed_origins
    )
end


# ============================================================
# HELPER 2
# Load one forecast origin and compute predictive summaries
# ============================================================

function load_origin_forecast(
    model,
    origin_id;
    alpha = 0.05
)

    file =
        joinpath(
            model.results_dir,
            model.best_model,
            "origin_" *
            lpad(string(origin_id), 4, '0') *
            ".jld2"
        )

    @assert isfile(file) """
    Forecast file not found:

    $file
    """

    d =
        JLD2.load(file)

    @assert haskey(d, "yPred") """
    No yPred stored in:

    $file

    Predictive draws must have been saved.
    """

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
    # Back-transform all predictive draws
    # --------------------------------------------------------

    zPred =
        (yPred_model .+ center_value) .*
        scale_factor

    yPred_raw =
        log_transform ?
        exp.(zPred) :
        zPred


    # --------------------------------------------------------
    # Predictive median and interval
    # --------------------------------------------------------

    H =
        size(yPred_raw, 1)

    medianPred =
        Vector{Float64}(undef, H)

    lowerPred =
        Vector{Float64}(undef, H)

    upperPred =
        Vector{Float64}(undef, H)

    qlo =
        alpha / 2

    qhi =
        1 - alpha / 2


    for h in 1:H

        samples =
            view(
                yPred_raw,
                h,
                :
            )

        medianPred[h] =
            median(samples)

        lowerPred[h] =
            quantile(
                samples,
                qlo
            )

        upperPred[h] =
            quantile(
                samples,
                qhi
            )
    end


    return (
        timestamp =
            timestamp_test,

        truth =
            y_test_raw,

        median =
            medianPred,

        lower =
            lowerPred,

        upper =
            upperPred
    )
end


# ============================================================
# LOAD ALL MODEL DEFINITIONS
# ============================================================

models =
    load_model_info.(
        model_specs
    )

nModels =
    length(models)

println()
println("Number of model classes: ", nModels)


# ============================================================
# FIND COMMON FORECAST ORIGINS
# ============================================================

common_origins =
    reduce(
        intersect,
        [
            m.completed_origins
            for m in models
        ]
    )

sort!(
    common_origins
)

println(
    "Common forecast origins: ",
    common_origins
)

println(
    "Number of common origins: ",
    length(common_origins)
)


# ============================================================
# CREATE ONE OVERLAY PLOT PER FORECAST ORIGIN
# ============================================================

origin_plots =
    Plots.Plot[]


for (k, origin_id) in enumerate(common_origins)

    # --------------------------------------------------------
    # Load this origin for every model
    # --------------------------------------------------------

    fc =
        [
            load_origin_forecast(
                m,
                origin_id
            )
            for m in models
        ]


    # --------------------------------------------------------
    # Check same evaluation observations
    # --------------------------------------------------------

    reference_time =
        fc[1].timestamp

    reference_truth =
        fc[1].truth


    for j in 2:nModels

        @assert fc[j].timestamp == reference_time """
        Forecast timestamps differ at origin $origin_id.
        """

        @assert isapprox(
            fc[j].truth,
            reference_truth;
            rtol = 0,
            atol = 1e-10
        ) """
        Actual observations differ at origin $origin_id.
        """
    end


    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    panel_title =
        string(
            Date(
                reference_time[1]
            )
        )


    # --------------------------------------------------------
    # Start empty plot
    # --------------------------------------------------------

    plt = plot(
        title = panel_title,
        titlefont = font(10),
        legend = k == 1 ? :topright : false,
        legendfontsize = 7,
        xlabel = "",
        ylabel = ""
    )


    # --------------------------------------------------------
    # Overlay every model
    # --------------------------------------------------------

    for j in 1:nModels

        m =
            models[j]

        f =
            fc[j]


        # Lower predictive limit
        plot!(
            plt,
            f.timestamp,
            f.lower;
            color = m.color,
            alpha = 0.65,
            lw = 1.1,
            linestyle = :dash,
            label = false
        )


        # Upper predictive limit
        plot!(
            plt,
            f.timestamp,
            f.upper;
            color = m.color,
            alpha = 0.65,
            lw = 1.1,
            linestyle = :dash,
            label = false
        )


        # Predictive median
        plot!(
            plt,
            f.timestamp,
            f.median;
            color = m.color,
            lw = 2,
            linestyle = :solid,
            label =
                k == 1 ?
                m.label :
                false
        )
    end


    # --------------------------------------------------------
    # Observed demand
    # --------------------------------------------------------

    plot!(
        plt,
        reference_time,
        reference_truth;
        color = col_true,
        lw = 2,
        linestyle = :dot,
        label =
            k == 1 ?
            "Observed" :
            false
    )


    push!(
        origin_plots,
        plt
    )

end


# ============================================================
# SPLIT AUTOMATICALLY INTO PAGES
#
# 10 origins per figure = 5 × 2
# ============================================================

plots_per_page =
    10

nPages =
    ceil(
        Int,
        length(origin_plots) /
        plots_per_page
    )

page_plots =
    Plots.Plot[]


for page in 1:nPages

    first_idx =
        (page - 1) *
        plots_per_page +
        1

    last_idx =
        min(
            page *
            plots_per_page,
            length(origin_plots)
        )


    these_plots =
        origin_plots[
            first_idx:last_idx
        ]


    # Number of rows actually needed
    nrows =
        ceil(
            Int,
            length(these_plots) / 2
        )


    p =
        plot(
            these_plots...;
            layout = (nrows, 2),
            size = (
                1200,
                280 * nrows
            ),
            margin = 3Plots.mm
        )


    push!(
        page_plots,
        p
    )


    display(p)


    outfile =
        joinpath(
            save_dir,
            "TVSAR_forecast_overlay_" *
            lpad(
                string(first_idx),
                2,
                '0'
            ) *
            "_" *
            lpad(
                string(last_idx),
                2,
                '0'
            ) *
            ".pdf"
        )


    savefig(
        p,
        outfile
    )

    println(
        "Saved: ",
        outfile
    )
end