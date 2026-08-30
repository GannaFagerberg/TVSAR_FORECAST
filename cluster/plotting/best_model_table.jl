using CSV
using DataFrames

model_summary = CSV.read(
    raw"C:\Users\Anna Fagerberg\Documents\AUS_ELEC_SAR_DSP_STATIC_MODEL_SELECTION_double\model_summary.csv",
    DataFrame
)

display(model_summary)

using CSV
using DataFrames
using PrettyTables
using CSV
using DataFrames
using Printf

model_summary = CSV.read(
    raw"C:\Users\Anna Fagerberg\Documents\AUS_ELEC_SAR_DSP_STATIC_MODEL_SELECTION_single\model_summary.csv",
    DataFrame
)

# Keep only columns used in the paper
report_table = select(
    model_summary,
    :model,
    :p_regular,
    #:p_daily,
    :p_weekly,
    :mean_LPS,
    :MAE_model,
    :RMSE_model,
    :MAE_raw,
    :RMSE_raw
)

# Highest LPS = best model
best_LPS = maximum(report_table.mean_LPS)

# Sort with best model first
sort!(report_table, :mean_LPS, rev = true)

# ---------------------------------------------------------
# Save clean LaTeX table
# ---------------------------------------------------------

outfile = raw"C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST\cluster\results\model_selection_table_single.tex"

open(outfile, "w") do io

    println(io, raw"\begin{table}[htbp]")
    println(io, raw"\centering")
    println(io, raw"\caption{Model selection results for the Australian electricity-demand series.}")
    println(io, raw"\label{tab:aus_model_selection}")
    println(io, raw"\begin{tabular}{lcccccccc}")
    println(io, raw"\toprule")

    println(
        io,
        raw"Model & $p$ & $P_{24}$ & $P_{168}$ & LPS & MAE (model) & RMSE (model) & MAE & RMSE \\"
    )

    println(io, raw"\midrule")

    for row in eachrow(report_table)

        # Bold highest LPS
        lps_string =
            row.mean_LPS == best_LPS ?
            "\\textbf{$(@sprintf("%.3f", row.mean_LPS))}" :
            @sprintf("%.3f", row.mean_LPS)

        println(
            io,
            "$(row.model) & " *
            "$(row.p_regular) & " *
            #"$(row.p_daily) & " *
            "$(row.p_weekly) & " *
            "$lps_string & " *
            "$(@sprintf("%.3f", row.MAE_model)) & " *
            "$(@sprintf("%.3f", row.RMSE_model)) & " *
            "$(@sprintf("%.1f", row.MAE_raw)) & " *
            "$(@sprintf("%.1f", row.RMSE_raw)) \\\\"
        )
    end

    println(io, raw"\bottomrule")
    println(io, raw"\end{tabular}")
    println(io, raw"\end{table}")
end

println("Saved table to:")
println(outfile)

####


 # ============================================================
# Performance by forecast horizon
# ============================================================

mean_LPS_h = vec(mean(LPSs, dims = 1))
MAE_h      = vec(mean(AEs_raw, dims = 1))
RMSE_h     = vec(sqrt.(mean(AEs_raw .^ 2, dims = 1)))

daily_lines = collect(24:24:168)


# ------------------------------------------------------------
# LPS
# ------------------------------------------------------------

p_lps = plot(
    forecastHorizons,
    mean_LPS_h;
    ylabel = "LPS",
    title = "(a) LPS",
    linewidth = 2,
    label = false,
    xticks = false
)

vline!(
    p_lps,
    daily_lines;
    linestyle = :dash,
    linewidth = 0.8,
    alpha = 0.35,
    label = false
)


# ------------------------------------------------------------
# MAE
# ------------------------------------------------------------

p_mae = plot(
    forecastHorizons,
    MAE_h;
    ylabel = "MAE",
    title = "(b) MAE",
    linewidth = 2,
    label = false,
    xticks = false
)

vline!(
    p_mae,
    daily_lines;
    linestyle = :dash,
    linewidth = 0.8,
    alpha = 0.35,
    label = false
)


# ------------------------------------------------------------
# RMSE
# ------------------------------------------------------------

p_rmse = plot(
    forecastHorizons,
    RMSE_h;
    xlabel = "Forecast horizon (hours)",
    ylabel = "RMSE",
    title = "(c) RMSE",
    linewidth = 2,
    label = false
)

vline!(
    p_rmse,
    daily_lines;
    linestyle = :dash,
    linewidth = 0.8,
    alpha = 0.35,
    label = false
)


# ------------------------------------------------------------
# Combined
# ------------------------------------------------------------

p_performance = plot(
    p_lps,
    p_mae,
    p_rmse;
    layout = (3, 1),
    link = :x,
    size = (900, 900),
    margin = 5Plots.mm
)

display(p_performance)

outfile_pdf = joinpath(
    raw"C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST\cluster\results",
    "$(best_model)_forecast_performance.pdf"
)

savefig(p_performance, outfile_pdf)

#####################

# At what hour of the day does TV-SAR forecast best or worst?
using DataFrames

# Horizon 1 = 04:00
forecast_start_hour = 4

hour_of_day = mod.(
    forecast_start_hour .+ forecastHorizons .- 1,
    24
)

LPS_hour  = zeros(24)
MAE_hour  = zeros(24)
RMSE_hour = zeros(24)

for hr in 0:23

    idx = findall(hour_of_day .== hr)

    LPS_hour[hr + 1] =
        mean(LPSs[:, idx])

    MAE_hour[hr + 1] =
        mean(AEs_raw[:, idx])

    RMSE_hour[hr + 1] =
        sqrt(
            mean(
                AEs_raw[:, idx] .^ 2
            )
        )
end

hour_summary = DataFrame(
    Hour = 0:23,
    LPS  = LPS_hour,
    MAE  = MAE_hour,
    RMSE = RMSE_hour
)

display(hour_summary)


p_hour_lps = plot(
    0:23,
    LPS_hour;
    ylabel = "LPS",
    title = "(a) LPS",
    linewidth = 2,
    label = false,
    xticks = false
)

p_hour_mae = plot(
    0:23,
    MAE_hour;
    ylabel = "MAE",
    title = "(b) MAE",
    linewidth = 2,
    label = false,
    xticks = false
)

p_hour_rmse = plot(
    0:23,
    RMSE_hour;
    xlabel = "Hour of day",
    ylabel = "RMSE",
    title = "(c) RMSE",
    linewidth = 2,
    label = false,
    xticks = 0:2:23
)

p_hour = plot(
    p_hour_lps,
    p_hour_mae,
    p_hour_rmse;
    layout = (3,1),
    link = :x,
    size = (900,900),
    margin = 5Plots.mm
)

display(p_hour)

outfile_pdf = joinpath(
    raw"C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST\cluster\results",
    "$(best_model)_hour_day_performance.pdf"
)

savefig(p_hour, outfile_pdf)


#ypically within the high-load part of the day, although the exact peak depends on season



#################
