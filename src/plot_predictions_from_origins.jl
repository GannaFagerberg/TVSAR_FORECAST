using JLD2
using Statistics
using Plots

############################################################
# Choose origin
############################################################

origin_id = 5

resultsFolder =
    raw"C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST\cluster\results"

origin_file =
    joinpath(
        resultsFolder,
        "origin_$(lpad(origin_id, 4, '0')).jld2"
    )


############################################################
# Load result
############################################################

d = JLD2.load(origin_file)

yPred          = d["yPred"]
y_test_raw     = d["y_test_raw"]
timestamp_test = d["timestamp_test"]
train_mean     = d["train_mean"]

println(size(yPred))

using Dates
using DataFrames

forecast_info = DataFrame(
    horizon  = 1:length(timestamp_test),
    datetime = timestamp_test,
    date     = Date.(timestamp_test),
    hour     = hour.(timestamp_test),
    weekday  = dayname.(timestamp_test)
)

println(forecast_info)

############################################################
# Transform predictive draws back to original demand scale
############################################################
using Dates
using Plots

res_transf =
    reshape(
        yPred_raw,
        size(yPred_raw, 1),
        1,
        size(yPred_raw, 2)
    )

truth = y_test_raw
h_length = size(yPred_raw, 1)


############################################################
# Original plot_state design
############################################################

plot_state(
    res_transf[1:h_length, :, :];

    prefix   = "TV-SAR(2,2,2), s = 24,168",
    ylim     = (3000, 8000),
    xlim     = (1, h_length),
    true_phi = truth[1:h_length],
    alpha    = 0.05,
    use_hdi  = true
)


############################################################
# Get the plot produced by plot_state
############################################################

p = current()


############################################################
# One tick every 24 hours
############################################################

tick_pos = collect(1:24:h_length)

tick_labels = [
    string(
        dayabbr(timestamp_test[i]),
        "\n",
        Dates.format(timestamp_test[i], "dd u")
    )
    for i in tick_pos
]


############################################################
# Change ONLY x-axis labels
############################################################

plot!(
    p;
    xticks = (tick_pos, tick_labels),
    xlabel = "Forecast date"
)

display(p)