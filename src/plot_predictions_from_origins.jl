using JLD2
using Statistics
using Plots

############################################################
# Choose origin
############################################################

origin_id = 1

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

############################################################
# Transform predictive draws back to original demand scale
############################################################

yPred_raw =
    exp.(yPred .+ train_mean)

# 168 × 100
println(size(yPred_raw))


############################################################
# plot_state expects:
# time × variable × draws
############################################################

res_transf =
    reshape(
        yPred_raw,
        size(yPred_raw, 1),
        1,
        size(yPred_raw, 2)
    )

# 168 × 1 × 100
println(size(res_transf))

truth = y_test_raw

h_length = size(yPred_raw, 1)

plot_state(
    res_transf[1:h_length, :, :];

    prefix   = "TV-SAR(2,2,2), s = 24,168",
    ylim     = (3000, 8000),
    xlim     = (1, h_length),
    true_phi = truth[1:h_length],
    alpha    = 0.05,
    use_hdi  = true
)