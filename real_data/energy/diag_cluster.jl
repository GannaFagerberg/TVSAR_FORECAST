using CSV
using DataFrames

model_summary = CSV.read(
    raw"C:\Users\Anna Fagerberg\Documents\AUS_ELEC_SAR_DSP_STATIC_MODEL_SELECTION\model_summary.csv",
    DataFrame
)

display(model_summary)


using CSV
using DataFrames
using JLD2
using Statistics
using Plots


############################################################
# 1. Load model summary
############################################################

results_dir =
    raw"C:\Users\Anna Fagerberg\Documents\AUS_ELEC_SAR_DSP_STATIC_MODEL_SELECTION"


model_summary = CSV.read(
    joinpath(results_dir, "model_summary.csv"),
    DataFrame
)


############################################################
# 2. Find best model by mean LPS
############################################################

best_idx =
    argmax(model_summary.mean_LPS)

best_model =
    model_summary.model[best_idx]

println("Best model = ", best_model)

############################################################
# 3. Load combined origin × horizon results
############################################################

combined =
    JLD2.load(
        joinpath(results_dir, "combined_results.jld2")
    )

forecastHorizons =
    combined["forecastHorizons"]

results =
    combined["models"][best_model]


LPSs =
    results["LPSs"]

AEs_raw =
    results["AEs_raw"]

completed_origins =
    results["completed_origins"]


println("size(LPSs)   = ", size(LPSs))
println("size(AEs_raw)= ", size(AEs_raw))

p_lps = plot(
    xlabel = "Forecast horizon (hours)",
    ylabel = "Log predictive score",
    title  = "$best_model: LPS by forecast origin",
    legend = :topright
)

for i in axes(LPSs, 1)

    plot!(
        p_lps,
        forecastHorizons,
        LPSs[i, :],
        alpha = 0.35,
        linewidth = 1,
        label = false
    )

end


mean_LPS_h =
    vec(mean(LPSs, dims = 1))

plot!(
    p_lps,
    forecastHorizons,
    mean_LPS_h,
    linewidth = 3,
    label = "Mean across origins"
)

display(p_lps)