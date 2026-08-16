############################################################
# Compare completed models locally
############################################################

if !ON_SLURM

    model_summary,
    horizon_summary,
    combined =
        collect_results(
            models,
            nOrigins,
            forecastHorizons,
            results_dir
        )

end

best_LPS_model = model_summary[1, :]

println("Best model by LPS:")
println(best_LPS_model)

ΔLPS =
    horizon_summary[
        horizon_summary.model .== "SAR111",
        :mean_LPS
    ] .-
    horizon_summary[
        horizon_summary.model .== "SAR211",
        :mean_LPS
    ]

    plot(
    forecastHorizons,
    ΔLPS,
    xlabel = "Forecast horizon",
    ylabel = "LPS(SAR111) - LPS(SAR211)",
    label = false
)

hline!([0.0], linestyle = :dash, label = false)