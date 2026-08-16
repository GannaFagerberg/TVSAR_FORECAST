### MAE and RMSE over time
#How does performance change from 1 hour ahead to 168 hours ahead?

plot(
    summary.horizon,
    summary.LPS,
    xlabel = "Forecast horizon",
    ylabel = "Mean LPS",
    legend = false
)

plot(
    summary.horizon,
    summary.MAE,
    xlabel = "Forecast horizon",
    ylabel = "Mean MAE",
    legend = false
)

plot(
    summary.horizon,
    summary.RMSE,
    xlabel = "Forecast horizon",
    ylabel = "RMSE",
    legend = false
)

### By forecast origin / time
### 24 hours aheadh_idx = findfirst(==(24), forecastHorizons)
### How did the model perform over time specifically for the 24-hour-ahead forecast?

h_idx = findfirst(==(24), forecastHorizons)

plot(
    origin_dates[ids],
    LPSs[ids, h_idx],
    xlabel = "Forecast origin",
    ylabel = "LPS at horizon 24",
    legend = false
)

plot(
    origin_dates[ids],
    MAEs[ids, h_idx],
    xlabel = "Forecast origin",
    ylabel = "AE at horizon 24",
    legend = false
)


### Average over horizons for each origin:
## How good was the entire 168-hour forecast path issued at this origin, on average?

mean_LPS_by_origin = vec(mean(LPSs[ids, :], dims = 2))
mean_AE_by_origin  = vec(mean(MAEs[ids, :], dims = 2))

plot(
    origin_dates[ids],
    mean_LPS_by_origin,
    xlabel = "Forecast origin",
    ylabel = "Mean LPS across horizons",
    legend = false
)

plot(
    origin_dates[ids],
    mean_AE_by_origin,
    xlabel = "Forecast origin",
    ylabel = "Mean AE across horizons",
    legend = false
)
### Selecte dhorisons:
### Tells whether, say, 24-hour forecasts deteriorated during a particular period.

horizons_plot = [1, 24, 72, 168]

for h in horizons_plot
    h_idx = findfirst(==(h), forecastHorizons)

    plot!(
        origin_dates[ids],
        LPSs[ids, h_idx],
        label = "h = $h"
    )
end

#the whole one-week forecast distribution deteriorated at that origin
mean_LPS_by_origin =vec(mean(LPSs[ids, :], dims = 2))

# Mean acroos all horisons
mean_LPS_by_horizon = vec(mean(LPSs, dims=1))

overall_LPS = mean(LPSs)
overall_MAE  = mean(MAEs)
overall_RMSE = sqrt(mean(MAEs.^2))