using JLD2
using DataFrames
using Dates
using Statistics
using Plots


############################################################
# 1. Locate all completed forecast-origin files
############################################################

resultsFolder =
    raw"C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST\cluster\results"


origin_files =
    sort(
        filter(
            f -> occursin(r"^origin_\d{4}\.jld2$", f),
            readdir(resultsFolder)
        )
    )

println("Number of completed origins = ", length(origin_files))

isempty(origin_files) &&
    error("No origin_XXXX.jld2 files found.")


############################################################
# 2. Dimensions from first origin
############################################################

d0 = JLD2.load(
    joinpath(resultsFolder, origin_files[1])
)

forecastHorizons =
    Int.(d0["forecastHorizons"])

nH =
    length(forecastHorizons)

nOrigins =
    length(origin_files)


############################################################
# 3. Storage
############################################################

# For interval diagnostics:
interval_width =
    fill(NaN, nOrigins, nH)

coverage95 =
    fill(false, nOrigins, nH)


# For weekday × hour diagnostics:
forecast_diag =
    DataFrame(
        origin  = Int[],
        horizon = Int[],
        datetime = DateTime[],
        weekday = Int[],
        hour     = Int[],
        AE       = Float64[],
        LPS      = Float64[]
    )


############################################################
# 4. Read every forecast origin
############################################################

for (i, file) in enumerate(origin_files)

    d =
        JLD2.load(
            joinpath(resultsFolder, file)
        )

    yPred =
        d["yPred"]

    y_test_raw =
        d["y_test_raw"]

    timestamp_test =
        d["timestamp_test"]

    train_mean =
        d["train_mean"]

    LPS =
        d["LPS"]


    # New files:
    if haskey(d, "AE")
        AE = d["AE"]

    # Compatibility with older files:
    elseif haskey(d, "MAE")
        AE = d["MAE"]

    else
        error("No AE/MAE in $file")
    end


    ########################################################
    # Predictive draws back to ORIGINAL demand scale
    #
    # model scale = log(y) - train_mean
    ########################################################

    yPred_raw =
        exp.(yPred .+ train_mean)


    ########################################################
    # 95% posterior predictive interval at each horizon
    ########################################################

    for j in 1:nH

        h =
            forecastHorizons[j]

        draws_h =
            @view yPred_raw[h, :]

        lower =
            quantile(draws_h, 0.025)

        upper =
            quantile(draws_h, 0.975)


        interval_width[i, j] =
            upper - lower


        coverage95[i, j] =
            lower <= y_test_raw[h] <= upper


        ####################################################
        # Weekday/hour information
        ####################################################

        t =
            timestamp_test[h]

        push!(
            forecast_diag,
            (
                origin   = i,
                horizon  = h,
                datetime = t,
                weekday  = dayofweek(t),
                hour     = hour(t),
                AE       = AE[j],
                LPS      = LPS[j]
            )
        )

    end

end

############################################################
# 5. Weekday × hour mean MAE
############################################################

weekday_hour =
    combine(
        groupby(
            forecast_diag,
            [:weekday, :hour]
        ),
        :AE  => mean => :MAE,
        :LPS => mean => :mean_LPS,
        nrow => :N
    )


day_labels = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]


MAE_mat =
    fill(NaN, 7, 24)

LPS_mat =
    fill(NaN, 7, 24)


for row in eachrow(weekday_hour)

    MAE_mat[
        row.weekday,
        row.hour + 1
    ] = row.MAE

    LPS_mat[
        row.weekday,
        row.hour + 1
    ] = row.mean_LPS

end

p_mae = heatmap(
    0:23,
    1:7,
    MAE_mat;

    xlabel = "Hour of day",
    ylabel = "Day of week",

    yticks = (
        1:7,
        day_labels
    ),

    title =
        "Forecast MAE by weekday and hour",

    colorbar_title =
        "MAE"
)

display(p_mae)

p_lps = heatmap(
    0:23,
    1:7,
    LPS_mat;

    xlabel = "Hour of day",
    ylabel = "Day of week",

    yticks = (
        1:7,
        day_labels
    ),

    title =
        "Mean LPS by weekday and hour",

    colorbar_title =
        "Mean LPS"
)

display(p_lps)

############################################################
# 6. Mean predictive-interval width by horizon
############################################################

mean_interval_width =
    vec(
        mean(
            interval_width,
            dims = 1
        )
    )


p_width = plot(
    forecastHorizons,
    mean_interval_width;

    xlabel =
        "Forecast horizon (hours)",

    ylabel =
        "Mean 95% predictive interval width",

    title =
        "Predictive uncertainty versus forecast horizon",

    linewidth = 2,

    legend = false
)

display(p_width)

############################################################
# 7. Empirical predictive coverage by horizon
############################################################

coverage_by_horizon =
    vec(
        mean(
            coverage95,
            dims = 1
        )
    )


p_cov = plot(
    forecastHorizons,
    coverage_by_horizon;

    xlabel =
        "Forecast horizon (hours)",

    ylabel =
        "Empirical coverage",

    title =
        "Empirical coverage of 95% predictive intervals",

    ylim =
        (0.0, 1.0),

    linewidth =
        2,

    label =
        "Empirical coverage"
)


hline!(
    p_cov,
    [0.95];

    linestyle = :dash,

    label =
        "Nominal 95%"
)

display(p_cov)