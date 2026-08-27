cd("C:/Users/Anna Fagerberg/Desktop/PROJECTS IN JULIA/DATA")

using CSV, DataFrames, Dates

df = CSV.read(
    "AustralianElecPriceDemand202512_hourly.csv",
    DataFrame
)

# ============================================================
# Training data
# ============================================================
df_train = select_period(
    df,
    #DateTime(2001, 1, 9, 2),
    #DateTime(2005, 12, 14, 3)

)

g = 24 * 30

df_train_full = select_period(
    df,
    DateTime(2001, 1, 9, 2),
    DateTime(2010, 12, 14, 3)
)

# Largest number of observations divisible by g
n_use = fld(nrow(df_train_full), g) * g

# Keep complete groups only
df_train = df_train_full[1:n_use, :]

@show nrow(df_train)
@show nrow(df_train) ÷ g
@show df_train.datetime[1]
@show df_train.datetime[end]

@assert nrow(df_train) % g == 0

#timestamp_train[1,:]
data_train      = df_train.demand_VIC
timestamp_train = df_train.datetime

# ============================================================
# Test data: next 7 days = 24*7 hours
# ============================================================

#df_test = select_period(
    #df,
    #DateTime(2005, 12, 14, 4),
    #DateTime(2005, 12, 21, 3)
#)

# ============================================================
# Test period: immediately after training period
# ============================================================

test_start = df_train.datetime[end] + Hour(1)
test_end   = test_start + Day(7) - Hour(1)

df_test = select_period(
    df,
    test_start,
    test_end
)

@show test_start
@show test_end
@show nrow(df_test)

@assert nrow(df_test) == 24 * 7

y_test_raw    = df_test.demand_VIC
timestamp_test = df_test.datetime

@show length(y_test_raw)       # should be 168
@show first(timestamp_test)
@show last(timestamp_test)
#the level and seasonal pattern of electricity demand changed, I would first overlay daily average demand by calendar day for each year.

using CSV, DataFrames, Dates, Statistics, Plots

xlog = log.(data_train)
train_mean = mean(xlog)
x_detrend = (xlog .- train_mean)
#length(x_detrend)/(24*30)
#43200/(24*30)
#24*30*60=43200
#length(x_detrend)-43200

y_test = log.(y_test_raw) .- train_mean
#.- train_mean

# ------------------------------------------------------------
# Select period
# ------------------------------------------------------------

df_covid = select_period(
    df,
    DateTime(2016, 1, 1),
    DateTime(2022, 12, 31, 23)
)

# ------------------------------------------------------------
# Add year and calendar date
# ------------------------------------------------------------

df_covid.year = year.(df_covid.datetime)
df_covid.date = Date.(df_covid.datetime)

# Remove February 29 so every year aligns exactly
df_plot = filter(
    row -> !(month(row.date) == 2 && day(row.date) == 29),
    df_covid
)

# ------------------------------------------------------------
# Daily average demand
# ------------------------------------------------------------

daily = combine(
    groupby(df_plot, [:year, :date]),
    :demand_VIC => mean => :demand
)

# ------------------------------------------------------------
# Common day-of-year axis
#
# Use a non-leap reference year (2001) so that
# Jan 1, Feb 1, ..., Dec 31 align across all years.
# ------------------------------------------------------------

daily.day = [
    dayofyear(Date(2001, month(d), day(d)))
    for d in daily.date
]

# ------------------------------------------------------------
# Plot yearly overlays
# ------------------------------------------------------------

p = plot(
    xlabel = "Month",
    ylabel = "Average daily demand",
    title  = "Victorian electricity demand by year",
    legend = :outerright,
    size   = (1000, 550)
)

for yr in 2016:2022

    tmp = daily[daily.year .== yr, :]

    plot!(
        p,
        tmp.day,
        tmp.demand,
        label = string(yr),
        linewidth = yr in (2020, 2021) ? 2.5 : 1.2,
        alpha = yr in (2020, 2021) ? 1.0 : 0.65
    )
end

# Month labels
month_starts = [
    dayofyear(Date(2001, m, 1))
    for m in 1:12
]

plot!(
    p,
    xticks = (
        month_starts,
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
)

display(p)

##########


using DataFrames, Statistics, Plots

# ------------------------------------------------------------
# Pre-COVID baseline: 2016-2019
# For every calendar day calculate mean, min and max
# ------------------------------------------------------------

pre = daily[in.(daily.year, Ref(2016:2019)), :]

baseline = combine(
    groupby(pre, :day),
    :demand => mean => :mean_pre,
    :demand => minimum => :min_pre,
    :demand => maximum => :max_pre
)

sort!(baseline, :day)

# ------------------------------------------------------------
# COVID years
# ------------------------------------------------------------

d2020 = daily[daily.year .== 2020, :]
d2021 = daily[daily.year .== 2021, :]

sort!(d2020, :day)
sort!(d2021, :day)

function moving_average(x, w=7)
    n = length(x)
    y = similar(x, Float64)

    for i in 1:n
        lo = max(1, i - w ÷ 2)
        hi = min(n, i + w ÷ 2)
        y[i] = mean(@view x[lo:hi])
    end

    return y
end

baseline.mean_smooth = moving_average(baseline.mean_pre, 7)
baseline.min_smooth  = moving_average(baseline.min_pre, 7)
baseline.max_smooth  = moving_average(baseline.max_pre, 7)

d2020.smooth = moving_average(d2020.demand, 7)
d2021.smooth = moving_average(d2021.demand, 7)

month_starts = [
    dayofyear(Date(2001, m, 1))
    for m in 1:12
]

month_names = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

p = plot(
    baseline.day,
    baseline.mean_smooth,

    ribbon = (
        baseline.mean_smooth .- baseline.min_smooth,
        baseline.max_smooth .- baseline.mean_smooth
    ),

    label = "2016-2019 range",
    linewidth = 2,
    alpha = 0.35,

    xlabel = "Month",
    ylabel = "Average daily demand",
    title = "Victorian electricity demand: COVID vs pre-COVID",
    xticks = (month_starts, month_names),
    size = (1000, 550)
)

plot!(
    p,
    d2020.day,
    d2020.smooth,
    label = "2020",
    linewidth = 3
)

plot!(
    p,
    d2021.day,
    d2021.smooth,
    label = "2021",
    linewidth = 3
)

display(p)

d2020.diff_pct =
    100 .* (d2020.smooth .- baseline.mean_smooth) ./ baseline.mean_smooth

d2021.diff_pct =
    100 .* (d2021.smooth .- baseline.mean_smooth) ./ baseline.mean_smooth

p2 = plot(
    d2020.day,
    d2020.diff_pct,
    label = "2020",
    linewidth = 3,

    xlabel = "Month",
    ylabel = "Difference from 2016-2019 mean (%)",
    title = "Deviation from pre-COVID electricity demand",
    xticks = (month_starts, month_names),
    size = (1000, 500)
)

plot!(
    p2,
    d2021.day,
    d2021.diff_pct,
    label = "2021",
    linewidth = 3
)

hline!(
    p2,
    [0],
    linestyle = :dash,
    label = "Pre-COVID level"
)

display(p2)