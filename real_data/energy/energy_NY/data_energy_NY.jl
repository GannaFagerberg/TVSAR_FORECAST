############################################################
# New York Electricity Data
############################################################

using CSV
using DataFrames
using Dates
using Statistics
using Plots


# ============================================================
# 1. Load data
# ============================================================

cd("C:/Users/Anna Fagerberg/Desktop/data")

df = CSV.read("NYelectricity.csv", DataFrame)

rename!(df, [:DATE, :energy])
df.DATE = Date.(df.DATE)
df.energy = Float64.(df.energy)

@show nrow(df)
@show first(df.DATE)#2015-07-01
@show last(df.DATE)#2024-06-30

# Expected:
# First observation: 2015-07-01
# Last observation:  2024-06-30
# Number of observations: 3288


# ============================================================
# 2. Training and test periods
# ============================================================

train_end  = Date(2022, 1, 1)
test_start = Date(2022, 1, 2)
test_end   = Date(2022, 1, 20)

df_train = df[df.DATE .<= train_end, :]

df_test = df[
    (df.DATE .>= test_start) .&
    (df.DATE .<= test_end),
    :
]


# ============================================================
# 3. Training data
# ============================================================

y_train         = df_train.energy
timestamp_train = df_train.DATE

@show nrow(df_train)
@show first(timestamp_train)
@show last(timestamp_train)
@show minimum(y_train)
@show maximum(y_train)


# ============================================================
# 4. Test data
# ============================================================

y_test         = df_test.energy
timestamp_test = df_test.DATE
ylog_test     = log.(y_test)

@show nrow(df_test)
@show first(timestamp_test)
@show last(timestamp_test)

# May 2--20 inclusive corresponds to 19 daily observations
# if all dates are present.


# ============================================================
# 5. Plot original series
# ============================================================

plot(
    timestamp_train,
    y_train;
    xlabel = "Date",
    ylabel = "Energy",
    label = false,
    title = "New York Electricity Demand"
)


# ============================================================
# 6. Log transformation and centering
# ============================================================

@assert all(y_train .> 0) "Energy must be positive before taking logs."

logy = log.(y_train)

plot(
    timestamp_train,
    logy;
    xlabel = "Date",
    ylabel = "Log energy",
    label = false,
    title = "Log Electricity Demand"
)

# Center using the median of the first 500 training observations
train_center = median(logy[1:500])

# Currently no additional scaling
scale_factor = 1.0

x_data = (logy .- train_center) ./ scale_factor

plot(
    timestamp_train,
    x_data;
    xlabel = "Date",
    ylabel = "Centered log energy",
    label = false,
    title = "Transformed Training Series"
)


# ============================================================
# 7. Plot test period
# ============================================================

plot(
    timestamp_test,
    y_test;
    xlabel = "Date",
    ylabel = "Energy",
    label = false,
    marker = :circle,
    title = "Test Period"
)