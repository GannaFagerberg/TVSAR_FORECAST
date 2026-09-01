############################################################
# Australian Electricity Demand — Victoria
############################################################

using CSV
using DataFrames
using Dates
using Statistics
using Plots


# ============================================================
# 1. Load data
# ============================================================

data_dir = raw"C:\Users\Anna Fagerberg\Desktop\PROJECTS IN JULIA\DATA"

df = CSV.read(
    joinpath(data_dir, "AustralianElecPriceDemand202512_hourly.csv"),
    DataFrame
)

@show first(df)
@show last(df)
@show nrow(df)


# ============================================================
# 2. Training period
# ============================================================

# Estimation group size:
# 30 days × 24 hourly observations
nPerGroup = 24 * 30   # 720

train_start = DateTime(2015, 4, 19, 4)
train_end   = DateTime(2020, 3, 23, 4)

df_train_full = select_period(
    df,
    train_start,
    train_end
)


# ------------------------------------------------------------
# Keep complete 30-day groups only
# ------------------------------------------------------------

n_use = fld(nrow(df_train_full), nPerGroup) * nPerGroup

df_train = df_train_full[1:n_use, :]

@assert nrow(df_train) % nPerGroup == 0

@show nrow(df_train)
@show nrow(df_train) ÷ nPerGroup
@show first(df_train.datetime)
@show last(df_train.datetime)


# ------------------------------------------------------------
# Extract training series
# ------------------------------------------------------------

data_train      = Float64.(df_train.demand_VIC)
timestamp_train = df_train.datetime


# ============================================================
# 3. Save training data
# ============================================================

train_out = DataFrame(
    datetime   = timestamp_train,
    demand_VIC = data_train
)

save_file = raw"C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST\cluster\results\victoria_training_data.csv"

CSV.write(save_file, train_out)

println("Saved training data to:")
println(save_file)


# ============================================================
# 4. Test period
# ============================================================

# Seven days immediately following the final training observation
forecast_horizon = 24*7   # 168 hours

test_start = last(timestamp_train) + Hour(1)
test_end   = test_start + Hour(forecast_horizon - 1)

df_test = select_period(
    df,
    test_start,
    test_end
)

@assert nrow(df_test) == forecast_horizon

@show test_start
@show test_end
@show nrow(df_test)


# ------------------------------------------------------------
# Extract test series
# ------------------------------------------------------------

y_test_raw     = Float64.(df_test.demand_VIC)
timestamp_test = df_test.datetime

@show length(y_test_raw)
@show first(timestamp_test)
@show last(timestamp_test)


# ============================================================
# 5. Log transformation and centering
# ============================================================

@assert all(data_train .> 0)
@assert all(y_test_raw .> 0)

log_train = log.(data_train)

# Center determined entirely from training data
train_mean = mean(log_train)

x_train = log_train .- train_mean

plot(
    timestamp_train,
    x_train;
    xlabel = "Date",
    ylabel = "Centered log demand",
    label = false
)


# ============================================================
# 6. Transform test data
# ============================================================

log_test = log.(y_test_raw)

# Use exactly the same centering as for the training data
x_test = log_test .- train_mean


#plot(x_train)