############################################################
# Australian Electricity Demand Data
############################################################

using CSV
using DataFrames
using Statistics
using Plots


############################################################
# 1. Read Data
############################################################

cd("C:/Users/Anna Fagerberg/Desktop/PROJECTS IN JULIA/DATA")

using CSV, DataFrames, Dates

df = CSV.read(
    "AustralianElecPriceDemand202512_hourly.csv",
    DataFrame
)

df_covid = select_period(
    df,
    DateTime(2020, 3, 1),# 1, March 00:00
    DateTime(2021, 12, 31, 23) #31 December 2021 at 23:00:00
)

data      = df_covid.demand_VIC
timestamp = df_covid.datetime

df_covid_start = select_period(
    df,
    DateTime(2016, 1, 1),
    DateTime(2020, 12, 31, 23)
)

data      = df_covid_start.demand_VIC
timestamp = df_covid_start.datetime


# Make sure datetime is DateTime
#df.datetime = DateTime.(df.datetime)

# ---------------------------------------------------------
# Choose any period you want
# ---------------------------------------------------------
#start_date = DateTime(2020, 3, 1)
#end_date   = DateTime(2021, 12, 31, 23, 59, 59)

#idx = (df.datetime .>= start_date) .&(df.datetime .<= end_date)
#df_sub = df[idx, :]

# Victorian electricity demand and timestamps
#data      = df_sub.demand_VIC
#timestamp = df_sub.datetime

#println(df[1,:])#2001-01-01T01:00:00
#println(df[end,:]) # 2026-01-01T00:00:00

############################################################
# 2. Model Specification
############################################################

# Seasonal periods:
#   1   = ordinary AR component
#   24  = daily seasonality
#   168 = weekly seasonality
season = [1, 24, 168]

# One AR coefficient at each seasonal frequency
p = [1, 1, 1]

s1 = season
p1 = p

s2 = season
p2 = p

pFit = sum(p1)

# Maximum lag implied by AR/MA polynomials
p_max = [
    sum(p1 .* s1),
    sum(p2 .* s2)
]


############################################################
# 3. Preprocessing
############################################################

# ----------------------------------------------------------
# Options
# ----------------------------------------------------------

log_transform = true     # true = log(data), false = original data
scale_factor  = 1.0

if log_transform
    x = log.(data)
else
    x = copy(data)
end

x_scaled = x ./ scale_factor


############################################################
# 4. Training Sample
############################################################

# Five years of hourly observations
T_train = 5 * 24 * 30 * 12
#T_train = length(data)

# Presample observations
x_init = x_scaled[1:p_max[1]]

# Training indices
train_idx = (p_max[1] + 1):(p_max[1] + T_train)

x_train         = x_scaled[train_idx]
timestamp_train = timestamp[train_idx]

T = length(x_train)

############################################################
# 5. Forecast / Test Sample
############################################################

# Forecast horizon: two weeks
h = 2 * 168

test_idx = (p_max[1] + T_train + 1):(p_max[1] + T_train + h)

# Test data on transformed/model scale
x_test = x_scaled[test_idx]

# Original-scale observations for forecast evaluation
y_test = data[test_idx]

time_test = timestamp[test_idx]


############################################################
# 6. Detrending / Centering
############################################################

# Initial level estimated from first 30 training observations
med_window = 30

train_mean = median(x_train[1:med_window])

# Center training observations
x_detrend = x_train .- train_mean


############################################################
# 7. Diagnostic Plot
############################################################

nplot = min(1000, T)

plot(
    timestamp_train[1:nplot],
    x_detrend[1:nplot];
    label  = "Detrended",
    xlabel = "Time",
    ylabel = log_transform ? "Log electricity demand" : "Electricity demand",
    title  = log_transform ?
             "Victorian Electricity Demand (log scale)" :
             "Victorian Electricity Demand"
)

plot!(
    timestamp_train[1:nplot],
    x_train[1:nplot];
    label = log_transform ? "Log demand" : "Original demand"
)