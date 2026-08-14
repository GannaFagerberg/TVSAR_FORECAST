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

df = CSV.read(
    "AustralianElecPriceDemand202512_hourly.csv",
    DataFrame
)

# Victorian electricity demand and timestamps
data      = df.demand_VIC
timestamp = df.datetime


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


# ----------------------------------------------------------
# Transformation
# ----------------------------------------------------------

if log_transform
    x = log.(data)
else
    x = copy(data)
end


# ----------------------------------------------------------
# Scaling
# ----------------------------------------------------------

x_scaled = x ./ scale_factor


############################################################
# 4. Training Sample
############################################################

# Five years of hourly observations
T_train = 5 * 24 * 30 * 12

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

x_med = median(x_train[1:med_window])

# Center training observations
x_detrend = x_train .- x_med


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