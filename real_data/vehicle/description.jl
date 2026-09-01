
using HTTP, CSV, DataFrames, Dates
#👉 https://fred.stlouisfed.org/series/TRFVOLUSM227NFWA
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=TRFVOLUSM227NFWA"
df = CSV.read(IOBuffer(HTTP.get(url).body), DataFrame)

rename!(df, [:DATE, :passengers])

df.DATE = Date.(df.DATE)

# ============================================================
# Initial training period: through June 2016
# ============================================================

df_train = df[
    df.DATE .<= Date(2016, 6, 1),
    :
]
y_train =Float64.(df_train.passengers)
time_train =df_train.DATE
timestamp_train = df_train.DATE

plt_training=plot(timestamp_train, y_train, legend=false)
save_dir = raw"C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST\real_data\vehicle"
savefig(plt_training,joinpath(save_dir, "TVSAR_vehicle_training.pdf"))

plt_all=plot(df.DATE, df.passengers, legend=false)
save_dir = raw"C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST\real_data\vehicle"
savefig(plt_all,joinpath(save_dir, "TVSAR_vehicle_all.pdf"))



@show nrow(df_train)
@show first(df_train.DATE)
@show last(df_train.DATE)

#plot(y)
logy= log.(y_train)
#plot(logy[600:610])
#T_thr = 605

#sd     = maximum(logy[1:675-36])
sd     = 1
#sd = std(logy[1:675-36])
train_mean = median(logy[1:30])
x = (logy .-train_mean)./sd

#y_test = logy[675-36+1:end]
#x_data = x_data[1:675-36]
plot(x)