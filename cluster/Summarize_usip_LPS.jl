using Pkg
Pkg.activate(abspath(joinpath(dirname(@__DIR__),".."))) # Activate envir
Pkg.instantiate(); Pkg.precompile() # Set up environment dictated by manifest

using JLD2, Statistics, Plots, ProgressMeter, Measures, StatsPlots, LaTeXStrings
using Latexify
using LocalMultiSAR

experimentName = "usip"
mainFolder = joinpath(dirname(@__DIR__), experimentName)
resultsFolder = mainFolder*"/results"
dataFolder = mainFolder*"/data"
figFolder = mainFolder*"/figs"

res = load(resultsFolder*"/usipSAR11_long.jld2")
testDates = res["testDates"]
forecastHorizons = res["forecastHorizons"]

nruns = 3
pMax = 3
PMax = 3
LPS = zeros(pMax,PMax,length(forecastHorizons), nruns)
MAE = zeros(pMax,PMax,length(forecastHorizons), nruns)
LPS_time = Array{Matrix}(undef, pMax, PMax, nruns)
MAE_time = Array{Matrix}(undef, pMax, PMax, nruns)
#res = Matrix{Matrix}(undef, pMax, PMax)
for r in 0:(nruns-1)
    for p in 1:pMax
        for P in 1:PMax
            LPS_time[p,P,r+1] = load(resultsFolder*"/usipSAR$(p)$(P)_long_$(r).jld2")["LPSs"]
            MAE_time[p,P,r+1] = load(resultsFolder*"/usipSAR$(p)$(P)_long_$(r).jld2")["MAEs"]
            for (i,h) in enumerate(forecastHorizons)
                LPS[p,P,i,r+1] = sum(LPS_time[p,P,r+1][.!isnan.(LPS_time[p,P,r+1][:,i]),i])
                MAE[p,P,i,r+1] = sum(MAE_time[p,P,r+1][.!isnan.(MAE_time[p,P,r+1][:,i]),i])
            end
        end
    end
end

latexify(round.(mean(LPS[:,:,1,:], dims = 3)[:,:,1], digits = 2))
latexify(round.(std(LPS[:,:,1,:], dims = 3)[:,:,1]/sqrt(nruns), digits = 2))


plot(mean(LPS[:,:,1,:], dims = 3)[1,:], label = L"p=1", col = colors[1])
plot!(mean(LPS[:,:,1,:], dims = 3)[2,:], label = L"p=2", col = colors[2])
plot!(mean(LPS[:,:,1,:], dims = 3)[3,:], label = L"p=3", col = colors[3])

plts = []
for p = 1:pMax
    for P = 1:PMax
        plt = plot()
        for r = 0:(nruns-1)
            plot!(LPS_time[p,P,r+1][:,1], col = colors[r+1], 
                label = "Run $(r), LPS = $(round(LPS[p,P,1,r+1], digits =2))")
        end
        push!(plts, plt)
    end
end 
plot(plts..., layout = (pMax, PMax), size = (900,600))

# STATIC MODELS
res = load(resultsFolder*"/usipStaticSAR11_long.jld2")
testDates = res["testDates"]
forecastHorizons = res["forecastHorizons"]
pMax = 3
PMax = 5
LPSStatic = zeros(pMax,PMax,length(forecastHorizons))
MAEStatic = zeros(pMax,PMax,length(forecastHorizons))
LPSStatic_time = Matrix{Matrix}(undef, pMax, PMax)
MAEStatic_time = Matrix{Matrix}(undef, pMax, PMax)
for p in 1:pMax
    for P in 1:PMax
        LPSStatic_time[p,P] = 
            load(resultsFolder*"/usipStaticSAR$(p)$(P)_long.jld2")["LPSs"]
        MAEStatic_time[p,P] = 
            load(resultsFolder*"/usipStaticSAR$(p)$(P)_long.jld2")["MAEs"]
        for (i,h) in enumerate(forecastHorizons)
            LPSStatic[p,P,i] = 
                sum(LPSStatic_time[p,P][.!isnan.(LPSStatic_time[p,P][:,i]),i])
            MAEStatic[p,P,i] = 
                sum(MAEStatic_time[p,P][.!isnan.(MAEStatic_time[p,P][:,i]),i])
        end
    end
end
latexify(round.(LPSStatic[:,:,1], digits = 2))
latexify(round.(MAEStatic[:,:,1], digits = 3))
