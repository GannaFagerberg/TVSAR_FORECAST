if gethostname() == "pop-os"
    remote = false
else 
    remote = true
end
if remote
    slurm_id = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"]) 
else
    if isempty(ARGS) # no command line argumens
        slurm_id = 1
    else
        slurm_id = parse(Int, ARGS[1]) # first command line argument
    end
end
println("The slurm id of this worker is $(slurm_id)")

Random.seed!(1)

ps = 1:3
Ps = 1:3
models = []
for p = ps
    for P = Ps
        push!(models,[p,P])
    end
end

using Pkg
Pkg.activate(abspath(joinpath(dirname(@__DIR__),".."))) # Activate envir
Pkg.instantiate(); Pkg.precompile() # Set up environment dictated by manifest

using SpecialFunctions, Plots, Measures, PolyaGammaSamplers, PDMats, LaTeXStrings
using LinearAlgebra, BandedMatrices, SparseArrays, MCMCDiagnosticTools
using KernelDensity, Distributions, Dates
using Random, StatsBase, DSP
using CSV, DataFrames, JLD2, ProgressMeter
using Polynomials, RCall
using SpecTools, LocalMultiSAR 
using LogisticBetaDistribution

experimentName = "usip"
mainFolder = joinpath(dirname(@__DIR__), experimentName)
resultsFolder = mainFolder*"/results"
dataFolder = mainFolder*"/data"
figFolder = mainFolder*"/figs"

# Plot settings
gr(legend = :topleft, grid = false, color = colors[2], lw = 2, legendfontsize=8,
    xtickfontsize=8, ytickfontsize=8, xguidefontsize=8, yguidefontsize=8,
    titlefontsize = 10, markerstrokecolor = :auto)

# Fitted model settings
statTrans = "monahan"
algorithm = :FFBSx
nParticles = 100
season = [1,12]
pFit = models[slurm_id]    # Number of fitted lags in each AR polynomial
SV = true       # Fit stochastic volatility model for errors, εₜ
nIter = 10000
nBurn = 3000                        # Burn-in period
nInitFFBS = 500                     # Number of runs with ffbs-x to get initial values
initVal = "fixed"                   # "prior" or "fixed"
offset = eps()                      # offset log-volatility. Number or "kowal"
intercept = false                   # should an intercept be fitted, or process is zero mean

# Log predictive score options
testDates = Date("2014-07-01"):Month(1):Date("2024-06-01")
forecastHorizons = [1,12]
nPredPerIter = 100

nTest = length(testDates)
modelName = experimentName*"SAR$(pFit[1])$(pFit[2])_long"
Random.seed!(1251222)

# load the data
data = load(dataFolder*"/usipdata_long.jld2")
x = data["x"]
dates = data["dates"]

activeLags = FindActiveLagsMultiSAR(pFit, season) # Non-zero lags in ϕ(L)Φ(L^s) polynomial

# Fit using OLS with all lags - note that T is now T - p_max
p_max = sum(pFit .* season)
y, Z, T = SetupARReg(x, p_max);
ydates = dates[(p_max+1):end] # Dates after the lost initial values
if intercept
    Z = [Z ones(T)]
end
ϕ̂ = Z \ y # Includes also the non-zero coeff. Could do better, is only for initial val
sₑ = sqrt(sum(((y - Z*ϕ̂).^2))/(T-p_max))

μ₀, Σ₀ = NormalApproxUniformStationary(pFit)
priorSettings = (
    ϕ₀ = 0.5, κ₀ = 0.3, ν₀ = 3, ψ₀ = 1, m₀ = -15, σ₀ = 3, νₑ = 3, ψₑ = sₑ, μ₀ = μ₀, Σ₀ = Σ₀,
    ϕ̄₀ = 0.86, κ̄₀ = 0.11, ν̄₀ = 3, ψ̄₀ = 0.1, m̄₀ = log(sₑ^2), σ̄₀ = 3 # SV parameters
); 

modelSettings = (p = pFit, season = season, ztrans = statTrans, 
fixσₙ = 1, α = 1/2, β = 1/2, intercept = false, SV = SV) 
algoSettings = (θupdate = algorithm, nIter = nIter, nBurn = nBurn, 
    nParticles = nParticles, nInitFFBS = nInitFFBS, initVal = initVal, 
    offset = offset) 

# Animate predictions
animatePred = true
startPlotIdx =  findfirst(dates .== testDates[1] - Month(12))
extendedDates = [dates;dates[end] + Month(1):Month(1):(dates[end] + Month(maximum(forecastHorizons)))]

thinFactor = 10 # Only keep every thinFactor:th sample when doing predictions
LPSs = zeros(nTest, length(forecastHorizons))
MAEs = zeros(nTest, length(forecastHorizons))
maxHorizon = maximum(forecastHorizons)
initialValues = nothing
anim = @animate for (j, testDate) in enumerate(testDates)  

    endTrainingIdx = findfirst(dates .== testDate) - 1

    # Run Gibbs
    θpost, Hpost, σₑpost, ϕpost, σ²ₙpost, μpost, ϕARpost, ϕ̄post, μ̄post, σ̄²ₙpost =   
        GibbsLocalMultiSAR_SV(x[1:endTrainingIdx], modelSettings, priorSettings, algoSettings, initialValues);

    # Make predictions and compute LPS
    yTest = x[findfirst(dates .== testDate):end]
    zₜall = Z[findfirst(ydates .== testDate),:] # contains all p_max lags. Later trimmed.
    yPred, LPS, MAE = PredLocalMultiSAR_SV(nPredPerIter, yTest, zₜall, pFit, season, statTrans,
        θpost[end,:,1:thinFactor:end], Hpost[end,:,1:thinFactor:end], 
        ϕpost[:,1:thinFactor:end], σ²ₙpost[:,1:thinFactor:end], μpost[:,1:thinFactor:end], σₑpost[end,1:thinFactor:end], ϕ̄post[1:thinFactor:end], μ̄post[1:thinFactor:end], σ̄²ₙpost[1:thinFactor:end], forecastHorizons)
    LPSs[j,:] .= LPS
    MAEs[j,:] .= MAE

    if animatePred
        medianPred = median(yPred, dims = 2)
        plotIdx = startPlotIdx:length(x)
        plot(dates[plotIdx], x[plotIdx], label = "training", ylims = (-0.35,0.25),
            xlims = (dates[startPlotIdx],dates[length(x)] + Month(maximum(forecastHorizons))));
        plotTestIdx = endTrainingIdx+1:length(x);
        plot!(dates[plotTestIdx], x[plotTestIdx], color = colors[1], label = "test");
        plotPredIdx = endTrainingIdx+1:endTrainingIdx + length(medianPred)
        plot!(extendedDates[plotPredIdx], medianPred, color = colors[3], 
            label = "prediction")
        plot!(extendedDates[plotPredIdx], medianPred - 1.96*std(yPred, dims = 2), 
            color = colors[3], label = "", linestyle = :solid, lw = 1)
        plot!(extendedDates[plotPredIdx], medianPred + 1.96*std(yPred, dims = 2), 
            color = colors[3], label = "", linestyle = :solid, lw = 1)
    else
        plot()
    end

    # Update initial values using most recent posterior
    θ = median(θpost, dims = 3)[:,:,1]
    θ = [θ; θ[end,:]']
    H = median(Hpost, dims = 3)[:,:,1]
    H = [H; H[end,:]']
    global initialValues = (θ = θ, H = H, μ = median(μpost, dims = 2)[:], 
        ϕ = median(ϕpost, dims = 2)[:])
    if j == 1 # Good initial values now, cut down on nBurn and nIter
        global algoSettings = (θupdate = algorithm, nIter = round(Int, nIter/1), 
            nBurn = round(Int, nBurn/1), nParticles = nParticles, 
            nInitFFBS = nInitFFBS, initVal = initVal, offset = offset) 
        #global thinFactor = round(Int, thinFactor/5)
    end

end # end loop over test data

if animatePred 
    gif(anim, figFolder*"/$(modelName)_anim.gif", fps = 5)
end

save(resultsFolder*"/$(modelName).jld2", Dict(
    "x" => x, 
    "testDates" => testDates,
    "LPSs" => LPSs,
    "MAEs" => MAEs,
    "forecastHorizons" => forecastHorizons
    )
)
