
using Polynomials
using Plots
using Random
using LinearAlgebra
using Statistics

function SimLocalMultiSAR(ϕevol, σₑevol, μₓevol, p, s)

    T = size(ϕevol[1], 1) # T includes p presample values
    nSeasons = length(p)
    p_max = sum(s.*p)
    
    errors     = randn(T) .* σₑevol # innovations
    x          = zeros(T)
    x[1:p_max].= μₓevol[1:p_max]

    activelags = FindActiveLagsMultiSAR(p, s)
    ϕ_vec = zeros(T,pmax)

    # Simulate forward
    for ind in p_max+1:T
        
        ARpolys = Vector{Polynomial}(undef, length(p1))
        for (l, pₗ) in enumerate(p1)
            block = [zeros(s1[l]-1, pₗ); -ϕevol[l][ind, :]' .+ eps()]
            ARpolys[l] = Polynomial([1; block[:]], :z)
        end
        ϕ_vec[ind, :] = -coeffs(prod(ARpolys))[2:end]

        x[ind] =
            μₓevol[ind] +
            dot(ϕ_vec[ind, :][activeLags_ar], (x[ind .- activeLags_ar]-μₓevol[ind .- activeLags_ar])) +
            errors[ind]
    end

    #true_b0 =  μₓevol .*(1.0 .-sum(ϕ_vec[:, activelags], dims=2))
    #true_mean = true_b0 ./(1.0 .-sum(ϕ_vec[:, activelags], dims=2))

    #return x, ϕ_vec, true_b0, true_mean
    #return x, ϕout, σₑout, μₓout
    return x
end


#################
# Simulate SAR11 
#################

n = 500 +24                    # sample size (after burn-in)
p = [1, 1]                 # AR(1) regular, AR(1) seasonal
s = [1, 12]                # seasonal lag 12
pmax = sum(p .* s)         # presample size
T = n + pmax               # total length including presample
σₑevol = ones(T)
p1=p;s1=s;season1=s
#μₓevol = true_b0

t = range(0, 2π, length=T)

μₓevol = zeros(Float64, T)
μₓevol .= 10.0 .* cos.(2.0 .* t .+ 1.0) 
#μₓevol .= 5.0 .+0.5 .* t
#μₓevol[300:400] .= 6.0

μₓevol[1:250]    .= 10.0
μₓevol[251:end]  .=3.0
plot(μₓevol)

phi_t  = zeros(T)
Phi_t  = zeros(T)

σₑevol = ones(T)
t = range(0, 2π, length=T)

activeLags_ar    = FindActiveLagsMultiSAR(p1, s1)

# regular AR(1)
#phi_t .= 0.1 .* t
#phi_t  = 0.9 .* sin.(0.5 .* t)    # stays in (-0.8, 0.8)
#phi_t  = 0.5 .* sin.(0.5 .* t)    # stays in (-0.8, 0.8)

phi_t[1:250] .= 0.8
phi_t[251:end] .= -0.8
Phi_t  = 0.8 .* cos.(0.3 .* t .+ 1.0)  # stays in (-0.8, 0.7)
plot(phi_t)
#phi_t[1:250] .= 0.96
#phi_t[251:end] .= 0.96
#Phi_t .= 0.7
#Phi_t[1:250] .= 0.8
#Phi_t[251:end] .= 0.8

ϕevol = [
    reshape(phi_t, T, 1),     # regular AR(1)
    reshape(Phi_t, T, 1)      # seasonal AR(1)
]

Random.seed!(130)
data_sim = SimLocalMultiSAR(ϕevol, σₑevol, μₓevol, p, s)
#true_intercept = true_b0[pmax+1:end]

phi  = phi_t[pmax+1:end]
Phi  = Phi_t[pmax+1:end]
true_intercept =μₓevol[pmax+1:end].*(1.0 .- phi .- Phi .+ phi.*Phi)
true_mean =μₓevol[pmax+1:end]

#cd("C:/Users/Anna Fagerberg/Desktop/PROJECTS IN JULIA/PAPERS_Julia/SARMA/SIM_DATA")
##df = DataFrame(value = data)
#CSV.write("sar11_niden.csv", df)

# Plot x
data = data_sim[pmax+1:end]
plot(data)
plot!(true_intercept)
x_mean = 0
sd=1
#plot!(true_mean)
#minx=minimum(x)
#maxx=maximum(x)
true_intercept[1]

phi  = phi_t[pmax+1:end, 1]    # regular AR coefficient
Phi = Phi_t[pmax+1:end, 1]      # seasonal AR coefficient

plot(
    phi,
    label = "phi(t)",
    xlabel = "Time",
    ylabel = "Coefficient value",
    title = "Time-Varying AR Coefficients",
    lw = 2
)

plot!(
    Phi,
    label = "Phi(t)",
    lw = 2,
    ls = :dash
)

#_,_sd = robust_standardize(data)
#sd = maximum(data)
sd = 1
x_mean = median(data[1:30])
x_data = (data .- x_mean)./sd
plot(x_data)
#plot(data)


train_mean =x_mean

y_test = x_data[501:end]
x_data = x_data[1:500]
######################################################
#x_data = data[14:end]
#true_intercept = true_b0[14:end]
#true_intercept = μₓevol[14:end]
#plot(true_intercept)


# Compute true intercept/mean
ϕ_vec = zeros(513,pmax)
for ind in 14:513 # t0=t0
        ARpolys = Vector{Polynomial}(undef, length(p1))
        for (l, pₗ) in enumerate(p1)
            block = [zeros(s1[l]-1, pₗ); -ϕevol[l][ind, :]' .+ eps()]
            ARpolys[l] = Polynomial([1; block[:]], :z)
        end
        ϕ_vec[ind-13, :] = -coeffs(prod(ARpolys))[2:end]
end

true_mean = true_b0 ./(1 .- sum(ϕ_vec[:,activeLags_ar], dims = 2))
#plot(true_mean)

#true_intercept = μₓevol .*(1 .- sum(ϕ_vec[:,activeLags_ar], dims = 2))
#true_intercept = true_intercept[p_max_ar+1:end,:]
#plot(true_mean)




#################
# Simulate SAR111 
#################


n = 2000                    # sample size (after burn-in)

p = [1, 1, 1]          # SAR(1) for each seasonal period
s = [7, 30, 365]       # weekly, monthly-ish, annual

pmax = sum(p .* s)         # presample size
T = n + pmax + pmax              # total length including presample

nSeasons = length(p)

# ---------------------------
# Build time-varying seasonal AR coefficients in (-1,1)
# ---------------------------
ϕevol = Vector{Matrix{Float64}}(undef, nSeasons)

for j in 1:nSeasons
    # Smooth sinusoidal evolution within (-0.8, 0.8)
    ϕj = 0.8 .* sin.(2π .* (1:T) ./ (500 * j))
    ϕevol[j] = reshape(ϕj, T, 1)   # T × p_j matrix
end

# ---------------------------
# Zero mean & constant sigma
# ---------------------------
μₓevol = zeros(T)
σₑevol = ones(T)

# ---------------------------
# Simulate with your function
# ---------------------------
x, ϕout, σout, μout = SimLocalMultiSAR(ϕevol, σₑevol, μₓevol, p, s)


println("Length of simulated series: ", length(x))

phi  = ϕout[1][(pmax+1):end, 1]    # regular AR coefficient
#Phi1 = ϕout[2][(pmax+1):end, 1]     # seasonal AR coefficient
#Phi2 = ϕout[3][(pmax+1):end, 1]

plot(
    phi,
    label = "phi(t)",
    xlabel = "Time",
    ylabel = "Coefficient value",
    title = "Time-Varying AR Coefficients",
    lw = 2
)

plot!(
    Phi1,
    label = "Phi(t)",
    lw = 2,
    ls = :dash
)

plot!(
    Phi2,
    label = "Phi(t)",
    lw = 2,
    ls = :dash
)

#################
# Simulate AR1 
#################

n = 500                    # sample size (after burn-in)
p = [1]                 # AR(1) regular, AR(1) seasonal
s = [1]                # seasonal lag 12
pmax = sum(p .* s)         # presample size
T = n + pmax + pmax              # total length including presample

μₓevol = zeros(T)
σₑevol = ones(T)

# regular AR(1)
phi_t  = [fill(-0.98, (n ÷ 2) + 2* pmax); fill(0.97, n ÷ 2)]
#phi_t  = [fill(0.5, (n ÷ 2) + 2* pmax); fill(-0.5, n ÷ 2)]

ϕevol = [
    reshape(phi_t, T, 1)
]

Random.seed!(100)
x, ϕout, σout, μout = SimLocalMultiSAR(ϕevol, σₑevol, μₓevol, p, s)


#using Plots

# --- 1. Plot the simulated time series ---
plot(
    x,
    title = "Simulated TV–SAR(1,1)\${}_{12}\$ Series",
    xlabel = "Time",
    ylabel = "xₜ",
    lw = 1.5,
    legend = false
)

x = hcat(x)
df = DataFrame(x, :auto)   # convert matrix → DataFrame
CSV.write("AR1_nonst.csv", df)

phi  = ϕout[1][(pmax+1):end, 1]    # regular AR coefficient
#Phi = ϕout[2][:, 1]     # seasonal AR coefficient

plot(
    phi,
    label = "phi(t)",
    xlabel = "Time",
    ylabel = "Coefficient value",
    title = "Time-Varying AR Coefficients",
    lw = 2
)


# Trying to istall R
using RCall
function Arima(y; order = [0,0,0], seasonal = [0,0,0], xreg = nothing, include_mean = false,
    include_drift = false, include_constant = false, frequency = 1, deltat = 1)
    R"""
    suppressMessages(library(forecast))
    fittedModel = Arima(ts($y, frequency = $frequency, deltat = $deltat), order = $order, seasonal = $seasonal, xreg = $xreg, 
        include.mean = $include_mean, include.drift = $include_drift, include.constant = $include_constant)
    """
    @rget fittedModel
    return fittedModel
end

fitted_model = Arima(data[1:100,:], order=[5,0,0], seasonal=[5,0,0])
#residuals = fitted_model[:residuals]  # extracts residuals as a Julia vector
#resid_variance = var(residuals)  # sample variance of residuals
#println("Estimated residual variance: ", resid_variance)
#sqrt(resid_variance)

resid_variance = fitted_model[:sigma2]
println("Residual variance from ARIMA fit: ", resid_variance)
sqrt(resid_variance)



#################
# Simulate SAR22 
#################

n = 500                   

p = [2,2]          # SAR(1) for each seasonal period
s = [1, 12]       # weekly, monthly-ish, annual

pmax = sum(p .* s)         # presample size
T = n + pmax + pmax              # total length including presample

nSeasons = length(p)

# ---------------------------
# Build time-varying seasonal AR coefficients in (-1,1)
# ---------------------------
ϕevol = Vector{Matrix{Float64}}(undef, nSeasons)

for j in 1:nSeasons
    # Smooth sinusoidal evolution within (-0.8, 0.8)
    ϕj = 0.8 .* sin.(2π .* (1:T) ./ (500 * j))
    ϕevol[j] = reshape(ϕj, T, 1)   # T × p_j matrix
end

# ---------------------------
# Zero mean & constant sigma
# ---------------------------
μₓevol = zeros(T)
σₑevol = ones(T)

# ---------------------------
# Simulate with your function
# ---------------------------
x, ϕout, σout, μout = SimLocalMultiSAR(ϕevol, σₑevol, μₓevol, p, s)


println("Length of simulated series: ", length(x))

phi  = ϕout[1][(pmax+1):end, 1]    # regular AR coefficient
#Phi1 = ϕout[2][(pmax+1):end, 1]     # seasonal AR coefficient
#Phi2 = ϕout[3][(pmax+1):end, 1]

plot(
    phi,
    label = "phi(t)",
    xlabel = "Time",
    ylabel = "Coefficient value",
    title = "Time-Varying AR Coefficients",
    lw = 2
)

plot!(
    Phi1,
    label = "Phi(t)",
    lw = 2,
    ls = :dash
)

plot!(
    Phi2,
    label = "Phi(t)",
    lw = 2,
    ls = :dash
)



# Compute true intercept/mean
ϕ_vec = zeros(513,pmax)
for ind in 14:513 # t0=t0
        ARpolys = Vector{Polynomial}(undef, length(p1))
        for (l, pₗ) in enumerate(p1)
            block = [zeros(s1[l]-1, pₗ); -ϕevol[l][ind, :]' .+ eps()]
            ARpolys[l] = Polynomial([1; block[:]], :z)
        end
        ϕ_vec[ind-13, :] = -coeffs(prod(ARpolys))[2:end]
end

#true_mean = μₓevol ./(1 .- sum(ϕ_vec[:,activeLags_ar], dims = 2))
#plot(true_mean)

#true_mean = μₓevol[pmax+1:end]
# Compute true intercept/mean
ϕ_vec = zeros(513,pmax)
for ind in 14:513 # t0=t0
        ARpolys = Vector{Polynomial}(undef, length(p1))
        for (l, pₗ) in enumerate(p1)
            block = [zeros(s1[l]-1, pₗ); -ϕevol[l][ind, :]' .+ eps()]
            ARpolys[l] = Polynomial([1; block[:]], :z)
        end
        ϕ_vec[ind-13, :] = -coeffs(prod(ARpolys))[2:end]
end

true_mean = true_b0 ./(1 .- sum(ϕ_vec[:,activeLags_ar], dims = 2))
plot(true_mean)

true_intercept = μₓevol .*(1 .- sum(ϕ_vec[:,activeLags_ar], dims = 2))
plot(true_intercept)