# ============================================================
# FUNCTIONS
# ============================================================

###############################
### Phi coeffs from Paper 2
###############################

function SimLocalMultiMA(ϕevol, σₑevol, μₓevol, p, s)

    T = size(ϕevol[1], 1) # T includes p presample values
    nSeasons = length(p)
    p_max    = sum(s.*p)

    x      = zeros(T)
    errors = zeros(T)
    errors = randn(T) .* σₑevol

    activelags = FindActiveLagsMultiSAR(p, s)

    # Simulate forward
    for t in (p_max+1):T

        MApolynomials = Array{Polynomial}(undef, length(p))

        for (l, pₗ) ∈ enumerate(p)
            MApolymat = [zeros(s[l]-1, pₗ); ϕevol[l][[t],:] .+ eps()]
            MApolynomials[l] = Polynomial([1; MApolymat[:]], :z)
        end

        ϕ̃ = coeffs(prod(MApolynomials))[2:end] # no S

        x[t] = μₓevol[t] +
               ϕ̃[activelags] ⋅ errors[t .- activelags] +
               errors[t]
    end

    x = x[p_max+1:end]

    return x, errors
end


###############################
### ARMA reparameterization
###############################

function arma_reparam_coef(x; ztrans="monahan", threshold=nothing, negative_signs=true)

    p = length(x)

    if ztrans == "sigmoid"
        P = tanh.(x./2)
    elseif ztrans == "monahan"
        P = x./sqrt.(1 .+ x.^2)
    elseif ztrans == "partials"
        P = clamp.(x, -0.9999, 0.9999)
        #P = x
    elseif ztrans == "linear" # No transformation
        return x, NaN
    else
        error("ztrans must be either 'sigmoid' or 'monahan'")
    end

    if !isnothing(threshold)
        P = clamp.(P, -threshold, threshold)
    end

    if negative_signs
        P = -P
    end

    ϕ = zeros(eltype(x), p, p) # Not sure we even need to allocate, but let's not worry about that.
    ϕ[1,1] = P[1]

    for k = 2:p
        for j = 1:k
            if k == j
                ϕ[k, j] = P[k]
            else
                ϕ[k, j] = ϕ[k-1, j] + P[k]*ϕ[k-1, k-j]
            end
        end
    end

    if negative_signs
        return -ϕ[end,:], P # polynomial 1 - ϕ₁B - ϕ₂B² - ... used for AR
    else
        return ϕ[end,:], P  # polynomial 1 + ϕ₁B + ϕ₂B² + ... used for MA
    end
end


###############################
### Transition matrices
###############################

function sma_transition_matrix(qmax::Int)

    G = zeros(qmax + 1, qmax + 1)

    # Bottom-left block: I_qmax
    G[2:end, 1:qmax] .= LinearAlgebra.I(qmax)

    return G
end


function sar_transition_matrix(ϕ̃::AbstractVector)

    p = length(ϕ̃)
    Φ = zeros(eltype(ϕ̃), p, p)

    # First row
    Φ[1, :] .= ϕ̃

    # Subdiagonal of ones
    @inbounds for i in 2:p
        Φ[i, i-1] = 1
    end

    return Φ
end


# ============================================================
# SIMULATION SETUP
# ============================================================

n = 500                    # sample size (after burn-in)
p = [1,1]                  # AR(1) regular, AR(1) seasonal
s = [1,12]                 # seasonal lag 12

pmax = sum(p .* s)         # presample size
T    = n + pmax            # total length including presample


# ============================================================
# VARIANCE
# ============================================================

σₑevol = ones(T)

σₑevol[1:200]     .= 1.0   ### OBS! Standard deviation
σₑevol[200+1:400] .= 1.0
σₑevol[400+1:end] .= 0.5

true_sd = σₑevol[14:end]


# ============================================================
# INTERCEPT
# ============================================================

t = range(0, 2π, length=T)
μₓevol = zeros(Float64, T)

#μₓevol = 3 .* cos.(2.8 .* t .+ 1.0) .+ 10
#true_intercept = μₓevol[p_max_ar+1:end]

μₓevol[1:150]     .= 5.0
μₓevol[150:300]   .= 1.0
μₓevol[300+1:T]   .= 5.0

#μₓevol = 2 .* cos.(0.8 .* t .+ 1.0)

true_intercept = μₓevol[pmax+1:end]

#plot(true_intercept)


# ============================================================
# REGULAR COEFFICIENT
# ============================================================

#0.6#-0.3
X = vcat(
    fill(0.9, div(n,2) + pmax),
    fill(-0.6, div(n,2))
)

X = reshape(X, :, 1)

phi_t = zeros(Float64, T)

phi_t = -[
    arma_reparam_coef(
        X[i];
        ztrans="monahan",
        threshold=nothing,
        negative_signs=false
    )[1]
    for i in 1:length(X)
]

#phi_t .= -0.5


# ============================================================
# SEASONAL COEFFICIENT
# ============================================================

Phi_t = zeros(Float64, T)

S = reshape(-0.9 .* sin.(2 .* π .* (1:T) ./ T), :, 1)
S = reshape(-S, :, 1)

Phi_t = [
    arma_reparam_coef(
        S[i];
        ztrans="monahan",
        negative_signs=false
    )[1]
    for i in 1:length(S)
]

#Phi_t = phi_t


# ============================================================
# FORMAT COEFFICIENT PATHS
# ============================================================

phi_flat = vcat(phi_t...)       # concatenate inner vectors
phi_t    = -reshape(phi_flat, :, 1)

Phi_flat = vcat(Phi_t...)       # concatenate inner vectors
Phi_t    = reshape(Phi_flat, :, 1)

ϕevol = [
    reshape(phi_t, T, 1),       # regular AR(1)
    reshape(Phi_t, T, 1)        # seasonal AR(1)
]


# ============================================================
# SIMULATE
# ============================================================

Random.seed!(10)

x, true_errors = SimLocalMultiMA(
    ϕevol,
    σₑevol,
    μₓevol,
    p,
    s
)


# ============================================================
# OPTIONAL: SAVE / STANDARDIZE
# ============================================================

#cd("C:/Users/Anna Fagerberg/Desktop/PROJECTS IN JULIA/DATA")

#df = DataFrame(value = data)
#CSV.write("sma11_niden.csv", df)

#_, md, sd = robust_standardize(x)
#x_data = (x .- md) ./ sd
#plot(x_data[1:100])
#x_data = data

#err_df = DataFrame(value = true_errors)
#CSV.write("sma11_error.csv", err_df)


# ============================================================
# TRUE COEFFICIENT PATHS AFTER PRESAMPLE
# ============================================================

#phi_t = zeros(Float64, T)
#phi_t[1:250] .= 0.5
#phi_t[250+1:T] .= -0.5

#Phi_t = 0.7 .* cos.(0.3 .* t .+ 1.0)

#Phi_t = zeros(Float64, T)
#Phi_t[1:250] .= -0.9
#Phi_t[250+1:T] .= 0.9

phi = phi_t[pmax+1:end]
Phi = Phi_t[pmax+1:end]


# ============================================================
# PLOTS
# ============================================================

# --- Simulated series ---

plot(
    x,
    title = "Simulated TV–SAR(1,1)\${}_{12}\$ Series",
    xlabel = "Time",
    ylabel = "xₜ",
    lw = 1.5,
    legend = false
)

plot!(true_intercept)


# --- Time-varying coefficients ---

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


