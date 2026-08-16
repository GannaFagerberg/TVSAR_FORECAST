############################################################
# Model Type and Configuration
############################################################


# ----------------------------------------------------------
# Deterministic Fourier terms
# ----------------------------------------------------------

# deterministic_fourier = false

# if deterministic_fourier
#     x_all = x_data - s_year
# else
#     x_all = x_data
# end


############################################################
# Data
############################################################

x = x_detrend
plot(x)
# T / (30 * 24)

############################################################
# Model Specification
############################################################
#model_type = :SMA
model_type  = :SAR
#model_type = :SARMA

SAR_conditional = model_type == :SAR ? false : false

# Variance specification
obs_var_type   = :static     # :SV, :SVDSP, :static
state_var_type = :DSP       # :DSP, :static

INTERCEPT = true
PerGroup = 24 * 30
#nPerGroup = 1

use_fourier = false

############################################################
# Intercept Dynamics
############################################################

if INTERCEPT
    intercept_dynamics = :rw   # :ll also possible, but here RW is used as usual
else
    intercept_dynamics = nothing
end


# ----------------------------------------------------------
# Starting column
# ----------------------------------------------------------

if INTERCEPT && intercept_dynamics == :ll
    startcol = 3

elseif INTERCEPT && intercept_dynamics == :rw   # rw
    startcol = 2

else
    startcol = 1
end

############################################################
# Seasonality and Model Orders
############################################################

# ==================================================
# Model specification
# ==================================================

if model_type == :SMA

    season = [1, 12]
    p      = [1, 5]

    s1 = season
    p1 = p

    s2 = season
    p2 = p

    pFit = sum(p2)


elseif model_type == :SARMA

    s1 = [1, 12]
    s2 = [1, 12]

    p1 = [1, 1]
    p2 = [1, 1]

    pFit = sum(p1) + sum(p2)


elseif model_type == :SAR

    season = [1, 12]
    p      = [1, 2]

    #season = [1, 12]
    #p      = [1, 1]

    s1 = season
    p1 = p

    s2 = season
    p2 = p

    pFit = sum(p1)


else

    error("model_type must be :SAR, :SMA, or :SARMA")

end

############################################################
# Maximum Lag
############################################################

# Total number of lags in all AR and MA polynomials
p_max = [
    sum(p1 .* s1),
    sum(p2 .* s2)
]


############################################################
# Number of State Components
# Additional Intercept + Slope
############################################################

if INTERCEPT

    nLags = INTERCEPT && intercept_dynamics == :ll ? pFit + 2 : pFit + 1

else

    nLags = pFit

end


############################################################
# DataFrame / Forecast Settings
############################################################

# T = length(data)
# h = 0


############################################################
# Prior Parameters
############################################################

if model_type in (:SARMA, :SMA)

    fitted_model = Arima(
        x,
        order = [5, 2, 5],
        seasonal = [5, 1, 5]
    )

    residuals      = fitted_model[:residuals]
    resid_variance = fitted_model[:sigma2]

    fitted_model = Arima(
        x[1:100],
        order = [2, 2, 5],
        seasonal = [5, 1, 5],
        include_mean = false,
        include_drift = true,
        include_constant = false
    )

    resid_variance = fitted_model[:sigma2]
    σ0 = sqrt(resid_variance)


elseif model_type == :SAR

    fitted_model = Arima(
        x[1:200],
        order = [2, 2, 5],
        seasonal = [5, 2, 5],
        include_mean = false,
        include_drift = false,
        include_constant = true
    )

    resid_variance = fitted_model[:sigma2]
    σ0 = sqrt(resid_variance)


else

    error("model_type must be :SAR, :SMA, or :SARMA")

end

############################################################
# Hyperparameters and Regressor Setup
############################################################

# ----------------------------------------------------------
# Static measurement-variance hyperparameters
# ----------------------------------------------------------

T = length(x)

alpha_sigma = 0.001
beta_sigma  = 0.001


############################################################
# SARMA Model
############################################################

# ============================================================
# Construct regressors by model type
# ============================================================

if model_type == :SARMA

    # MA regressors
    init_errors = vcat(
        fill(0.0, p_max[2]),
        residuals
    )

    _, Z_ma, T = SetupARReg(
        init_errors[:, 1],
        p_max[2]
    )

    # AR regressors
    init_y = fill(mean(x[1:50]), p_max[1])
    obs    = vcat(init_y, x)

    Y, Z_ar, T = SetupARReg(
        obs,
        p_max[1]
    )

    # Active lags
    activeLags_ar = FindActiveLagsMultiSAR(p1, s1)
    activeLags_ma = FindActiveLagsMultiSAR(p2, s2)

    Z_ar = Z_ar[:, activeLags_ar]
    Z_ma = Z_ma[:, activeLags_ma]

    # Combined design
    Z = hcat(Z_ar, Z_ma)


elseif model_type == :SMA

    Y = x

    # Initialize MA errors
    init_errors = vcat(
        fill(median(Y), p_max[2]),
        residuals
    )

    # Active MA lags and regressors
    activeLags_ma = FindActiveLagsMultiSAR(p2, s2)

    _, Z, T = SetupARReg_active(
        init_errors[:, 1],
        activeLags_ma
    )

    activeLags_ar = activeLags_ma
    errors_reg    = init_errors


elseif model_type == :SAR

    # Initialize observations / presample values
    if SAR_conditional

        obs = vcat(
            x[1:25],
            x[26:end]
        )

    else

        init_y = fill(mean(x[1:50]), p_max[1])
        obs    = vcat(init_y, x)

    end

    # Active AR lags
    activeLags_ar = FindActiveLagsMultiSAR(p1, s1)
    activeLags_ma = activeLags_ar

    # Construct regressors
    Y, Z, T = SetupARReg_active(
        obs,
        activeLags_ar
    )

    # Optional Fourier-adjusted regressors
    if use_fourier

        init_f = fill(
            mean(x[1:50]) - s_year[1],
            p_max[1]
        )

        obs_f = vcat(
            init_y,
            x_data - s_year
        )

        _, Z, _ = SetupARReg_active(
            obs_f,
            activeLags_ar
        )
    end


else

    error("model_type must be :SAR, :SMA, or :SARMA")

end


# ============================================================
# Posterior shape parameters
# ============================================================

if model_type == :SAR

    alpha_sigma_hat = alpha_sigma + T / 2

    alpha_sigma_hat_state =
        alpha_sigma + (T / nPerGroup - 1) / 2

else

    # alpha_sigma_hat = alpha_sigma + (T + p_max[2]) / 2

    alpha_sigma_hat = alpha_sigma + T / 2

end

############################################################
# Posterior Shape Parameters
############################################################

if model_type == :SAR

    alpha_sigma_hat = alpha_sigma + T / 2

    alpha_sigma_hat_state =
        alpha_sigma + (T / nPerGroup - 1) / 2

else

    # alpha_sigma_hat = alpha_sigma + (T + p_max[2]) / 2

    alpha_sigma_hat = alpha_sigma + T / 2

end


############################################################
# Grouping
############################################################
# Number of observations per group
# Here: approximately one month of hourly observations

# Needed for inference of the initial presample y-values
group_map = build_group_map(p_max[1],nPerGroup)

# Grouped Observations and Regressors
y_g = group_vector(Y,nPerGroup)

Cargs   = [Z[t,:] for t in 1:T]
Cargs_g =  group_vector_view(Cargs, nPerGroup)

cache_ar = build_sarma_cache(p1, s1, activeLags_ar)   # single cache
cache_ma = build_sarma_cache(p2, s2, activeLags_ma)   # single cache (if needed)

        
relu = true
    if relu == true
    ztrans = "partials"
    clipped_partials = true
    p_threshold      = 0.99 #OBS! 0.99 seems better than 0.9999
else
    ztrans = "monahan" 
    clipped_partials = false
    p_threshold      = 1000.0
    #ztrans = "linear" 
    #ztrans  = "sigmoid" 
end

### Settings
nBurn = 3000
nIter = 5000

###############
# FILTER TYPE
###############
iterated  = false
if iterated 
    num_iters = 5
else
    num_iters = 1
end

filtering_methods = ("iekf", "iekfl", "iukf", "iukfl", "iplf", "diplf")
kf_method         = filtering_methods[1]

# Variance components
var_mat = fill(0.3^2, nLags)   # MA coeffs unchanged

if INTERCEPT
    var_mat[1] = 1.0^2     # level c₀ (weak prior)
    #var_mat[1] = 5^2 
    if intercept_dynamics ==:ll
        var_mat[2] = 0.005^2       # slope d₀ (STRONG shrinkage)
    end
end

Σ₀ = PDMat(Diagonal(var_mat))
μ₀ = zeros(nLags)

if INTERCEPT && model_type == :SMA
    μ₀[1] = mean(Y[1:10])
end


priorSettings = (
            # --------------------------------------------------
            # State evolution priors
            # --------------------------------------------------
            ϕ₀ = 0.5,                 # Prior mean for ϕ
            κ₀ = 0.3,                 # Prior std for ϕ ~ N(ϕ₀, κ₀²)

            m₀ = -15.0 + log(nPerGroup),                 # Prior mean for μ
            σ₀ = 3.0,                 # Prior std for μ ~ N(m₀, σ₀²)

            ν₀ = 3.0,                # Prior df for σ²ₙ
            ψ₀ = 1.0,                # Prior scale for σ²ₙ

            μ₀ = μ₀,                 # Prior mean for initial state
            Σ₀ = Σ₀,                 # Prior covariance for initial state

            # --------------------------------------------------
            # Observation noise
            # --------------------------------------------------
            σₑ  = fill(σ0, T),

            alpha_sigma     = alpha_sigma,
            beta_sigma      = beta_sigma,
            alpha_sigma_hat = alpha_sigma_hat,

            # --------------------------------------------------
            # UKF / IEKF parameters
            # --------------------------------------------------
            α_ukf = 1e-3,
            β_ukf = 2.0,
            κ_ukf = 0
        )

algoSettings = (
            # --------------------------------------------------
            # MCMC control
            # --------------------------------------------------
            nBurn = nBurn,
            nIter = nIter + nBurn,

            # --------------------------------------------------
            # Model switches
            # --------------------------------------------------
            INTERCEPT = INTERCEPT,

            resid_label = iterated,
            method_label = Symbol(kf_method),

            model_type = model_type,
            
            SAR_conditional = SAR_conditional,

            obs_var_type   = obs_var_type ,     # :SV, :SVDSP, :static
            state_var_type = state_var_type ,    # :DSP, :static

            #ma_regressor_type = :current # :current
            ma_regressor_type = :median_freeze
            
            )

        
modelSettings = (
            
            # --------------------------------------------------
            # Structure
            # --------------------------------------------------
            
            nPerGroup  = nPerGroup,

            s1 = s1,
            p1 = p1,
            s2 = s2,
            p2 = p2,

            p_max = p_max,
            nLags = nLags,

            iterations = num_iters,
            
            # --------------------------------------------------
            # Measurement equation
            # --------------------------------------------------
        
            #C  = C_fun3,
            #∂C = derivC_fun3,

            Cargs = Cargs_g,
            Z     = Z,

            activeLags_ma=activeLags_ma,
            activeLags_ar=activeLags_ar,

            cache_ma = cache_ma,
            cache_ar = cache_ar,

            ztrans=ztrans,

            # --------------------------------------------------
            # DSP / mixture settings
            # --------------------------------------------------
            updateσₙ = false,
            nMixComp = 10,

            α = 0.5,
            β = 0.5,

            # --------------------------------------------------
            # Stochastic volatility (SV / DSP)
            # --------------------------------------------------
            ϕ̄₀ = 0.5,
            κ̄₀ = 0.3,

            m̄₀ = -15,
            σ̄₀ = 3,

            ν̄₀ = 3,
            ψ̄₀ = 1.0,

            ϕ̄  = 0.5,
            μ̄  = -15.0,
            σ̄²ₙ = 1.0,

            # --------------------------------------------------
            # Indexing
            # --------------------------------------------------
            intercept_dynamics=intercept_dynamics,

            # --------------------------------------------------
            # MA presample window
            # --------------------------------------------------
            T_use = 2*p_max[2],
            #T_use = p_max[2]+1,
            #use_fourier = true
        )


p  = p_max[1]
bw = p
k  = length(activeLags_ar)

#ws        = PresampleWorkspace(bw, k)
group_map = build_group_map(p, nPerGroup)
int_exp   = zeros(Float64, p)
x0_buf    = zeros(Float64, p)
m0_buf    = zeros(Float64, p)

cond_mean = zeros(Float64, T)
residuals = zeros(Float64, T)
group_map_T = build_group_map(T, nPerGroup)


### Gibbs
static_intercept = false
d_order          = 1
negative_signs   = true

### CHANGE THE CODES SO THAT I SWITCH BETWEEN MA AND AR in CACHE
scaled = false
#S_sc = fill(1.0, nLags)
#S_sc[1] = var(Y)

t_st = time()
Random.seed!(1)
#Random.seed!(115)  
SAR_res = GibbsSamplerTVSARMA_full(y_g, Y, priorSettings, modelSettings, algoSettings)
#SAR44_res = GibbsSamplerTVSARMA_full_fourier(y_g, Y, priorSettings, modelSettings, algoSettings)
#SAR44_res = GibbsSamplerTVSARMA_block(y_g, Y, priorSettings, modelSettings, algoSettings)
t_end = time()
println("Elapsed: ", (t_end - t_st)/60, " mins") #3.48, 1.29


#using JLD2f
#file="C:/Users/Anna Fagerberg/Desktop/PROJECTS_IN_JULIA/PAPERS_Julia/SARMA/REAL_DATA/Electricity_UK/SAR111_48_336_2016_2019_relu098.jld2"
#file="C:/Users/Anna Fagerberg/Desktop/PROJECTS_IN_JULIA/PAPERS_Julia/SARMA/REAL_DATA/Electricity_Australia/SAR222_24_168_presentation_gr_m_dsp_prior.jld2"
#JLD2.@save file SAR44_res
#JLD2.@load file SAR44_res#11 mins, Gaus prior - awful, 4000 ityer with yearly dynamic!

    #SAR44_res1 = SAR44_res
    #SAR44_res2 = SAR44_res
    #SAR44_res3 = SAR44_res
    #SAR44_res4 = SAR44_res
    #SAR44_res5 = SAR44_res

    #SAR44_res6 = SAR44_res
    #SAR44_res7 = SAR44_res
    #SAR44_res8 = SAR44_res
    #SAR44_res9 = SAR44_res
    #SAR44_res10 = SAR44_res

multiple_seeds = false
scaled         = false

if multiple_seeds
seeds = [12, 16, 78, 77, 98]
#seeds = [1, 2, 3, 4, 5]

SAR44_results = Vector{Any}(undef, length(seeds))

for (i, s) in enumerate(seeds)

    println("Running chain $i with seed = $s")
    t_st = time()
    Random.seed!(s)
    SAR44_results[i] = GibbsSamplerTVSARMA_full(
        y_g,
        Y,
        priorSettings,
        modelSettings,
        algoSettings
    )

    t_end = time()
    println("Elapsed: ", round((t_end - t_st)/60, digits=2), " mins")
end
end

#airport_results_scaled_globally =  SAR44_results 
#SAR44_results_demean_scaled =  SAR44_results 

  #SAR44_results_stand =  SAR44_results 
  #SAR44_results_demean_scaled =  SAR44_results 
  #SAR44_results = SAR44_results[2]
  #SAR44_results = SAR44_results_demean[2]; sd=1
  #SAR44_results_monah = SAR44_results


  #sd