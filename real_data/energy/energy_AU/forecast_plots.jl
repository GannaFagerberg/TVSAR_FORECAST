############################################################
# Forecast Setup
############################################################

nPerGroup_fit = nPerGroup
g_new         = 1

T      = length(y_g)
season = s1

forecastHorizons = collect(1:length(y_test))

nPredPerIter = 10
statTrans    = ztrans

log_transform = true

if algoSettings.scaling == :none
    scaling       = false
else
    scaling       = true
end

normalization = algoSettings.normalization
#local_S = false

############################################################
# Initial Lag Vector for Forecasting
############################################################

T_all = length(x)
zₜall_orig = reverse!(x[T_all - p_max[1] + 1:T_all])
activeLags = activeLags_ar

############################################################
# Posterior State at Forecast Origin
############################################################

θₜpost = SAR_res[1][T, :, :]

############################################################
# State-Innovation Variance Parameters
############################################################

if scaling
    S_T = copy(SAR_res[10])
    if !algoSettings.normalization
        S_T .*= sqrt(nPerGroup_fit / g_new)
    end
else
 S_T = nothing
end

if algoSettings.state_var_type == :DSP

    Hₜpost = SAR_res[2][end, :, :] .-log(nPerGroup_fit) .+log(g_new)
    μpost = SAR_res[5] .-log(nPerGroup_fit) .+log(g_new)
    ϕpost = SAR_res[4]

else

    # Hₜpost is VARIANCE here
    Hₜpost = (SAR_res[7] ./ nPerGroup_fit) .* g_new
    ϕpost = nothing
    μpost = nothing

end


############################################################
# Observation Variance at Forecast Origin
############################################################
σₑₜpost = SAR_res[3]

############################################################
# Optional Fourier Terms
############################################################

if use_fourier

    fourier_coeffs = SAR_res[11][:, 25, :]   # static

    # fourier_coeffs = SAR_res[12]
    # fourier_coeffs = SAR_res[8]

    F_future = build_future_fourier_matrix(
        forecastHorizons;
        period = 24 * 365,
        K      = div(size(fourier_coeffs, 1), 2),
        t_last = T_all
    )

    S_future = F_future * fourier_coeffs

    four_s = reshape(
        S_future,
        size(S_future, 1),
        1,
        size(S_future, 2)
    )

    plot_state(
        four_s;
        prefix   = "Errors",
        ylim     = (-0.9, 0.9),
        xlim     = (0, 336),
        true_phi = nothing
    )
end


############################################################
# Optional SV / SVDSP Posterior Parameters
############################################################

σ²ₙpost = nothing
ϕ̃post   = nothing
μ̃post   = nothing
h̃post   = nothing
σ̄²ₙpost = nothing


# ----------------------------------------------------------
# Standard stochastic volatility
# ----------------------------------------------------------

if algoSettings.obs_var_type == :SV

    μ̃post   = SAR_res[6][1, :] .-log(nPerGroup_fit) .+log(g_new)
    ϕ̃post   = SAR_res[7][1, :]
    σ̄²ₙpost = SAR_res[9][1, :]


# ----------------------------------------------------------
# DSP stochastic volatility
# ----------------------------------------------------------

elseif algoSettings.obs_var_type == :SVDSP

    μ̃post = SAR_res[6][1, :]
    ϕ̃post = SAR_res[7][1, :]

    # h̃post = expand_grouped_states(
    #     SAR44_res[8],
    #     l,
    #     T_all
    # )[T_all, 1:end]

    h̃post = SAR_res[8][T_all, 1:end]

    # plot(h̃post)

end


############################################################
# Forecast Simulation
############################################################

t_st = time()

yPreds, LPS_est, AE = PredLocalMultiSAR_SV_gr(
    nPredPerIter,
    y_test,
    zₜall_orig,
    p1,
    season,
    statTrans,
    θₜpost,
    Hₜpost,
    ϕpost,
    σ²ₙpost,
    μpost,
    σₑₜpost,
    ϕ̃post,
    μ̃post,
    h̃post,
    σ̄²ₙpost,
    forecastHorizons;
    state_var_type      = state_var_type,
    obs_var_type        = obs_var_type,
    g_fc                = g_new,
    INTERCEPT           = INTERCEPT,

    H_freeze            = false,
    STATE_fixed         = false,
    use_fourier         = false,

    intercept_dynamics  = intercept_dynamics,
    startcol            = startcol,
    p_threshold         = p_threshold,

    scaling = scaling,
    S_T =nothing
)

t_end = time()

println(
    "Elapsed: ",
    (t_end - t_st) / 60,
    " mins"
)

############################################################
# Forecast Diagnostics
############################################################

# ----------------------------------------------------------
# Log predictive score
# ----------------------------------------------------------

plot(
    forecastHorizons,
    LPS_est;
    xlabel = "Forecast horizon",
    ylabel = "LPS",
    legend = false
)


# ----------------------------------------------------------
# Absolute forecast error
# ----------------------------------------------------------

plot(
    forecastHorizons,
    AE;
    xlabel = "Forecast horizon",
    ylabel = "Absolute error",
    legend = false
)


############################################################
# Extract Forecast Results
############################################################


h_length = maximum(forecastHorizons)

y_h50 = reshape(
    yPreds,
    length(y_test[1:h_length]),
    1,
    size(yPreds, 2)
)


############################################################
# Transform Forecasts Back
############################################################


y_mat = reshape(
    x_test,
    :,
    1
)



if log_transform

    res_transf = exp.(
        y_h50 .+ train_mean
    )

    truth = exp.(y_mat[1:h_length, :] .+train_mean)

else

    res_transf = y_h50 .+ train_mean
    truth = y_mat[1:h_length, :] .+train_mean

end

############################################################
# Plot Forecast Distribution and Truth
############################################################

plot_state(
    
# res_transf[1:forecastHorizons,:,:] .+
    # future_seasonal[1:forecastHorizons];

    res_transf[1:h_length, :, :];

    prefix   = "TV-SAR(1,1,1)s=24,168",
    ylim     = (-8000, 20000),
    xlim     = (0, forecastHorizons),
    true_phi = truth,
    alpha    = 0.05,
    use_hdi  = true

)