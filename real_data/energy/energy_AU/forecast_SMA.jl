############################################################
# Forecast Setup — SMA
############################################################

nPerGroup_fit = nPerGroup
g_new         = 1

T      = length(y_g)
season = s2

forecastHorizons = collect(1:length(y_test))

nPredPerIter = 10
statTrans    = ztrans

if algoSettings.scaling == :none
    scaling = false
else
    scaling = true
end

normalization = algoSettings.normalization


############################################################
# Initial MA Error Vector for Forecasting
############################################################

activeLags = activeLags_ma
maxlag_ma  = maximum(activeLags_ma)

errors_mx = SMA_res[6]

# errors_mx has dimensions
# (T_all + p_max[2]) × 1 × freeze_iter
#
# We only need the final maxlag_ma errors.

errors_tail = @view errors_mx[
    end-maxlag_ma+1:end,
    1,
    :
]

# Frozen median error trajectory used by median_freeze
errors_tail_med = vec(
    median(errors_tail; dims = 2)
)

# Required ordering:
# [e_T, e_{T-1}, e_{T-2}, ...]
eₜall_orig = reverse(errors_tail_med)


############################################################
# Posterior State at Forecast Origin
############################################################

θₜpost = SMA_res[1][T, :, :]


############################################################
# State-Innovation Variance Parameters
############################################################

if algoSettings.state_var_type == :DSP

    Hₜpost =
        SMA_res[2][end, :, :] .-
        log(nPerGroup_fit) .+
        log(g_new)

    μpost =
        SMA_res[5] .-
        log(nPerGroup_fit) .+
        log(g_new)

    ϕpost = SMA_res[4]

else

    error(
        "Current SMA return does not contain the posterior " *
        "static state variances."
    )

end


############################################################
# Observation Variance
############################################################

σₑₜpost = SMA_res[3]

t_st = time()

yPreds, LPS_est, AE = PredLocalMultiSMA_gr(
    nPredPerIter,
    y_test,
    eₜall_orig,
    p2,
    s2,
    statTrans,
    θₜpost,
    Hₜpost,
    ϕpost,
    μpost,
    σₑₜpost,
    forecastHorizons;
    g_fc               = g_new,
    INTERCEPT           = INTERCEPT,
    H_freeze            = false,
    STATE_fixed         = false,
    intercept_dynamics  = intercept_dynamics,
    startcol            = startcol,
    p_threshold         = p_threshold,
    scaling             = scaling,
    S_T                 = scaling ? S_T : nothing
)

println(
    "Elapsed: ",
    (time() - t_st) / 60,
    " mins"
)

#SMA_res= SAR_res

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
    h_length,
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

    truth = exp.(
        y_mat[1:h_length, :] .+ train_mean
    )

else

    res_transf = y_h50 .+ train_mean

    truth = y_mat[1:h_length, :] .+ train_mean

end


############################################################
# Plot Forecast Distribution and Truth
############################################################

plot_state(
    res_transf[1:h_length, :, :];
    prefix   = "TV-SMA",
    ylim     = (0, 8000),
    xlim     = (0, h_length),
    true_phi = truth,
    alpha    = 0.05,
    use_hdi  = true
)