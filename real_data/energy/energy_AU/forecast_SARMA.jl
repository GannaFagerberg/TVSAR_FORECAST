############################################################
# Forecast Setup — SARMA
############################################################
SARMA_res = SAR_res

nPerGroup_fit = nPerGroup
g_new         = 1

T     = length(y_g)
T_all = length(x)

forecastHorizons = collect(1:length(y_test))

nPredPerIter = 10
statTrans    = ztrans

log_transform = true


############################################################
# Current SARMA forecast restrictions
############################################################

@assert algoSettings.state_var_type == :DSP
@assert algoSettings.obs_var_type   == :static

if algoSettings.scaling == :none
    scaling = false
else
    error(
        "Current SARMA return does not contain S_T. " *
        "For scaled forecasting, return Svec[:, :, T] as well."
    )
end

S_T = nothing


############################################################
# AR and MA lag structures
############################################################

activeLags_ar = FindActiveLagsMultiSAR(
    p1,
    s1
)

activeLags_ma = FindActiveLagsMultiSAR(
    p2,
    s2
)

maxlag_ar = maximum(activeLags_ar)
maxlag_ma = maximum(activeLags_ma)


############################################################
# Initial AR History at Forecast Origin
#
# zₜall_orig =
# [y_T, y_{T-1}, y_{T-2}, ...]
############################################################

zₜall_orig = reverse(
    copy(
        x[
            T_all - maxlag_ar + 1:T_all
        ]
    )
)


############################################################
# Initial MA Error History at Forecast Origin
#
# errors_mx is:
#
# (T_all + p_max[2]) × 1 × freeze_iter
#
# We use the frozen median error trajectory.
############################################################

errors_mx = SARMA_res[6]

errors_tail = @view errors_mx[
    end - maxlag_ma + 1:end,
    1,
    :
]

errors_tail_med = vec(
    median(
        errors_tail;
        dims = 2
    )
)

# Required order:
#
# [ε_T, ε_{T-1}, ε_{T-2}, ...]
#
eₜall_orig = reverse(
    errors_tail_med
)


############################################################
# Optional: Presample AR posterior
#
# This is NOT required for forecasting.
# It is only useful for diagnostics of inferred presample y's.
############################################################

#y_mx = SARMA_res[7]


############################################################
# Posterior State at Forecast Origin
############################################################

θₜpost = SARMA_res[1][end, :, :]


############################################################
# DSP State-Innovation Variance Parameters
############################################################

Hₜpost =
    SARMA_res[2][end, :, :] .-
    log(nPerGroup_fit) .+
    log(g_new)

ϕpost = SARMA_res[4]

μpost =
    SARMA_res[5] .-
    log(nPerGroup_fit) .+
    log(g_new)


############################################################
# Observation Standard Deviation
############################################################

σₑₜpost = SARMA_res[3]

yPreds, LPS_est, AE = PredLocalMultiSARMA_gr(
    nPredPerIter,
    y_test,

    # AR and MA histories
    zₜall_orig,
    eₜall_orig,

    # AR specification
    p1,
    s1,

    # MA specification
    p2,
    s2,

    statTrans,

    # Posterior quantities
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

    scaling = scaling,
    S_T     = S_T
)

