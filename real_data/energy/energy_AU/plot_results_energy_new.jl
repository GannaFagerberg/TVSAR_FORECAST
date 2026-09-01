using LaTeXStrings


using Statistics
using StatsBase
using Plots
using Dates
using LaTeXStrings

# ==========================================================
# SAVE PLOTS
# ==========================================================

save_label =false
save_dir = raw"C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST\real_data\energy\NY\plots"

if save_label
    mkpath(save_dir)
end

function maybe_save_plot(filename)
    if save_label
        savefig(joinpath(save_dir, filename))
    end
end

default(
    background_color = :white,
    background_color_subplot = :white,
    framestyle = :axes,   # ✅ keeps x & y axes only
    grid = false
)

# ============================================================
# Plotting resolution
# ============================================================
plot_level = :grouped
# plot_level = :original


# ============================================================
# Time index
# ============================================================

# Hourly observations actually entering the model
timestamp_model = timestamp_train

# Group size used in estimation:
# 24 hours × 30 days = 720 observations
nPerGroup_fit = modelSettings.nPerGroup   # should be 720

# Number of estimated posterior groups
T_group = size(SAR_res[1], 1)

# Number of hourly observations represented by the posterior states
T_original = min(
    length(timestamp_model),
    T_group * nPerGroup_fit
)


# ------------------------------------------------------------
# Group-level dates
# Use midpoint of each 30-day estimation block
# ------------------------------------------------------------

idx_group = Int[]

for g in 1:T_group

    group_start = (g - 1) * nPerGroup_fit + 1
    group_end   = min(g * nPerGroup_fit, length(timestamp_model))

    group_start > length(timestamp_model) && break

    push!(
        idx_group,
        group_start + fld(group_end - group_start, 2)
    )
end

time_group = timestamp_model[idx_group]


# ============================================================
# Helper: select plotting resolution
# ============================================================

function states_for_plot(x)

    if plot_level == :grouped

        return (
            expand_grouped_states(
                x,
                1,
                T_group
            ),
            time_group
        )

    elseif plot_level == :original

        return (
            expand_grouped_states(
                x,
                nPerGroup_fit,
                T_original
            ),
            timestamp_model[1:T_original]
        )

    else
        error("plot_level must be :grouped or :original")
    end
end


# Labels used in titles / filenames
plot_label = plot_level == :grouped ?
             "Grouped" :
             "Original resolution"

file_suffix = plot_level == :grouped ?
              "grouped" :
              "original"


# ============================================================
# Date ticks
# ============================================================

time_plot_ref = plot_level == :grouped ?
                time_group :
                timestamp_model[1:T_original]

first_year = year(first(time_plot_ref))
last_year  = year(last(time_plot_ref))

tick_years = collect(
    ceil(Int, first_year / 1) * 1 : 1 : last_year
)

xticks_custom = (
    [DateTime(y, 1, 1) for y in tick_years],
    string.(tick_years)
)

# ============================================================
# Intercept
# ============================================================

if INTERCEPT

    state_plot, time_plot = states_for_plot(
        SAR_res[1][:, 1:1, :]
    )

    summarize_and_plot_t(
        state_plot;
        ylim = (-0.1, 0.1),
        prefix = "Intercept",
        xindex = time_plot,
        xticks = xticks_custom
    )

    maybe_save_plot(
        "intercept_$(file_suffix).pdf"
    )
end

if model_type == :SAR

    # ========================================================
    # AR BLOCK
    # ========================================================

    # --------------------------------------------------------
    # Regular AR coefficients
    # --------------------------------------------------------

    p_reg = p1[1]

    trans_AR_reg = transform_theta(
        SAR_res[1][:, startcol:(startcol + p_reg - 1), :],
        ztrans = ztrans,
        negative_signs = true
    )

    for j in 1:p_reg

        state_plot, time_plot = states_for_plot(
            trans_AR_reg[:, j:j, :]
        )

        summarize_and_plot_t(
            state_plot;
            ylim = (-2, 2),
            prefix = latexstring("\\phi_{$j,t}"),
            xindex = time_plot,
            xticks = xticks_custom
        )

        maybe_save_plot(
            "phi_$(j)_$(file_suffix).pdf"
        )
    end


    # --------------------------------------------------------
    # Seasonal AR coefficients
    # --------------------------------------------------------

    seasonal_startcol = startcol + p1[1]

    for k in 2:length(s1)

        p_seas = p1[k]
        s      = s1[k]

        cols = seasonal_startcol:(
            seasonal_startcol + p_seas - 1
        )

        trans_AR_seas = transform_theta(
            SAR_res[1][:, cols, :],
            ztrans = ztrans,
            negative_signs = true
        )

        for j in 1:p_seas

            state_plot, time_plot = states_for_plot(
                trans_AR_seas[:, j:j, :]
            )

            summarize_and_plot_t(
                state_plot;
                ylim = (-1, 1),
                prefix = latexstring("\\Phi_{$j,t}^{($s)}"),
                xindex = time_plot,
                xticks = xticks_custom
            )

            maybe_save_plot(
                "Phi_$(j)_s$(s)_$(file_suffix).pdf"
            )
        end

        seasonal_startcol += p_seas
    end

    elseif model_type == :SMA

    # ========================================================
    # MA BLOCK
    # ========================================================

    # --------------------------------------------------------
    # Regular MA coefficients
    # --------------------------------------------------------

    p_reg = p2[1]

    trans_MA_reg = transform_theta(
        SAR_res[1][:, startcol:(startcol + p_reg - 1), :],
        ztrans = ztrans,
        negative_signs = false
    )

    for j in 1:p_reg

        state_plot, time_plot = states_for_plot(
            trans_MA_reg[:, j:j, :]
        )

        summarize_and_plot_t(
            state_plot;
            ylim = (-1, 1),
            prefix = latexstring("\\psi_{$j,t}"),
            xindex = time_plot,
            xticks = xticks_custom
        )

        maybe_save_plot(
            "psi_$(j)_$(file_suffix).pdf"
        )
    end


    # --------------------------------------------------------
    # Seasonal MA coefficients
    # --------------------------------------------------------

    seasonal_startcol = startcol + p2[1]

    for k in 2:length(s2)

        p_seas = p2[k]
        s      = s2[k]

        cols = seasonal_startcol:(
            seasonal_startcol + p_seas - 1
        )

        trans_MA_seas = transform_theta(
            SAR_res[1][:, cols, :],
            ztrans = ztrans,
            negative_signs = false
        )

        for j in 1:p_seas

            state_plot, time_plot = states_for_plot(
                trans_MA_seas[:, j:j, :]
            )

            summarize_and_plot_t(
                state_plot;
                ylim = (-1, 1),
                prefix = latexstring("\\Psi_{$j,t}^{($s)}"),
                xindex = time_plot,
                xticks = xticks_custom
            )

            maybe_save_plot(
                "Psi_$(j)_s$(s)_$(file_suffix).pdf"
            )
        end

        seasonal_startcol += p_seas
    end

    elseif model_type == :SARMA

    # ========================================================
    # SARMA
    # ========================================================

    # ========================================================
    # AR BLOCK
    # ========================================================

    ar_startcol = startcol


    # --------------------------------------------------------
    # Regular AR coefficients
    # --------------------------------------------------------

    p_reg_ar = p1[1]

    trans_AR_reg = transform_theta(
        SAR_res[1][
            :,
            ar_startcol:(ar_startcol + p_reg_ar - 1),
            :
        ],
        ztrans = ztrans,
        negative_signs = true
    )

    for j in 1:p_reg_ar

        state_plot, time_plot = states_for_plot(
            trans_AR_reg[:, j:j, :]
        )

        summarize_and_plot_t(
            state_plot;
            ylim = (-1, 1),
            prefix = latexstring("\\phi_{$j,t}"),
            xindex = time_plot,
            xticks = xticks_custom
        )

        maybe_save_plot(
            "phi_$(j)_sarma_$(file_suffix).pdf"
        )
    end


    # --------------------------------------------------------
    # Seasonal AR coefficients
    # --------------------------------------------------------

    seasonal_startcol = ar_startcol + p1[1]

    for k in 2:length(s1)

        p_seas = p1[k]
        s      = s1[k]

        cols = seasonal_startcol:(
            seasonal_startcol + p_seas - 1
        )

        trans_AR_seas = transform_theta(
            SAR_res[1][:, cols, :],
            ztrans = ztrans,
            negative_signs = true
        )

        for j in 1:p_seas

            state_plot, time_plot = states_for_plot(
                trans_AR_seas[:, j:j, :]
            )

            summarize_and_plot_t(
                state_plot;
                ylim = (-1, 1),
                prefix = latexstring("\\Phi_{$j,t}^{($s)}"),
                xindex = time_plot,
                xticks = xticks_custom
            )

            maybe_save_plot(
                "Phi_$(j)_s$(s)_sarma_$(file_suffix).pdf"
            )
        end

        seasonal_startcol += p_seas
    end


    # ========================================================
    # MA BLOCK
    # ========================================================

    ma_startcol = startcol + sum(p1)


    # --------------------------------------------------------
    # Regular MA coefficients
    # --------------------------------------------------------

    p_reg_ma = p2[1]

    trans_MA_reg = transform_theta(
        SAR_res[1][
            :,
            ma_startcol:(ma_startcol + p_reg_ma - 1),
            :
        ],
        ztrans = ztrans,
        negative_signs = false
    )

    for j in 1:p_reg_ma

        state_plot, time_plot = states_for_plot(
            trans_MA_reg[:, j:j, :]
        )

        summarize_and_plot_t(
            state_plot;
            ylim = (-1, 1),
            prefix = latexstring("\\psi_{$j,t}"),
            xindex = time_plot,
            xticks = xticks_custom
        )

        maybe_save_plot(
            "psi_$(j)_sarma_$(file_suffix).pdf"
        )
    end


    # --------------------------------------------------------
    # Seasonal MA coefficients
    # --------------------------------------------------------

    seasonal_startcol = ma_startcol + p2[1]

    for k in 2:length(s2)

        p_seas = p2[k]
        s      = s2[k]

        cols = seasonal_startcol:(
            seasonal_startcol + p_seas - 1
        )

        trans_MA_seas = transform_theta(
            SAR_res[1][:, cols, :],
            ztrans = ztrans,
            negative_signs = false
        )

        for j in 1:p_seas

            state_plot, time_plot = states_for_plot(
                trans_MA_seas[:, j:j, :]
            )

            summarize_and_plot_t(
                state_plot;
                ylim = (-1, 1),
                prefix = latexstring("\\Psi_{$j,t}^{($s)}"),
                xindex = time_plot,
                xticks = xticks_custom
            )

            maybe_save_plot(
                "Psi_$(j)_s$(s)_sarma_$(file_suffix).pdf"
            )
        end

        seasonal_startcol += p_seas
    end
end


# ============================================================
# Measurement volatility σₑ,t
# ============================================================

if obs_var_type in (:SV, :SVDSP)

    sd_meas = reshape(
        SAR_res[3],
        size(SAR_res[3], 1),
        1,
        size(SAR_res[3], 2)
    )

    # Transformation back to the observation scale.
    # Keep this factor consistent with the model specification.
    sd_scale = 1.0

    sd_plot_raw =
        sd_meas .* sd_scale ./ sqrt(nPerGroup_fit)

    # Measurement volatility is also estimated at group level,
    # so use the same grouped/original plotting switch.
    state_plot, time_plot = states_for_plot(
        sd_plot_raw
    )

    summarize_and_plot_t(
        state_plot;
        ylim = (0, 7.0),
        prefix = L"\sigma_{e,t}",
        xindex = time_plot,
        xticks = xticks_custom
    )

    maybe_save_plot(
        "sigma_e_$(file_suffix).pdf"
    )


elseif obs_var_type == :static

    histogram(
        SAR_res[3],
        xlabel = L"\sigma_e",
        title = "Observation standard deviation",
        legend = false
    )

    maybe_save_plot(
        "sigma_e_static.pdf"
    )
end

# ============================================================
# Initial values / presample / MA errors
# ============================================================

if model_type == :SAR

    # --------------------------------------------------------
    # Presample observations
    # --------------------------------------------------------

    if !SAR_conditional

        # Position of y_mx depends on observation variance model
        y_mx_plot =
            obs_var_type in (:SV, :SVDSP) ?
            SAR_res[10] :
            SAR_res[6]

        summarize_and_plot_t(
            exp.(y_mx_plot .+ train_mean);
            ylim = (2500, 8000),
            prefix = latexstring("Presample"),
            true_phi =   y_init,
            #true_phi = nothing,
             xindex = time_plot,
            xticks = xticks_custom
        )

        maybe_save_plot(
            "presample_sar.pdf"
        )
    end


elseif model_type == :SMA

    # --------------------------------------------------------
    # Learned MA errors during adaptation
    # SAR_res[6] = errors_mx
    # --------------------------------------------------------

    start1 = 200

    # Do not use T_group here:
    # these are MA errors, not grouped coefficient states.
    indx = size(SAR_res[6], 1)

    summarize_and_plot_t(
        SAR_res[6][start1:indx, :, :];
        ylim = (-10, 10),
        prefix = latexstring("Errors"),
        #true_phi = true_errors[start1:indx]
        true_phi = nothing
    )

    maybe_save_plot(
        "ma_errors_sma.pdf"
    )


elseif model_type == :SARMA

    # --------------------------------------------------------
    # Learned MA errors
    #
    # SARMA return:
    # θpost, Hpost, σₑpost, errors_mx, y_mx
    # --------------------------------------------------------

    start1 = 200

    # Again, these are on the MA-error scale, not the
    # grouped posterior-state scale.
    indx = size(SAR_res[4], 1)

    summarize_and_plot_t(
        SAR_res[4][start1:indx, :, :];
        ylim = (-10, 10),
        prefix = latexstring("MA\\ Errors"),
        true_phi = true_errors[start1:indx]
    )

    maybe_save_plot(
        "ma_errors_sarma.pdf"
    )


    # --------------------------------------------------------
    # AR presample values
    # SAR_res[5] = y_mx
    # --------------------------------------------------------

    summarize_and_plot_t(
        SAR_res[5] .+ x_med;
        ylim = (-1, 20),
        prefix = latexstring("Presample"),
        xindex = nothing
    )

    maybe_save_plot(
        "presample_sarma.pdf"
    )
end


# ============================================================
# Optional diagnostics
# ============================================================

# histogram(SAR_res[5][1, :])
# histogram(SAR_res[4][3, :])


