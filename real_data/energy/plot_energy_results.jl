# ==========================================================
# INTERCEPT
# ==========================================================

using Statistics
using StatsBase
using Plots
using Dates
using LaTeXStrings


default(
    background_color = :white,
    background_color_subplot = :white,
    framestyle = :axes,   # ✅ keeps x & y axes only
    grid = false
)

# Posterior states: one point per estimated 30-day group
T_group = size(SAR_res[1], 1)

# Group size used in estimation
nPerGroup_fit = modelSettings.nPerGroup   # = 720

# Data actually entering the model
timestamp_model = timestamp[169:end]

# Put each posterior state at midpoint of its 30-day estimation block
idx_group = [
    (g - 1) * nPerGroup_fit + cld(nPerGroup_fit, 2)
    for g in 1:T_group
]

time_ind = timestamp_model[idx_group]

years = unique(year.(time_ind))

xticks_custom = (
    [
        time_ind[findfirst(==(yr), year.(time_ind))]
        for yr in years
    ],
    string.(years)
)


#############
nPerGroup=1
T =size( SAR_res[1][:, 1:1, :])[1]
#T =length(x)

if INTERCEPT
    summarize_and_plot_t(
        expand_grouped_states(
            SAR_res[1][:, 1:1, :],
            nPerGroup,
            T
        );
        ylim = (-0.1, 0.1),
        prefix = "Intercept",
        #true_phi = true_intercept,
        xindex = time_ind,
        xticks = xticks_custom
    )
end

# ==========================================================
# AR / MA COEFFICIENTS
# ==========================================================

if model_type == :SAR

    # ======================================================
    # AR BLOCK
    # ======================================================

    # Regular AR coefficients
    p_reg = p1[1]

    trans_AR_reg = transform_theta(
        SAR_res[1][:, startcol:(startcol + p_reg - 1), :],
        ztrans = ztrans,
        negative_signs = true
    )

    for j in 1:p_reg
        summarize_and_plot_t(
            expand_grouped_states(
                trans_AR_reg[:, j:j, :],
                nPerGroup,
                T
            );
            ylim = (-1, 1),
            prefix = latexstring("\\phi_{$j,t}"),
            #true_phi = phi,
            xindex = time_ind,
        xticks = xticks_custom
        )
    end

    # Seasonal AR coefficients
    seasonal_startcol = startcol + p1[1]

    for k in 2:length(s1)

        p_seas = p1[k]
        s      = s1[k]

        cols = seasonal_startcol:(seasonal_startcol + p_seas - 1)

        trans_AR_seas = transform_theta(
            SAR_res[1][:, cols, :],
            ztrans = ztrans,
            negative_signs = true
        )

        for j in 1:p_seas
            summarize_and_plot_t(
                expand_grouped_states(
                    trans_AR_seas[:, j:j, :],
                    nPerGroup,
                    T
                );
                ylim = (-1, 1),
                prefix = latexstring("\\Phi_{$j,t}^{($s)}"),
                #true_phi = Phi,
                xindex = time_ind,
                xticks = xticks_custom
            )
        end

        seasonal_startcol += p_seas
    end

    
elseif model_type == :SMA

    # ======================================================
    # MA BLOCK
    # ======================================================

    # Regular MA coefficients
    p_reg = p2[1]

    trans_MA_reg = transform_theta(
        SAR_res[1][:, startcol:(startcol + p_reg - 1), :],
        ztrans = ztrans,
        negative_signs = false
    )

    for j in 1:p_reg
        summarize_and_plot_t(
            expand_grouped_states(
                trans_MA_reg[:, j:j, :],
                nPerGroup,
                T
            );
            ylim = (-1, 1),
            prefix = latexstring("\\psi_{$j,t}"),
            #true_phi = phi
        )
    end

    # Seasonal MA coefficients
    seasonal_startcol = startcol + p2[1]

    for k in 2:length(s2)

        p_seas = p2[k]
        s      = s2[k]

        cols = seasonal_startcol:(seasonal_startcol + p_seas - 1)

        trans_MA_seas = transform_theta(
            SAR_res[1][:, cols, :],
            ztrans = ztrans,
            negative_signs = false
        )

        for j in 1:p_seas
            summarize_and_plot_t(
                expand_grouped_states(
                    trans_MA_seas[:, j:j, :],
                    nPerGroup,
                    T
                );
                ylim = (-1, 1),
                prefix = latexstring("\\Psi_{$j,t}^{($s)}"),
                #true_phi = Phi
            )
        end

        seasonal_startcol += p_seas
    end


elseif model_type == :SARMA

    # ======================================================
    # AR BLOCK
    # ======================================================

    ar_startcol = startcol

    # ------------------------------------------------------
    # Regular AR coefficients
    # ------------------------------------------------------

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
        summarize_and_plot_t(
            expand_grouped_states(
                trans_AR_reg[:, j:j, :],
                nPerGroup,
                T
            );
            ylim = (-1, 1),
            prefix = latexstring("\\phi_{$j,t}")
            #true_phi = phi_AR
        )
    end


    # ------------------------------------------------------
    # Seasonal AR coefficients
    # ------------------------------------------------------

    seasonal_startcol = ar_startcol + p1[1]

    for k in 2:length(s1)

        p_seas = p1[k]
        s      = s1[k]

        cols = seasonal_startcol:(seasonal_startcol + p_seas - 1)

        trans_AR_seas = transform_theta(
            SAR_res[1][:, cols, :],
            ztrans = ztrans,
            negative_signs = true
        )

        for j in 1:p_seas
            summarize_and_plot_t(
                expand_grouped_states(
                    trans_AR_seas[:, j:j, :],
                    nPerGroup,
                    T
                );
                ylim = (-1, 1),
                prefix = latexstring("\\Phi_{$j,t}^{($s)}")
                #true_phi = Phi_AR
            )
        end

        seasonal_startcol += p_seas
    end


    # ======================================================
    # MA BLOCK
    # ======================================================

    ma_startcol = startcol + sum(p1)

    # ------------------------------------------------------
    # Regular MA coefficients
    # ------------------------------------------------------

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
        summarize_and_plot_t(
            expand_grouped_states(
                trans_MA_reg[:, j:j, :],
                nPerGroup,
                T
            );
            ylim = (-1, 1),
            prefix = latexstring("\\psi_{$j,t}")
            #true_phi = psi
        )
    end


    # ------------------------------------------------------
    # Seasonal MA coefficients
    # ------------------------------------------------------

    seasonal_startcol = ma_startcol + p2[1]

    for k in 2:length(s2)

        p_seas = p2[k]
        s      = s2[k]

        cols = seasonal_startcol:(seasonal_startcol + p_seas - 1)

        trans_MA_seas = transform_theta(
            SAR_res[1][:, cols, :],
            ztrans = ztrans,
            negative_signs = false
        )

        for j in 1:p_seas
            summarize_and_plot_t(
                expand_grouped_states(
                    trans_MA_seas[:, j:j, :],
                    nPerGroup,
                    T
                );
                ylim = (-1, 1),
                prefix = latexstring("\\Psi_{$j,t}^{($s)}")
                #true_phi = Psi
            )
        end

        seasonal_startcol += p_seas
    end

end


# ==========================================================
# MEASUREMENT VOLATILITY σ_{e,t}
# ==========================================================

if obs_var_type in (:SV, :SVDSP)

    sd_meas = reshape(
        SAR_res[3],
        size(SAR_res[3], 1),
        1,
        size(SAR_res[3], 2)
    )

    sd = 1

    summarize_and_plot_t(
        sd_meas .* sd ./ sqrt(nPerGroup);
        ylim = (0, 7.0),
        prefix = L"\sigma_{e,t}",
        #true_phi = true_sd
        #xindex = time_group,
        #xticks = nothing
    )

elseif obs_var_type == :static

    histogram(
        SAR_res[3],
        xlabel = L"\sigma_e",
        title = "Observation standard deviation",
        legend = false
    )

end


# ==========================================================
# INITIAL VALUES / PRESAMPLE / MA ERRORS
# ==========================================================

if model_type == :SAR

    # ------------------------------------------------------
    # Presample observations
    # ------------------------------------------------------

    if !SAR_conditional

        # y_mx position depends on observation variance model
        y_mx_plot =
            obs_var_type in (:SV, :SVDSP) ?
            SAR_res[10] :
            SAR_res[6]

        summarize_and_plot_t(
            exp.(y_mx_plot .+ x_med);
            ylim = (2500, 8000),
            prefix = latexstring("Presample"),
            true_phi = exp.(x_init),
            xindex = nothing,
            xticks = xticks_custom
        )

    end


elseif model_type == :SMA

    # ------------------------------------------------------
    # Learned MA errors during adaptation
    # SAR_res[6] = errors_mx
    # ------------------------------------------------------

    start1 = 200
    indx   = T

    summarize_and_plot_t(
        SAR_res[6][start1:indx, :, :];
        ylim = (-10, 10),
        prefix = latexstring("Errors"),
        true_phi = true_errors[start1:indx]
        #xindex = nothing,
        #xticks = xticks_custom
    )


elseif model_type == :SARMA

    # ------------------------------------------------------
    # SARMA learned MA errors
    # return:
    # θpost, Hpost, σₑpost, errors_mx, y_mx
    # ------------------------------------------------------

    start1 = 200
    indx   = T_all

    summarize_and_plot_t(
        SAR_res[4][start1:indx, :, :];
        ylim = (-10, 10),
        prefix = latexstring("MA\\ Errors"),
        true_phi = true_errors[start1:indx]
    )


    # ------------------------------------------------------
    # SARMA AR presample values
    # SAR_res[5] = y_mx
    # ------------------------------------------------------

    summarize_and_plot_t(
        SAR_res[5] .+ x_med;
        ylim = (-1,20),
        prefix = latexstring("Presample"),
        #true_phi = exp.(x_init),
        xindex = nothing,
        #xticks = xticks_custom
    )

end


# ==========================================================
# MU / PHI DIAGNOSTICS
# ==========================================================

# histogram(SAR_res[5][1, :])
# histogram(SAR_res[4][3, :])