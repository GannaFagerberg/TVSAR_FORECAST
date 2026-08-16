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
            true_phi = phi
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
                true_phi = Phi
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
            true_phi = phi
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
                true_phi = Phi
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