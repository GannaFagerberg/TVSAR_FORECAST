


# ==========================================================
# Forecasting from posterior (MultiSAR with SV / SVDSP)
# ==========================================================

@inline true_f(x) = exp(min(x, 700.0))


function PredLocalMultiSAR_SV_gr(
    nPredPerIter,
    yTest,
    zₜall_orig,
    pFit,
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
    state_var_type::Symbol = :DSP,
    obs_var_type::Symbol = :SV,
    g_fc::Int = 1,
    INTERCEPT::Bool = true,
    H_freeze::Bool = false,
    STATE_fixed::Bool = false,
    use_fourier::Bool = false,
    intercept_dynamics::Union{Symbol,Nothing},
    startcol::Int,
    p_threshold::Float64 = 0.99,
    scaling::Bool = false,
    S_T = nothing
)

    # ========================================================
    # Dimensions
    # ========================================================

    _, nIter = size(θₜpost)

    # Number of stochastic state innovations.
    #
    # RW intercept:
    #   length(H) == length(θ)
    #
    # Local-linear intercept:
    #   length(H) == length(θ) - 1,
    #   since θ[1] is updated deterministically by θ[2].
    #
    nInnov = size(Hₜpost, 1)


    # ========================================================
    # Fisher scaling at forecast origin
    # ========================================================

    if scaling && !STATE_fixed

        S_T === nothing &&
            error("scaling=true requires the terminal scaling matrix S_T.")

        size(S_T, 1) == nInnov &&
        size(S_T, 2) == nInnov ||
            error(
                "S_T must have size ($nInnov, $nInnov), " *
                "but has size $(size(S_T))."
            )

    end


    activeLags = FindActiveLagsMultiSAR(
        pFit,
        season
    )


    # ========================================================
    # Forecast-storage arrays
    # ========================================================

    LPS = zeros(length(forecastHorizons))
    MAE = fill(NaN, length(forecastHorizons))
    SE  = fill(NaN, length(forecastHorizons))

    forecastHorizons = forecastHorizons[
        (forecastHorizons .>= 1) .&
        (forecastHorizons .<= length(yTest))
    ]

    isempty(forecastHorizons) &&
        error("No forecast horizon lies within length(yTest).")

    maxHorizon = maximum(forecastHorizons)

    yPreds = fill(
        NaN,
        maxHorizon,
        nIter * nPredPerIter
    )


    # ========================================================
    # Build SAR cache once
    # ========================================================

    cache = build_sarma_cache(
        pFit,
        season,
        activeLags;
        T = Float64
    )

    ϕ̃reg = zeros(Float64, length(activeLags))


    # ========================================================
    # State-innovation buffers
    #
    # ν_std :
    #   innovation in Fisher-standardized coordinates
    #
    # Δθ :
    #   innovation mapped back to original state coordinates
    #
    # scaling = false:
    #       Δθ = ν_std
    #
    # scaling = true:
    #       Δθ = S_T * ν_std
    # ========================================================

    ν_std = zeros(Float64, nInnov)
    Δθ    = zeros(Float64, nInnov)


    # ========================================================
    # Posterior loop
    # ========================================================

    for i in 1:nIter

        for j in 1:nPredPerIter

            # =================================================
            # Initialize forecast path
            # =================================================

            zₜall = copy(zₜall_orig)

            θ = copy(@view θₜpost[:, i])


            # =================================================
            # State-innovation variance initialization
            # =================================================

            if state_var_type == :DSP

                # H is log variance
                h = copy(@view Hₜpost[:, i])

                μ = copy(@view μpost[:, i])
                ϕ = copy(@view ϕpost[:, i])

                σθ = similar(h)


            elseif state_var_type == :static

                # H is variance
                Hvar = @view Hₜpost[:, i]

                if any(x -> x < 0.0, Hvar)
                    error(
                        "Negative state variance encountered in Hₜpost."
                    )
                end

                σθ = sqrt.(Hvar)


            else

                error(
                    "state_var_type must be :DSP or :static"
                )

            end


            # =================================================
            # Measurement variance initialization
            # =================================================

            σₑₜ = σₑₜpost[i]


            if obs_var_type == :SV

                h̃ = log(σₑₜpost[i]^2)

                μ̃   = μ̃post[i]
                ϕ̃   = ϕ̃post[i]
                σ̄²ₙ = σ̄²ₙpost[i]


            elseif obs_var_type == :SVDSP

                hstar = log(σₑₜpost[i]^2)

                h̃ = h̃post[i]

                μ̃ = μ̃post[i]
                ϕ̃ = ϕ̃post[i]


            elseif obs_var_type == :static

                # nothing else needed


            else

                error(
                    "obs_var_type must be :SV, :SVDSP, or :static"
                )

            end


            # Same logic as reference function
            count = 0


            # =================================================
            # Forecast horizon loop
            # =================================================

            for hor in 1:maxHorizon


                # =================================================
                # Update parameter states
                # =================================================

                if hor == 1 || (hor - 1) % g_fc == 0


                    # =============================================
                    # DSP evolution
                    # =============================================

                    if state_var_type == :DSP

                        if !H_freeze

                            @inbounds for k in eachindex(h)

                                phi_k   = ϕ[k]
                                phi_eff = phi_k^g_fc

                                cg_k = grouped_Z_scale(
                                    phi_k,
                                    g_fc
                                )

                                # fixed σ²_n = 1
                                σₙk = 1.0
                                δ_k = 1e-4

                                ξ = rand(
                                    PolyaGammaPSWSampler(
                                        1,
                                        0.0
                                    )
                                )

                                η = rand(
                                    Normal(
                                        0.0,
                                        cg_k *
                                        σₙk /
                                        sqrt(ξ + δ_k)
                                    )
                                )

                                h[k] =
                                    μ[k] +
                                    phi_eff *
                                    (h[k] - μ[k]) +
                                    η

                            end
                        end


                        # -----------------------------------------
                        # h is log variance in standardized
                        # innovation coordinates:
                        #
                        #     σθ[k] = exp(h[k]/2)
                        # -----------------------------------------

                        @inbounds for k in eachindex(h)

                            σθ[k] =
                                true_f(h[k] / 2)

                        end

                    end


                    # =============================================
                    # Evolve coefficient states
                    # =============================================

                    if !STATE_fixed

                        # -----------------------------------------
                        # Draw innovation in standardized
                        # coordinates:
                        #
                        # ν_std[k] =
                        #     σθ[k] * z[k],
                        #
                        # z[k] ~ N(0,1)
                        # -----------------------------------------

                        randn!(ν_std)

                        @inbounds for k in eachindex(ν_std)

                            ν_std[k] *= σθ[k]

                        end


                        # -----------------------------------------
                        # Fisher scaling
                        #
                        # scaling = false:
                        #
                        #     Δθ = ν_std
                        #
                        # scaling = true:
                        #
                        #     Δθ = S_T * ν_std
                        #
                        # The terminal matrix S_T is kept FIXED
                        # throughout all forecast horizons.
                        # -----------------------------------------

                        if scaling

                            mul!(
                                Δθ,
                                S_T,
                                ν_std
                            )

                        else

                            copyto!(
                                Δθ,
                                ν_std
                            )

                        end


                        # -----------------------------------------
                        # Update coefficient states
                        # -----------------------------------------

                        if INTERCEPT &&
                           intercept_dynamics === :ll

                            # Local-linear intercept:
                            #
                            # θ[1] = level
                            # θ[2] = slope
                            #
                            # Level evolves deterministically:
                            #
                            # c_{t+1} = c_t + d_t
                            #
                            θ[1] += θ[2]


                            # Stochastic innovation applies to
                            #
                            #   [θ[2], θ[3], ..., θ[end]]
                            #
                            # so Δθ has length length(θ)-1.
                            #
                            @inbounds for k in eachindex(Δθ)

                                θ[k + 1] += Δθ[k]

                            end


                        else

                            # RW intercept or no intercept:
                            #
                            # stochastic innovation applies to
                            # every state component.
                            #
                            @inbounds for k in eachindex(Δθ)

                                θ[k] += Δθ[k]

                            end

                        end

                    end


                    # =============================================
                    # Transform to SAR coefficients
                    # =============================================

                    if INTERCEPT

                        MultiSARMAtoReg_cached!(
                            ϕ̃reg,
                            @view(θ[startcol:end]),
                            cache;
                            threshold = p_threshold,
                            ztrans = statTrans,
                            negative_signs = true
                        )

                    else

                        MultiSARMAtoReg_cached!(
                            ϕ̃reg,
                            θ,
                            cache;
                            threshold = p_threshold,
                            ztrans = statTrans,
                            negative_signs = true
                        )

                    end
                end


                # =================================================
                # Measurement variance evolution
                # =================================================

                if obs_var_type == :SV

                    h̃ =
                        μ̃ +
                        ϕ̃ * (h̃ - μ̃) +
                        sqrt(σ̄²ₙ) * randn()

                    σₑₜ =
                        true_f(h̃ / 2)


                elseif obs_var_type == :SVDSP

                    ξ_sv = rand(
                        PolyaGammaPSWSampler(
                            1,
                            0.0
                        )
                    )

                    η_sv = rand(
                        Normal(
                            0.0,
                            1.0 /
                            sqrt(ξ_sv + 0.01)
                        )
                    )

                    h̃ =
                        μ̃ +
                        ϕ̃ * (h̃ - μ̃) +
                        η_sv

                    hstar +=
                        true_f(h̃ / 2) *
                        randn()

                    σₑₜ =
                        true_f(hstar / 2)


                elseif obs_var_type == :static

                    σₑₜ =
                        σₑₜpost[i]

                end


                # =================================================
                # Predictive mean
                # =================================================

                pred_mean = 0.0

                @inbounds @simd for k in eachindex(activeLags)

                    pred_mean +=
                        zₜall[activeLags[k]] *
                        ϕ̃reg[k]

                end

                if INTERCEPT

                    pred_mean += θ[1]

                end


                # =================================================
                # Predictive distribution
                # =================================================

                predDist = Normal(
                    pred_mean,
                    σₑₜ
                )

                yPred =
                    rand(predDist)


                # =================================================
                # Store predictive draw
                # =================================================

                col =
                    j +
                    (i - 1) * nPredPerIter

                yPreds[hor, col] =
                    yPred


                # =================================================
                # LPS -- EXACTLY AS REFERENCE FUNCTION
                # =================================================

                if hor ∈ forecastHorizons

                    count += 1

                    LPS[count] +=
                        pdf(
                            predDist,
                            yTest[hor]
                        )

                end


                # =================================================
                # Update lag history
                # =================================================

                @inbounds begin

                    for k in length(zₜall):-1:2

                        zₜall[k] =
                            zₜall[k - 1]

                    end

                    zₜall[1] =
                        yPred

                end

            end
        end
    end


    # ========================================================
    # LPS -- same as reference
    # ========================================================

    nh =
        length(forecastHorizons)

    LPS[1:nh] .=
        log.(
            LPS[1:nh] /
            (nIter * nPredPerIter)
        )

    LPS[(nh + 1):end] .=
        NaN


    # ========================================================
    # Posterior predictive median
    # ========================================================

    medianPred =
        vec(
            median(
                yPreds,
                dims = 2
            )
        )


    # ========================================================
    # Forecast errors
    # ========================================================

    errors =
        medianPred[forecastHorizons] .-
        yTest[forecastHorizons]


    # ========================================================
    # MAE contribution -- EXACTLY same as reference
    # ========================================================

    MAE[1:nh] .=
        abs.(errors)


    return yPreds, LPS, MAE
end

# ============================================================
# Safe exponential
# ============================================================

@inline true_f(x) = exp(min(x, 700.0))

function PredLocalMultiSAR_SV_gr_ref(
    nPredPerIter,
    yTest,
    zₜall_orig,
    pFit,
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
    state_var_type::Symbol = :DSP,
    obs_var_type::Symbol = :SV,
    g_fc::Int = 1,
    INTERCEPT::Bool = true,
    H_freeze::Bool = false,
    STATE_fixed::Bool = false,
    use_fourier::Bool = false,
    intercept_dynamics::Union{Symbol,Nothing},
    startcol::Int,
    p_threshold::Float64 = 0.99
)

    # ========================================================
    # Dimensions
    # ========================================================

    _, nIter = size(θₜpost)

    activeLags = FindActiveLagsMultiSAR(
        pFit,
        season
    )

    # Same structure as reference function
    LPS = zeros(length(forecastHorizons))
    MAE = fill(NaN, length(forecastHorizons))
    SE  = fill(NaN, length(forecastHorizons))

    forecastHorizons = forecastHorizons[
        (forecastHorizons .>= 1) .&
        (forecastHorizons .<= length(yTest))
    ]

    isempty(forecastHorizons) &&
        error("No forecast horizon lies within length(yTest).")

    maxHorizon = maximum(forecastHorizons)

    yPreds = fill(
        NaN,
        maxHorizon,
        nIter * nPredPerIter
    )


    # ========================================================
    # Build SAR cache once
    # ========================================================

    cache = build_sarma_cache(
        pFit,
        season,
        activeLags;
        T = Float64
    )

    ϕ̃reg = zeros(Float64, length(activeLags))


    # ========================================================
    # Posterior loop
    # ========================================================

    for i in 1:nIter

        for j in 1:nPredPerIter

            # =================================================
            # Initialize forecast path
            # =================================================

            zₜall = copy(zₜall_orig)

            θ = copy(@view θₜpost[:, i])


            # =================================================
            # State-innovation variance initialization
            # =================================================

            if state_var_type == :DSP

                # H is log variance
                h = copy(@view Hₜpost[:, i])

                μ = copy(@view μpost[:, i])
                ϕ = copy(@view ϕpost[:, i])

                σθ = similar(h)

            elseif state_var_type == :static

                # H is variance
                Hvar = @view Hₜpost[:, i]

                if any(x -> x < 0.0, Hvar)
                    error("Negative state variance encountered in Hₜpost.")
                end

                σθ = sqrt.(Hvar)

            else

                error("state_var_type must be :DSP or :static")

            end


            # =================================================
            # Measurement variance initialization
            # =================================================

            σₑₜ = σₑₜpost[i]

            if obs_var_type == :SV

                h̃ = log(σₑₜpost[i]^2)

                μ̃   = μ̃post[i]
                ϕ̃   = ϕ̃post[i]
                σ̄²ₙ = σ̄²ₙpost[i]

            elseif obs_var_type == :SVDSP

                hstar = log(σₑₜpost[i]^2)

                h̃ = h̃post[i]

                μ̃ = μ̃post[i]
                ϕ̃ = ϕ̃post[i]

            elseif obs_var_type == :static

                # nothing else needed

            else

                error("obs_var_type must be :SV, :SVDSP, or :static")

            end


            # Same logic as reference function
            count = 0


            # =================================================
            # Forecast horizon loop
            # ========================================================

            for hor in 1:maxHorizon


                # =================================================
                # Update parameter states
                # =================================================

                if hor == 1 || (hor - 1) % g_fc == 0


                    # =============================================
                    # DSP evolution
                    # =============================================

                    if state_var_type == :DSP

                        if !H_freeze

                            @inbounds for k in eachindex(h)

                                phi_k   = ϕ[k]
                                phi_eff = phi_k^g_fc

                                cg_k = grouped_Z_scale(
                                    phi_k,
                                    g_fc
                                )

                                # fixed σ²_n = 1
                                σₙk = 1.0
                                δ_k = 1e-4

                                ξ = rand(
                                    PolyaGammaPSWSampler(
                                        1,
                                        0.0
                                    )
                                )

                                η = rand(
                                    Normal(
                                        0.0,
                                        cg_k *
                                        σₙk /
                                        sqrt(ξ + δ_k)
                                    )
                                )

                                h[k] =
                                    μ[k] +
                                    phi_eff *
                                    (h[k] - μ[k]) +
                                    η
                            end

                        end


                        # h is log variance
                        @inbounds for k in eachindex(h)
                            σθ[k] = true_f(h[k] / 2)
                        end

                    end


                    # =============================================
                    # Evolve coefficient states
                    # =============================================

                    if !STATE_fixed

                        if INTERCEPT &&
                           intercept_dynamics === :ll

                            θ[1] += θ[2]

                            θ[2] +=
                                σθ[1] * randn()

                            if length(θ) > 2

                                @inbounds for k in 3:length(θ)

                                    θ[k] +=
                                        σθ[k - 1] *
                                        randn()

                                end
                            end

                        else

                            @inbounds for k in eachindex(θ)

                                θ[k] +=
                                    σθ[k] *
                                    randn()

                            end

                        end
                    end


                    # =============================================
                    # Transform to SAR coefficients
                    # =============================================

                    if INTERCEPT

                        MultiSARMAtoReg_cached!(
                            ϕ̃reg,
                            @view(θ[startcol:end]),
                            cache;
                            threshold = p_threshold,
                            ztrans = statTrans,
                            negative_signs = true
                        )

                    else

                        MultiSARMAtoReg_cached!(
                            ϕ̃reg,
                            θ,
                            cache;
                            threshold = p_threshold,
                            ztrans = statTrans,
                            negative_signs = true
                        )

                    end
                end


                # =================================================
                # Measurement variance evolution
                # =================================================

                if obs_var_type == :SV

                    h̃ =
                        μ̃ +
                        ϕ̃ * (h̃ - μ̃) +
                        sqrt(σ̄²ₙ) * randn()

                    σₑₜ = true_f(h̃ / 2)


                elseif obs_var_type == :SVDSP

                    ξ_sv = rand(
                        PolyaGammaPSWSampler(
                            1,
                            0.0
                        )
                    )

                    η_sv = rand(
                        Normal(
                            0.0,
                            1.0 /
                            sqrt(ξ_sv + 0.01)
                        )
                    )

                    h̃ =
                        μ̃ +
                        ϕ̃ * (h̃ - μ̃) +
                        η_sv

                    hstar +=
                        true_f(h̃ / 2) *
                        randn()

                    σₑₜ = true_f(hstar / 2)


                elseif obs_var_type == :static

                    σₑₜ = σₑₜpost[i]

                end


                # =================================================
                # Predictive mean
                # =================================================

                pred_mean = 0.0

                @inbounds @simd for k in eachindex(activeLags)

                    pred_mean +=
                        zₜall[activeLags[k]] *
                        ϕ̃reg[k]

                end

                if INTERCEPT
                    pred_mean += θ[1]
                end


                # =================================================
                # Predictive distribution
                # =================================================

                predDist = Normal(
                    pred_mean,
                    σₑₜ
                )

                yPred = rand(predDist)


                # =================================================
                # Store predictive draw
                # =================================================

                col =
                    j +
                    (i - 1) * nPredPerIter

                yPreds[hor, col] = yPred


                # =================================================
                # LPS -- EXACTLY AS REFERENCE FUNCTION
                # =================================================

                if hor ∈ forecastHorizons

                    count += 1

                    LPS[count] +=
                        pdf(
                            predDist,
                            yTest[hor]
                        )

                end


                # =================================================
                # Update lag history
                # =================================================

                @inbounds begin

                    for k in length(zₜall):-1:2
                        zₜall[k] = zₜall[k - 1]
                    end

                    zₜall[1] = yPred

                end

            end
        end
    end


    # ========================================================
    # LPS -- same as reference
    # ========================================================

    nh = length(forecastHorizons)

    LPS[1:nh] .=
        log.(
            LPS[1:nh] /
            (nIter * nPredPerIter)
        )

    LPS[(nh + 1):end] .= NaN


    # ========================================================
    # Posterior predictive median
    # ========================================================

    medianPred =
        vec(
            median(
                yPreds,
                dims = 2
            )
        )


    # ========================================================
    # Forecast errors
    # ========================================================

    errors =
        medianPred[forecastHorizons] .-
        yTest[forecastHorizons]


    # ========================================================
    # MAE contribution -- EXACTLY same as reference
    # ========================================================

    MAE[1:nh] .=
        abs.(errors)

    return yPreds, LPS, MAE
end

