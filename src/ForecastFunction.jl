


# ==========================================================
# Forecasting from posterior (MultiSAR with SV / SVDSP)
# ==========================================================


# ============================================================
# Safe exponential
# ============================================================

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

# ============================================================
# Forecasting from posterior
# ============================================================
### eithout LPS
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

    # Regression coefficients implied by transformed AR states
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

                # Hₜpost is LOG variance:
                #
                #     h_t = log(σ²_{θ,t})
                #
                h = copy(@view Hₜpost[:, i])

                μ = copy(@view μpost[:, i])
                ϕ = copy(@view ϕpost[:, i])

                σθ = similar(h)

            elseif state_var_type == :static

                # Hₜpost is VARIANCE:
                #
                #     Hₜpost = σ²_θ
                #
                Hvar = @view Hₜpost[:, i]

                if any(x -> x < 0.0, Hvar)
                    error("Negative state variance encountered in Hₜpost.")
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

                # Current log observation variance
                h̃ = log(σₑₜpost[i]^2)

                μ̃   = μ̃post[i]
                ϕ̃   = ϕ̃post[i]
                σ̄²ₙ = σ̄²ₙpost[i]

            elseif obs_var_type == :SVDSP

                # Observation log variance RW level
                hstar = log(σₑₜpost[i]^2)

                # DSP log innovation variance
                h̃ = h̃post[i]

                μ̃ = μ̃post[i]
                ϕ̃ = ϕ̃post[i]

            elseif obs_var_type == :static

                # Nothing else needed

            else

                error(
                    "obs_var_type must be :SV, :SVDSP, or :static"
                )

            end


            # =================================================
            # Forecast horizon loop
            # =================================================

            for hor in 1:maxHorizon

                # =================================================
                # Update parameter states every g_fc observations
                # =================================================

                if hor == 1 || (hor - 1) % g_fc == 0


                    # =================================================
                    # DSP log state-variance evolution
                    # =================================================

                    if state_var_type == :DSP

                        if !H_freeze

                            @inbounds for k in eachindex(h)

                                phi_k   = ϕ[k]
                                phi_eff = phi_k^g_fc

                                # Independent within-group shocks
                                cg_k = grouped_Z_scale(
                                    phi_k,
                                    g_fc
                                )

                                # DSP scale.
                                #
                                # If σ²ₙ is fixed to 1 in estimation,
                                # σ²ₙpost should contain ones.
                                #σₙk = sqrt(σ²ₙpost[k, i])
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
                        # h is LOG VARIANCE
                        #
                        # σθ = exp(h/2)
                        # -----------------------------------------

                        @inbounds for k in eachindex(h)
                            σθ[k] = true_f(h[k] / 2)
                        end

                    end


                    # =================================================
                    # Evolve coefficient states θ
                    # =================================================

                    if !STATE_fixed

                        if INTERCEPT &&
                           intercept_dynamics === :ll

                            # -----------------------------------------
                            # Local linear trend:
                            #
                            # c_t = c_{t-1} + d_{t-1}
                            # d_t = d_{t-1} + innovation
                            #
                            # h does NOT correspond to c_t.
                            # -----------------------------------------

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

                            # -----------------------------------------
                            # Random-walk state evolution
                            # -----------------------------------------

                            @inbounds for k in eachindex(θ)

                                θ[k] +=
                                    σθ[k] *
                                    randn()

                            end

                        end
                    end


                    # =================================================
                    # Transform partial states → SAR coefficients
                    # =================================================

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
                # Observation variance evolution
                # =================================================

                if obs_var_type == :SV

                    h̃ =
                        μ̃ +
                        ϕ̃ * (h̃ - μ̃) +
                        sqrt(σ̄²ₙ) *
                        randn()

                    σₑₜ = true_f(h̃ / 2)


                elseif obs_var_type == :SVDSP

                    # ---------------------------------------------
                    # DSP evolution for observation volatility
                    # ---------------------------------------------

                    ξ_sv = rand(
                        PolyaGammaPSWSampler(
                            1,
                            0.0
                        )
                    )

                    # Currently fixed DSP scale = 1
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

                    # Log variance random walk
                    hstar +=
                        true_f(h̃ / 2) *
                        randn()

                    σₑₜ = true_f(hstar / 2)


                elseif obs_var_type == :static

                    σₑₜ = σₑₜpost[i]

                end


                # =================================================
                # Predictive conditional mean
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
                # Draw forecast
                # =================================================

                yPred =
                    pred_mean +
                    σₑₜ *
                    randn()

                col =
                    j +
                    (i - 1) *
                    nPredPerIter

                yPreds[hor, col] = yPred


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


    return yPreds
end


