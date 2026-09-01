


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

        S_T === nothing && error("scaling=true requires the terminal scaling matrix S_T.")

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

    isempty(forecastHorizons) && error("No forecast horizon lies within length(yTest).")
    maxHorizon = maximum(forecastHorizons)
    yPreds = fill(NaN,maxHorizon,nIter * nPredPerIter)

    # ========================================================
    # Build SAR cache once
    # ========================================================

    cache = build_sarma_cache( pFit,season,activeLags;T = Float64)
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
                                ξ = rand(PolyaGammaPSWSampler(1,0.0))
                                η = rand(Normal(0.0,cg_k *σₙk /sqrt(ξ + δ_k)))
                                h[k] =μ[k] +phi_eff *(h[k] - μ[k]) +η
                            end
                        end


                        # -----------------------------------------
                        # h is log variance in standardized
                        # innovation coordinates:
                        #
                        #     σθ[k] = exp(h[k]/2)
                        # -----------------------------------------

                        @inbounds for k in eachindex(h)
                            σθ[k] =true_f(h[k] / 2)
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
                            mul!(Δθ,S_T,ν_std)
                        else
                            copyto!(Δθ,ν_std)
                        end


                        # -----------------------------------------
                        # Update coefficient states
                        # -----------------------------------------
                        if INTERCEPT &&intercept_dynamics === :ll
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
                    h̃ =μ̃ +ϕ̃ * (h̃ - μ̃) +sqrt(σ̄²ₙ) * randn()
                    σₑₜ =true_f(h̃ / 2)
                elseif obs_var_type == :SVDSP
                    ξ_sv = rand(PolyaGammaPSWSampler(1,0.0))
                    η_sv = rand(Normal(0.0,1.0 /sqrt(ξ_sv + 0.01)))
                    h̃ = μ̃ +ϕ̃ * (h̃ - μ̃) +η_sv
                    hstar +=true_f(h̃ / 2) *randn()
                    σₑₜ =true_f(hstar / 2)
                elseif obs_var_type == :static
                    σₑₜ =σₑₜpost[i]
                end

                # =================================================
                # Predictive mean
                # =================================================
                pred_mean = 0.0
                @inbounds @simd for k in eachindex(activeLags)
                    pred_mean +=zₜall[activeLags[k]] *ϕ̃reg[k]
                end

                if INTERCEPT
                    pred_mean += θ[1]
                end

                # =================================================
                # Predictive distribution
                # =================================================
                predDist = Normal(pred_mean,σₑₜ)
                yPred =rand(predDist)
                # =================================================
                # Store predictive draw
                # =================================================
                col =j +(i - 1) * nPredPerIter
                yPreds[hor, col] =yPred

                # =================================================
                # LPS -- EXACTLY AS REFERENCE FUNCTION
                # =================================================

                if hor ∈ forecastHorizons
                    count += 1
                    LPS[count] +=pdf(predDist,yTest[hor])
                end

                # =================================================
                # Update lag history
                # =================================================
                @inbounds begin
                    for k in length(zₜall):-1:2
                        zₜall[k] =zₜall[k - 1]
                    end
                    zₜall[1] =yPred
                end
            end
        end
    end


    # ========================================================
    # LPS -- same as reference
    # ========================================================
    nh =length(forecastHorizons)
    LPS[1:nh] .=log.(LPS[1:nh] /(nIter * nPredPerIter))
    LPS[(nh + 1):end] .=NaN

    # ========================================================
    # Posterior predictive median
    # ========================================================
    medianPred =vec(median(yPreds,dims = 2))

    # ========================================================
    # Forecast errors
    # ========================================================

    errors = medianPred[forecastHorizons] .-yTest[forecastHorizons]

    # ========================================================
    # MAE contribution -- EXACTLY same as reference
    # ========================================================
    MAE[1:nh] .=abs.(errors)
    return yPreds, LPS, MAE
end

# ============================================================
# Safe exponential
# ============================================================

#@inline true_f(x) = exp(min(x, 700.0))

### Forecast with SMA

function PredLocalMultiSMA_gr(
    nPredPerIter,
    yTest,
    eₜall_orig,
    pFit,
    season,
    statTrans,
    θₜpost,
    Hₜpost,
    ϕpost,
    μpost,
    σₑₜpost,
    forecastHorizons;
    g_fc::Int = 1,
    INTERCEPT::Bool = true,
    H_freeze::Bool = false,
    STATE_fixed::Bool = false,
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
    nInnov = size(Hₜpost, 1)

    # ========================================================
    # Scaling
    # ========================================================

    if scaling && !STATE_fixed

        S_T === nothing && error("scaling=true requires S_T.")

        size(S_T, 1) == nInnov &&
        size(S_T, 2) == nInnov ||
            error(
                "S_T must have size ($nInnov, $nInnov), " *
                "but has size $(size(S_T))."
            )
    end


    # ========================================================
    # Active MA lags
    # ========================================================

    activeLags = FindActiveLagsMultiSAR(pFit,season)
    maxlag_ma = maximum(activeLags)
    length(eₜall_orig) >= maxlag_ma ||
        error(
            "Initial MA error vector must contain at least " *
            "$maxlag_ma errors."
        )
    # ========================================================
    # Forecast horizons
    # ========================================================
    forecastHorizons = forecastHorizons[
        (forecastHorizons .>= 1) .&
        (forecastHorizons .<= length(yTest))
    ]
    isempty(forecastHorizons) &&
        error("No valid forecast horizons.")
    maxHorizon = maximum(forecastHorizons)

    # ========================================================
    # Storage
    # ========================================================
    LPS = zeros(length(forecastHorizons))
    MAE = fill(NaN, length(forecastHorizons))
    yPreds = fill(NaN,maxHorizon,nIter * nPredPerIter)

    # ========================================================
    # MA polynomial cache
    # ========================================================
    cache = build_sarma_cache(pFit,season,activeLags;T = Float64)
    ψ̃reg = zeros(Float64, length(activeLags))

    # ========================================================
    # State innovation buffers
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

            eₜall = copy(eₜall_orig)
            θ = copy(@view θₜpost[:, i])
            h = copy(@view Hₜpost[:, i])
            μ = copy(@view μpost[:, i])
            ϕ = copy(@view ϕpost[:, i])
            σθ = similar(h)
            σₑₜ = σₑₜpost[i]

            # =================================================
            # Forecast horizon
            # =================================================

            count = 0
            for hor in 1:maxHorizon

                # =================================================
                # Update parameter states
                # =================================================
                if hor == 1 || (hor - 1) % g_fc == 0

                    # =============================================
                    # DSP evolution
                    # =============================================

                    if !H_freeze

                        @inbounds for k in eachindex(h)
                            phi_k   = ϕ[k]
                            phi_eff = phi_k^g_fc
                            cg_k = grouped_Z_scale(phi_k,g_fc)
                            ξ = rand(PolyaGammaPSWSampler(1,0.0))
                            η = rand(Normal(0.0,cg_k /sqrt(ξ + 1e-4)))
                            h[k] =μ[k] +phi_eff *(h[k] - μ[k]) +η
                        end
                    end


                    # =============================================
                    # State innovation SD
                    # =============================================

                    @inbounds for k in eachindex(h)
                        σθ[k] = true_f(h[k] / 2)
                    end

                    # =============================================
                    # Evolve coefficient states
                    # =============================================

                    if !STATE_fixed

                        randn!(ν_std)

                        @inbounds for k in eachindex(ν_std)
                            ν_std[k] *= σθ[k]
                        end

                        if scaling
                            mul!(Δθ, S_T, ν_std)
                        else
                            copyto!(Δθ, ν_std)
                        end

                        # -----------------------------------------
                        # Local-linear intercept
                        # -----------------------------------------

                        if INTERCEPT &&
                           intercept_dynamics === :ll

                            θ[1] += θ[2]

                            @inbounds for k in eachindex(Δθ)
                                θ[k + 1] += Δθ[k]
                            end

                        else

                            @inbounds for k in eachindex(Δθ)
                                θ[k] += Δθ[k]
                            end
                        end
                    end


                    # =============================================
                    # Transform latent states to MA coefficients
                    # =============================================

                    if INTERCEPT

                        MultiSARMAtoReg_cached!(
                            ψ̃reg,
                            @view(θ[startcol:end]),
                            cache;
                            threshold = p_threshold,
                            ztrans = statTrans,

                            # IMPORTANT:
                            # same convention as SMA fitting
                            negative_signs = false
                        )

                    else

                        MultiSARMAtoReg_cached!(
                            ψ̃reg,
                            θ,
                            cache;
                            threshold = p_threshold,
                            ztrans = statTrans,
                            negative_signs = false
                        )
                    end
                end


                # =================================================
                # Conditional predictive mean
                #
                # y_t =
                #     c_t
                #     + ψ_1,t e_{t-1}
                #     + ...
                #     + e_t
                # =================================================

                pred_mean = 0.0

                @inbounds @simd for k in eachindex(activeLags)

                    pred_mean +=
                        eₜall[activeLags[k]] *
                        ψ̃reg[k]
                end


                if INTERCEPT
                    pred_mean += θ[1]
                end


                # =================================================
                # NEW innovation
                # =================================================
                εPred = σₑₜ * randn()
                yPred = pred_mean + εPred

                # =================================================
                # Predictive distribution
                # =================================================
                predDist = Normal(pred_mean,σₑₜ)

                # =================================================
                # Store predictive draw
                # =================================================
                col = j + (i - 1) * nPredPerIter
                yPreds[hor, col] = yPred

                # =================================================
                # LPS
                # =================================================

                if hor ∈ forecastHorizons
                    count += 1
                    LPS[count] +=pdf(predDist,yTest[hor])
                end

                # =================================================
                # Update MA ERROR history
                #
                # IMPORTANT:
                #
                # For SAR:
                #     history[1] = yPred
                #
                # For SMA:
                #     history[1] = εPred
                # =================================================

                @inbounds begin

                    for k in length(eₜall):-1:2
                        eₜall[k] = eₜall[k - 1]
                    end

                    eₜall[1] = εPred
                end

            end
        end
    end


    # ========================================================
    # LPS
    # ========================================================

    nh = length(forecastHorizons)
    LPS[1:nh] .= log.(LPS[1:nh] /(nIter * nPredPerIter))

    # ========================================================
    # Posterior predictive median
    # ========================================================

    medianPred = vec(median(yPreds;dims = 2))

    # ========================================================
    # Forecast errors
    # ========================================================

    errors =medianPred[forecastHorizons] .-yTest[forecastHorizons]
    MAE[1:nh] .= abs.(errors)

    return yPreds, LPS, MAE
end


### Forecats SARMA

function PredLocalMultiSARMA_gr(
    nPredPerIter,
    yTest,

    # Initial histories
    zₜall_orig,
    eₜall_orig,

    # AR specification
    pFit_ar,
    season_ar,

    # MA specification
    pFit_ma,
    season_ma,

    statTrans,

    # Posterior quantities
    θₜpost,
    Hₜpost,
    ϕpost,
    μpost,
    σₑₜpost,

    forecastHorizons;

    g_fc::Int = 1,
    INTERCEPT::Bool = true,
    H_freeze::Bool = false,
    STATE_fixed::Bool = false,
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

    nInnov = size(Hₜpost, 1)


    # ========================================================
    # State-vector layout
    #
    # θ =
    #
    # [ intercept |
    #   AR latent states |
    #   MA latent states ]
    # ========================================================

    nARstate = sum(pFit_ar)
    nMAstate = sum(pFit_ma)

    ar_cols = startcol : (startcol + nARstate - 1)

    ma_start = startcol + nARstate
    ma_cols = ma_start : (ma_start + nMAstate - 1)

    last(ma_cols) <= size(θₜpost, 1) ||
        error(
            "SARMA state layout inconsistent with θₜpost. " *
            "AR columns = $ar_cols, MA columns = $ma_cols, " *
            "but θ has $(size(θₜpost,1)) rows."
        )


    # ========================================================
    # Fisher scaling
    # ========================================================

    if scaling && !STATE_fixed

        S_T === nothing &&
            error(
                "scaling=true requires terminal scaling matrix S_T."
            )

        size(S_T, 1) == nInnov &&
        size(S_T, 2) == nInnov ||
            error(
                "S_T must have size ($nInnov, $nInnov), " *
                "but has size $(size(S_T))."
            )
    end


    # ========================================================
    # Active AR and MA lags
    # ========================================================

    activeLags_ar = FindActiveLagsMultiSAR(
        pFit_ar,
        season_ar
    )

    activeLags_ma = FindActiveLagsMultiSAR(
        pFit_ma,
        season_ma
    )

    maxlag_ar = maximum(activeLags_ar)
    maxlag_ma = maximum(activeLags_ma)


    length(zₜall_orig) >= maxlag_ar ||
        error(
            "zₜall_orig must contain at least $maxlag_ar observations."
        )

    length(eₜall_orig) >= maxlag_ma ||
        error(
            "eₜall_orig must contain at least $maxlag_ma innovations."
        )


    # ========================================================
    # Forecast horizons
    # ========================================================

    forecastHorizons = forecastHorizons[
        (forecastHorizons .>= 1) .&
        (forecastHorizons .<= length(yTest))
    ]

    isempty(forecastHorizons) &&
        error(
            "No forecast horizon lies within length(yTest)."
        )

    maxHorizon = maximum(forecastHorizons)
    nh         = length(forecastHorizons)


    # Map horizon -> position in LPS/MAE
    horizon_to_idx = zeros(
        Int,
        maxHorizon
    )

    for (idx, h) in enumerate(forecastHorizons)
        horizon_to_idx[h] = idx
    end


    # ========================================================
    # Forecast storage
    # ========================================================

    LPS = zeros(nh)

    MAE = fill(
        NaN,
        nh
    )

    yPreds = fill(
        NaN,
        maxHorizon,
        nIter * nPredPerIter
    )


    # ========================================================
    # Build AR and MA caches once
    # ========================================================

    cache_ar = build_sarma_cache(
        pFit_ar,
        season_ar,
        activeLags_ar;
        T = Float64
    )

    cache_ma = build_sarma_cache(
        pFit_ma,
        season_ma,
        activeLags_ma;
        T = Float64
    )


    # Expanded regular-form coefficients
    ϕ̃reg = zeros(
        Float64,
        length(activeLags_ar)
    )

    ψ̃reg = zeros(
        Float64,
        length(activeLags_ma)
    )


    # ========================================================
    # State innovation buffers
    # ========================================================

    ν_std = zeros(
        Float64,
        nInnov
    )

    Δθ = zeros(
        Float64,
        nInnov
    )


    # ========================================================
    # Posterior loop
    # ========================================================

    for i in 1:nIter

        for j in 1:nPredPerIter

            # =================================================
            # Predictive-path column
            # =================================================

            col =
                j +
                (i - 1) * nPredPerIter


            # =================================================
            # Initialize AR history
            #
            # [y_T, y_{T-1}, ...]
            # =================================================

            zₜall = copy(
                zₜall_orig
            )


            # =================================================
            # Initialize MA-error history
            #
            # [ε_T, ε_{T-1}, ...]
            # =================================================

            eₜall = copy(
                eₜall_orig
            )


            # =================================================
            # Initial coefficient state
            # =================================================

            θ = copy(
                @view θₜpost[:, i]
            )


            # =================================================
            # DSP initialization
            # ========================================================

            h = copy(
                @view Hₜpost[:, i]
            )

            μ = copy(
                @view μpost[:, i]
            )

            ϕ = copy(
                @view ϕpost[:, i]
            )

            σθ = similar(h)


            # =================================================
            # Static observation standard deviation
            # =================================================

            σₑₜ = σₑₜpost[i]


            # =================================================
            # Forecast horizon loop
            # =================================================

            for hor in 1:maxHorizon

                # =================================================
                # Evolve state at beginning of forecast group
                # =================================================

                if hor == 1 ||
                   (hor - 1) % g_fc == 0


                    # =============================================
                    # DSP log-variance evolution
                    # =============================================

                    if !H_freeze

                        @inbounds for k in eachindex(h)

                            phi_k =
                                ϕ[k]

                            phi_eff =
                                phi_k^g_fc

                            cg_k =
                                grouped_Z_scale(
                                    phi_k,
                                    g_fc
                                )

                            ξ = rand(
                                PolyaGammaPSWSampler(
                                    1,
                                    0.0
                                )
                            )

                            η = rand(
                                Normal(
                                    0.0,
                                    cg_k /
                                    sqrt(
                                        ξ +
                                        1e-4
                                    )
                                )
                            )

                            h[k] =
                                μ[k] +
                                phi_eff *
                                (
                                    h[k] -
                                    μ[k]
                                ) +
                                η
                        end
                    end


                    # =============================================
                    # Convert log variances to state SDs
                    # =============================================

                    @inbounds for k in eachindex(h)

                        σθ[k] =
                            true_f(
                                h[k] / 2
                            )
                    end


                    # =============================================
                    # Evolve coefficient states
                    # =============================================

                    if !STATE_fixed

                        randn!(
                            ν_std
                        )


                        @inbounds for k in eachindex(ν_std)

                            ν_std[k] *=
                                σθ[k]
                        end


                        # -----------------------------------------
                        # Optional Fisher scaling
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
                        # Update coefficient state
                        # -----------------------------------------

                        if INTERCEPT &&
                           intercept_dynamics === :ll

                            # deterministic level evolution
                            θ[1] += θ[2]

                            # stochastic innovations apply from
                            # slope onward
                            @inbounds for k in eachindex(Δθ)

                                θ[k + 1] +=
                                    Δθ[k]
                            end

                        else

                            @inbounds for k in eachindex(Δθ)

                                θ[k] +=
                                    Δθ[k]
                            end
                        end
                    end


                    # =============================================
                    # Expand AR coefficients
                    # =============================================

                    MultiSARMAtoReg_cached!(
                        ϕ̃reg,
                        @view(θ[ar_cols]),
                        cache_ar;
                        threshold =
                            p_threshold,
                        ztrans =
                            statTrans,
                        negative_signs =
                            true
                    )


                    # =============================================
                    # Expand MA coefficients
                    # =============================================

                    MultiSARMAtoReg_cached!(
                        ψ̃reg,
                        @view(θ[ma_cols]),
                        cache_ma;
                        threshold =
                            p_threshold,
                        ztrans =
                            statTrans,
                        negative_signs =
                            false
                    )
                end


                # =================================================
                # SARMA predictive mean
                # =================================================

                pred_mean = 0.0


                # -------------------------------------------------
                # AR contribution
                # -------------------------------------------------

                @inbounds @simd for k in eachindex(
                    activeLags_ar
                )

                    pred_mean +=
                        zₜall[
                            activeLags_ar[k]
                        ] *
                        ϕ̃reg[k]
                end


                # -------------------------------------------------
                # MA contribution
                # -------------------------------------------------

                @inbounds @simd for k in eachindex(
                    activeLags_ma
                )

                    pred_mean +=
                        eₜall[
                            activeLags_ma[k]
                        ] *
                        ψ̃reg[k]
                end


                # -------------------------------------------------
                # Intercept
                # -------------------------------------------------

                if INTERCEPT

                    pred_mean +=
                        θ[1]
                end


                # =================================================
                # New future innovation
                # =================================================

                εPred =
                    σₑₜ *
                    randn()


                # =================================================
                # New forecast observation
                # =================================================

                yPred =
                    pred_mean +
                    εPred


                # =================================================
                # Store predictive draw
                # =================================================

                yPreds[
                    hor,
                    col
                ] = yPred


                # =================================================
                # LPS
                # =================================================

                idx =
                    horizon_to_idx[
                        hor
                    ]

                if idx != 0

                    predDist =
                        Normal(
                            pred_mean,
                            σₑₜ
                        )

                    LPS[idx] +=
                        pdf(
                            predDist,
                            yTest[hor]
                        )
                end


                # =================================================
                # Update AR history
                #
                # [yPred, y_t, y_{t-1}, ...]
                # =================================================

                @inbounds begin

                    for k in length(zₜall):-1:2

                        zₜall[k] =
                            zₜall[k - 1]
                    end

                    zₜall[1] =
                        yPred
                end


                # =================================================
                # Update MA-error history
                #
                # [εPred, ε_t, ε_{t-1}, ...]
                # =================================================

                @inbounds begin

                    for k in length(eₜall):-1:2

                        eₜall[k] =
                            eₜall[k - 1]
                    end

                    eₜall[1] =
                        εPred
                end
            end
        end
    end


    # ========================================================
    # Log predictive score
    # ========================================================

    LPS .= log.(
        LPS /
        (
            nIter *
            nPredPerIter
        )
    )


    # ========================================================
    # Posterior predictive median
    # ========================================================

    medianPred =
        vec(
            median(
                yPreds;
                dims = 2
            )
        )


    # ========================================================
    # Absolute forecast errors
    # ========================================================

    errors =
        medianPred[
            forecastHorizons
        ] .-
        yTest[
            forecastHorizons
        ]

    MAE .= abs.(
        errors
    )


    return yPreds, LPS, MAE
end

#using PolyaGammaSamplers