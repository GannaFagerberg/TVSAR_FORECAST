
# Safe exponential (avoid overflow)
function true_f(x)
    return exp.(clamp.(x, -Inf, 700))
end

# ==========================================================
# Forecasting from posterior (MultiSAR with SV / SVDSP)
# ==========================================================

# thid is supposed to be faster than 2
function PredLocalMultiSAR_SV_gr(
    nPredPerIter, yTest, zₜall_orig,
    pFit, season, statTrans,
    θₜpost, Hₜpost, ϕpost, σ²ₙpost, μpost, σₑₜpost,
    ϕ̃post, μ̃post, h̃post, σ̄²ₙpost, forecastHorizons;
    DSP_label::Bool = true,
    g_fc::Int = 1,
    INTERCEPT::Bool = true
)

    # ------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------
    p, nIter = size(θₜpost)
    activeLags = FindActiveLagsMultiSAR(pFit, season)

    forecastHorizons = forecastHorizons[forecastHorizons .<= length(yTest)]
    maxHorizon = maximum(forecastHorizons)

    yPreds = NaN .* ones(maxHorizon, nIter * nPredPerIter)

    # ------------------------------------------------------------
    # Build cache once
    # ------------------------------------------------------------
    cache = build_sarma_cache(pFit, season, activeLags; T=Float64)

    # preallocate regression coefficients
    ϕ̃reg = zeros(Float64, length(activeLags))

    # ------------------------------------------------------------
    # Main posterior loop
    # ------------------------------------------------------------
    for i = 1:nIter
        for j = 1:nPredPerIter

            # ----------------------------------------------------
            # Initialize states at time T
            # ----------------------------------------------------
            zₜall = copy(zₜall_orig)

            θ = copy(θₜpost[:, i])
            h = copy(Hₜpost[:, i])

            if DSP_label 
              μ = copy(μpost[:, i])
            ϕ = copy(ϕpost[:, i])
            end
            
            σθ = similar(h)
            σₑₜ = σₑₜpost[i]

            σₙ = 1.0

            # ----------------------------------------------------
            # Measurement variance initialization
            # ----------------------------------------------------
            if SV
                h̃    = log(σₑₜpost[i]^2)
                μ̃    = μ̃post[i]
                ϕ̃    = ϕ̃post[i]
                σ̄²ₙ  = σ̄²ₙpost[i]

            elseif SVDSP
                hstar = log(σₑₜpost[i]^2)
                h̃    = h̃post[i]
                μ̃    = μ̃post[i]
                ϕ̃    = ϕ̃post[i]
            end

            # ====================================================
            # Forecast horizon loop
            # ====================================================
            for hor = 1:maxHorizon

                # ------------------------------------------------
                # Update parameters only every g_fc observations
                # ------------------------------------------------
                if hor == 1 || (hor - 1) % g_fc == 0

                    if DSP_label && !H_freeze

                        @inbounds for k in eachindex(h)

                            δ_k = (INTERCEPT && k == 1) ? 0.0001 : 0.0001
                            σₙk = (INTERCEPT && k == 1) ? 1.0 : 1.0

                            phi_eff = ϕ[k]^g_fc
                            phi_k   = ϕ[k]
                       
                            if g_fc == 0
                                # Normal approximation to convolution of independent Z innovations
                                σ2_eff = σₙk^2 * grouped_normal_variance(phi_k, g_fc)
                                η = rand(Normal(0.0, sqrt(σ2_eff)))
                            else
                                # PG/Z representation
                                #cg_k = cg_sum(ϕ[k], g_fc)
                                cg_k  = grouped_Z_scale(ϕ[k], g_fc)

                                ξ = rand(PolyaGammaPSWSampler(1, 0.0))
                                η = rand(Normal(0.0, cg_k * σₙk / sqrt(ξ + δ_k)))
                            end

                            h[k] = μ[k] + phi_eff * (h[k] - μ[k]) + η

                        end
                    end

                    σθ .= true_f.(h ./ 2)

                    # evolve θ
                    if !STATE_fixed
                        if INTERCEPT && intercept_dynamics === :ll
                            θ[1] += θ[2]
                            θ[2] += σθ[1] * randn()

                            if length(θ) > 2
                                @inbounds for k in 3:length(θ)
                                    θ[k] += σθ[k-1] * randn()
                                end
                            end
                        else
                            @inbounds for k in eachindex(θ)
                                θ[k] += σθ[k] * randn()
                            end
                        end
                    end

                    # --------------------------------------------
                    # Recompute regression coeffs ONLY when θ changes
                    # --------------------------------------------
                    if INTERCEPT
                        MultiSARMAtoReg_cached!(
                            ϕ̃reg,
                            @view(θ[startcol:end]),
                            cache;
                            threshold = 0.99,
                            ztrans = statTrans,
                            negative_signs = true
                        )
                    else
                        MultiSARMAtoReg_cached!(
                            ϕ̃reg,
                            θ,
                            cache;
                            threshold = 0.99,
                            ztrans = statTrans,
                            negative_signs = true
                        )
                    end
                end

                # --------------------------------------------
                # Evolve measurement variance
                # --------------------------------------------
                if SV

                    h̃   = μ̃ + ϕ̃ * (h̃ - μ̃) + sqrt(σ̄²ₙ) * randn()
                    σₑₜ = true_f(h̃ / 2)

                elseif SVDSP

                    ξ_sv = rand(PolyaGammaPSWSampler(1, 0.0))
                    η_sv = rand(Normal(0.0, σₙ / sqrt(ξ_sv + 0.01)))

                    h̃    = μ̃ + ϕ̃ * (h̃ - μ̃) + η_sv
                    hstar += exp(h̃ / 2) * randn()
                    σₑₜ   = true_f(hstar / 2)

                else
                    σₑₜ = σₑₜpost[i]
                end

                # ------------------------------------------------
                # Construct predictive mean without allocating zₜ
                # ------------------------------------------------
                pred_mean = 0.0
                @inbounds @simd for k in eachindex(activeLags)
                    pred_mean += zₜall[activeLags[k]] * ϕ̃reg[k]
                end

                if INTERCEPT
                    pred_mean += θ[1]
                end

                # ------------------------------------------------
                # Draw forecast directly
                # ------------------------------------------------
                yPred = pred_mean + σₑₜ * randn()
                yPreds[hor, j + (i - 1) * nPredPerIter] = yPred

                # ------------------------------------------------
                # Update lag vector in-place
                # ------------------------------------------------
                @inbounds begin
                    for k = length(zₜall):-1:2
                        zₜall[k] = zₜall[k - 1]
                    end
                    zₜall[1] = yPred
                end
            end
        end
    end

    return yPreds
end
