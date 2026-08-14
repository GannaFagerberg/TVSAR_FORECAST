# --- helper: stable c_g = 1 + ϕ + ... + ϕ^(g-1) (avoids 0/0 near ϕ≈1) ---

### Common shock assumption
@inline function cg_sum(phi, g::Integer)
    s = one(phi)
    p = one(phi)

    @inbounds for _ in 2:g
        p *= phi
        s += p
    end

    return s
end

"""
    grouped_Z_scale(s, phi, g)

Compute the moment-matched scale parameter s_g for the grouped
Z-distribution approximation:

    Z(1/2,1/2,0,s_g)

using

    s_g = s * sqrt((1 - phi^(2g)) / (1 - phi^2))

with a stable computation when phi ≈ ±1.

Arguments
---------
s    : original Z scale parameter
phi  : AR coefficient
g    : group size

Returns
-------
s_g  : grouped scale parameter
"""
### Independent shocks assumption
@inline function grouped_Z_scale(phi::Float64, g::Int)
    sumsq = abs(1 - phi^2) < 1e-10 ? g : (1 - phi^(2g)) / (1 - phi^2)
    return sqrt(sumsq)
end


# --- z buffers for MH step (length T-1) ---
function build_z!(zprev::Vector{Float64}, zcurr::Vector{Float64},
                  H::AbstractVector{<:Real}, ξ::AbstractVector{<:Real}, μ::Real)
    T = length(H)
    @inbounds for t in 2:T
        s = sqrt(float(ξ[t]))
        zprev[t-1] = s * (float(H[t-1]) - μ)
        zcurr[t-1] = s * (float(H[t])   - μ)
    end
    return nothing
end

# --- grouped log-likelihood for original phi (uses phi^g + sigma2 * c_g(phi)^2 ) ---
@inline function loglik_phi_grouped(zprev::Vector{Float64},
                                    zcurr::Vector{Float64},
                                    phi::Float64,
                                    g::Int,
                                    sigma2::Float64)
    if abs(phi) >= 1.0
        return -Inf
    end
    phig = (g == 1) ? phi : phi^g
    cg   = (g == 1) ? 1.0 : cg_sum(phi, g)

    sigma2g = sigma2 * (cg * cg)
    if !(sigma2g > 0.0) || !isfinite(sigma2g)
        return -Inf
    end

    invσ2 = 1.0 / sigma2g
    cst   = -0.5 * log(2π * sigma2g)

    ll = 0.0
    @inbounds @simd for i in eachindex(zcurr)
        r = zcurr[i] - phig * zprev[i]
        ll += cst - 0.5 * r*r * invσ2
    end
    return ll
end

@inline atanh_safe(x::Float64) = 0.5 * log((1 + x) / (1 - x))

function Updateϕ_grouped_MH_transformed!(
    phi::Float64,
    zprev::Vector{Float64},
    zcurr::Vector{Float64},
    sigma2::Float64,
    phi0::Float64,
    kappa0::Float64;
    g::Int = 1,
    proposal_sd_psi::Float64 = 0.05
)
    @inline function logprior_phi(phi::Float64)
        (abs(phi) < 1.0) ? logpdf(Normal(phi0, kappa0), phi) : -Inf
    end

    @inline function logpost_psi(psi::Float64)
        phi = tanh(psi)

        ll = loglik_phi_grouped(zprev, zcurr, phi, g, sigma2)
        if !isfinite(ll)
            return -Inf
        end

        lp = logprior_phi(phi)
        if !isfinite(lp)
            return -Inf
        end

        # Jacobian for phi = tanh(psi)
        logjac = log1p(-phi^2)   # log(1 - phi^2)

        return ll + lp + logjac
    end

    # current transformed value
    psi_cur = atanh_safe(phi)
    lp_cur  = logpost_psi(psi_cur)

    if !isfinite(lp_cur)
        return phi, false
    end

    # RW proposal on unconstrained scale
    psi_prop = psi_cur + proposal_sd_psi * randn()
    lp_prop  = logpost_psi(psi_prop)

    if log(rand()) < (lp_prop - lp_cur)
        return tanh(psi_prop), true
    else
        return phi, false
    end
end

# --- MH update for original phi under grouped likelihood ---
function Updateϕ_grouped_MH!(
    phi::Float64,
    zprev::Vector{Float64},
    zcurr::Vector{Float64},
    sigma2::Float64,
    phi0::Float64,
    kappa0::Float64;
    g::Int = 1,
    proposal_sd::Float64 = 0.05
)
    @inline function logprior(phi::Float64)
        (abs(phi) < 1.0) ? logpdf(Normal(phi0, kappa0), phi) : -Inf
    end

    lp_cur = logprior(phi) + loglik_phi_grouped(zprev, zcurr, phi, g, sigma2)
    if !isfinite(lp_cur)
        # recovery: draw from prior until finite
        for _ in 1:50
            cand = rand(Truncated(Normal(phi0, kappa0), -1, 1))  # OBS!used [0,1]
            lp = logprior(cand) + loglik_phi_grouped(zprev, zcurr, cand, g, sigma2)
            if isfinite(lp)
                return cand, true
            end
        end
        return phi, false
    end

    phi_prop = phi + proposal_sd * randn()
    lp_prop = logprior(phi_prop) + loglik_phi_grouped(zprev, zcurr, phi_prop, g, sigma2)

    if log(rand()) < (lp_prop - lp_cur)
        return phi_prop, true
    else
        return phi, false
    end
end

# ============================================================================
# UPDATED: update_dsp! with grouping that PRESERVES ORIGINAL-SCALE priors on phi
# - Uses phi_eff = phi^g and sigma2_eff = sigma2 * c_g(phi)^2 in Update_h / Updateξ1
# - Updates ORIGINAL phi via MH using grouped likelihood
# - Optionally: disable Updateσ²ₙ when g>1 unless you implement a grouped version
# ============================================================================
function update_dsp_grouped!(
    ν, S, H, H̃, ξ, ϕ, μ, σ²ₙ,
    ϕ₀, κ₀, m₀, σ₀, ν₀, ψ₀,
    mixLogχ²₁, m, v, postDist, Dᵩ;
    g::Int = 1,
    zprev_buf::Vector{Float64},
    zcurr_buf::Vector{Float64},
    prop_sd_phi::AbstractVector{<:Real},
    acc_phi::AbstractVector{<:Integer},
    offset = eps(),
    updateσₙ::Bool = false,
    α::Float64 = 1/2,
    β::Float64 = 1/2,
    INTERCEPT::Bool = false
)

    p = size(ν, 2)

    # --------------------------------------------------
    # Grouped state innovation:
    #
    #   ν_b² ≈ g exp(h_bg) χ₁²
    #
    # Hence subtract log(g) so H and μ remain on the
    # ORIGINAL fine-scale h parametrization.
    # --------------------------------------------------
    p = size(ν, 2)

    Ỹ = similar(ν)
    @inbounds @simd for j in eachindex(ν)
        Ỹ[j] = log(ν[j]^2 + eps())
    end

    @views for k in 1:p

        # ==================================================
        # Current ORIGINAL phi
        # ==================================================
        phi_k = float(ϕ[k])

        # grouped AR coefficient
        phi_eff = phi_k^g

        # Independent fine-scale Z shocks:
        #
        # A_g(phi) = sum_{j=0}^{g-1} phi^(2j)
        #
        # sigma²_eff = sigma² * A_g(phi)
        #
        scale_g     = grouped_Z_scale(phi_k, g)
        scale2_g    = scale_g^2
        sigma2_eff  = float(σ²ₙ[k]) * scale2_g

        # --------------------------------------------------
        # mixture allocation
        # --------------------------------------------------
        S[:, k] = UpdateMixAlloc(Ỹ[:, k],H̃[:, k] .+ μ[k],mixLogχ²₁)

        # --------------------------------------------------
        # H | ...
        # --------------------------------------------------
        H[:, k] = Update_h(
            Ỹ[:, k],
            m[S[:, k]],
            v[S[:, k]],
            Dᵩ,
            ξ[:, k],
            phi_eff,
            sigma2_eff,
            μ[k]
        )

        # --------------------------------------------------
        # ξ | ...
        # --------------------------------------------------
        δ = 0.0001

        ξ[:, k] = Updateξ1(
            H[:, k],
            phi_eff,
            sigma2_eff,
            μ[k],
            α,
            β;
            δ = δ
        )

        # ==================================================
        # Update ORIGINAL phi using grouped likelihood
        # ==================================================
        build_z!(
            zprev_buf,
            zcurr_buf,
            H[:, k],
            ξ[:, k],
            μ[k]
        )

        ϕ_new, acc = Updateϕ_grouped_MH!(
            phi_k,
            zprev_buf,
            zcurr_buf,
            float(σ²ₙ[k]),
            float(ϕ₀),
            float(κ₀);
            g = g,
            proposal_sd = float(prop_sd_phi[k])
        )

        ϕ[k] = ϕ_new
        acc_phi[k] += acc ? 1 : 0

        # ==================================================
        # IMPORTANT:
        # recompute grouped quantities using NEW phi
        # ==================================================
        phi_k    = float(ϕ[k])
        phi_eff  = phi_k^g

        scale_g  = grouped_Z_scale(phi_k, g)
        scale2_g = scale_g^2

        # --------------------------------------------------
        # Optional ORIGINAL-scale sigma² update
        # --------------------------------------------------
        if updateσₙ

            if g == 1

                σ²ₙ[k] = Updateσ²ₙ(
                    H[:, k],
                    ξ[:, k],
                    phi_k,
                    μ[k],
                    ν₀,
                    ψ₀
                )

            else

                # This function must use
                #
                # A_g(phi) = scale_g²
                #
                # rather than the old shared-shock
                # c_g(phi)².
                σ²ₙ[k] = Updateσ²ₙ_grouped(
                    H[:, k],
                    ξ[:, k],
                    phi_eff,
                    scale_g,
                    μ[k],
                    ν₀,
                    ψ₀
                )
            end
        end

        # Effective variance must also reflect a possible
        # newly sampled ORIGINAL sigma².
        sigma2_eff = float(σ²ₙ[k]) * scale2_g

        # --------------------------------------------------
        # μ remains on ORIGINAL fine-scale h scale
        # --------------------------------------------------
        μ[k] = Updateμ(
            H[:, k],
            ξ[:, k],
            phi_eff,
            sigma2_eff,
            m₀,
            σ₀
        )

        @. H̃[:, k] = H[:, k] - μ[k]
    end

    return nothing
end


function Updateσ²ₙ_grouped(y, ξ, phi_eff, cg, μ, ν₀, ψ₀)

    T = length(y)
    inv_cg = 1.0 / cg

    sumsq = 0.0

    s = sqrt(ξ[1]) * (y[1] - μ)
    sumsq += s*s

    @inbounds @simd for t in 2:T
        u = y[t] - μ - phi_eff * (y[t-1] - μ)
        s = sqrt(ξ[t]) * (u * inv_cg)
        sumsq += s*s
    end

    νT = ν₀ + T
    scale = (ν₀ * ψ₀^2 + sumsq) / νT

    # truncated draw ensuring σ²ₙ < 1
    while true
        σ2 = rand(ScaledInverseChiSq(νT, scale))
        σ2 < 1 && return σ2
    end
end


"""
    grouped_normal_variance(phi, g;
                            alpha=0.5,
                            beta=0.5,
                            scale=1.0)

Compute the variance of the Gaussian approximation to

    Zg = Z1 + phi*Z2 + ... + phi^(g-1)*Zg

where Zi ~ Z(alpha,beta,0,scale).

Returns:
    Var(Zg)
"""
@inline function grouped_normal_variance(phi::Float64,
                                         g::Int;
                                         alpha::Float64=0.5,
                                         beta::Float64=0.5,
                                         scale::Float64=1.0)

    # variance of one Z variable
    varZ = scale^2 * (trigamma(alpha) + trigamma(beta))

    # stable geometric sum
    sumsq =
        abs(1 - phi^2) < 1e-10 ?
        g :
        (1 - phi^(2g)) / (1 - phi^2)

    return varZ * sumsq
end

######################

struct LogChi2Mix10
    logω::NTuple{10,Float64}
    m::NTuple{10,Float64}
    invv::NTuple{10,Float64}
    lognorm::NTuple{10,Float64}
end

function prepare_logchi2_mix10(ω, m, v)
    logω    = ntuple(i -> log(ω[i]), 10)
    #logω    = ntuple(i -> ω[i], 10)
    mT      = ntuple(i -> m[i], 10)
    invv    = ntuple(i -> 1 / v[i], 10)
    lognorm = ntuple(i -> -0.5 * log(2π * v[i]), 10)
    return LogChi2Mix10(logω, mT, invv, lognorm)
end

function UpdateMixAlloc(ỹ, h, mix::LogChi2Mix10)
    T = length(ỹ)
    S = Vector{Int}(undef, T)

    @inbounds for t in 1:T
        r = ỹ[t] - h[t]

        # log posterior (unnormalized)
        l1  = mix.logω[1]  + mix.lognorm[1]  - 0.5 * (r - mix.m[1])^2  * mix.invv[1]
        l2  = mix.logω[2]  + mix.lognorm[2]  - 0.5 * (r - mix.m[2])^2  * mix.invv[2]
        l3  = mix.logω[3]  + mix.lognorm[3]  - 0.5 * (r - mix.m[3])^2  * mix.invv[3]
        l4  = mix.logω[4]  + mix.lognorm[4]  - 0.5 * (r - mix.m[4])^2  * mix.invv[4]
        l5  = mix.logω[5]  + mix.lognorm[5]  - 0.5 * (r - mix.m[5])^2  * mix.invv[5]
        l6  = mix.logω[6]  + mix.lognorm[6]  - 0.5 * (r - mix.m[6])^2  * mix.invv[6]
        l7  = mix.logω[7]  + mix.lognorm[7]  - 0.5 * (r - mix.m[7])^2  * mix.invv[7]
        l8  = mix.logω[8]  + mix.lognorm[8]  - 0.5 * (r - mix.m[8])^2  * mix.invv[8]
        l9  = mix.logω[9]  + mix.lognorm[9]  - 0.5 * (r - mix.m[9])^2  * mix.invv[9]
        l10 = mix.logω[10] + mix.lognorm[10] - 0.5 * (r - mix.m[10])^2 * mix.invv[10]

        maxlog = max(l1,l2,l3,l4,l5,l6,l7,l8,l9,l10)

        p1  = exp(l1  - maxlog)
        p2  = exp(l2  - maxlog)
        p3  = exp(l3  - maxlog)
        p4  = exp(l4  - maxlog)
        p5  = exp(l5  - maxlog)
        p6  = exp(l6  - maxlog)
        p7  = exp(l7  - maxlog)
        p8  = exp(l8  - maxlog)
        p9  = exp(l9  - maxlog)
        p10 = exp(l10 - maxlog)

        total = p1+p2+p3+p4+p5+p6+p7+p8+p9+p10
        u = rand() * total

        acc = p1
        if u ≤ acc; S[t] = 1; continue; end
        acc += p2
        if u ≤ acc; S[t] = 2; continue; end
        acc += p3
        if u ≤ acc; S[t] = 3; continue; end
        acc += p4
        if u ≤ acc; S[t] = 4; continue; end
        acc += p5
        if u ≤ acc; S[t] = 5; continue; end
        acc += p6
        if u ≤ acc; S[t] = 6; continue; end
        acc += p7
        if u ≤ acc; S[t] = 7; continue; end
        acc += p8
        if u ≤ acc; S[t] = 8; continue; end
        acc += p9
        if u ≤ acc; S[t] = 9; continue; end
        S[t] = 10
    end

    return S
end


function SetUpLogChi2Mixture(nComp, df = 1)

    if df !== 1
        error("Only df = 1 is implemented")
    end

    if nComp == 10
        ω = [0.00609, 0.04775, 0.13057, 0.20674, 0.22715,
             0.18842, 0.12047, 0.05591, 0.01575, 0.00115]

        m = [1.92677, 1.34744, 0.73504, 0.02266, -0.85173,
             -1.97278, -3.46788, -5.55246, -8.68384, -14.6500]

        v = [0.11265, 0.17788, 0.26768, 0.40611, 0.62699,
             0.98583, 1.57469, 2.54498, 4.16591, 7.33342]

        ω ./= sum(ω)
        return ω, m, v
    end

    error("Only 10 components are implemented")
end


function Update_h(ỹ, m, v, Dᵩ, ξ, ϕ, σ²ₙ, μ)
    T = length(ỹ)

    @inbounds Dᵩ[band(-1)] .= -ϕ

    invσ²ₙ = 1 / σ²ₙ

    main = Vector{Float64}(undef, T)
    sub  = Vector{Float64}(undef, T-1)

    x = ξ .* invσ²ₙ   # could avoid alloc but kept clear

    # t = 1
    main[1] = 1/v[1] + x[1] + ϕ^2 * x[2]

    @inbounds for t in 2:T-1
        main[t]  = 1/v[t] + x[t] + ϕ^2 * x[t+1]
        sub[t-1] = -ϕ * x[t]
    end

    # t = T
    main[T] = 1/v[T] + x[T]
    sub[T-1] = -ϕ * x[T]

    Qh̃ = PDSparseMat(spdiagm(-1 => sub, 0 => main, 1 => sub))

    lh̃ = Vector{Float64}(undef, T)
    @inbounds for t in 1:T
        lh̃[t] = (ỹ[t] - m[t] - μ) / v[t]
    end

    return rand(MvNormalCanon(lh̃, Qh̃)) .+ μ
end

function Updateξ1(y, ϕ, σ²ₙ, μ, α, β; δ = 0.0001)
    η = (y[2:end] .- μ .- ϕ .* (y[1:end-1] .- μ)) ./ sqrt.(σ²ₙ)
    ξ = rand.(PolyaGammaPSWSampler.(Int(α+β), [(y[1]-μ)/sqrt.(σ²ₙ); η]))
    return ξ .+ δ
end

# Centered parametrization
function Updateμ(y, ξ, ϕ, σ²ₙ, m₀, σ₀)
    T = length(y)

    one_minus_ϕ = 1 - ϕ
    scale = one_minus_ϕ^2

    sumξ = 0.0
    sumξz = 0.0

    # t = 1
    ξ1 = ξ[1]
    z1 = y[1]
    sumξ  += ξ1
    sumξz += ξ1 * z1

    @inbounds for t in 2:T
        ξt = scale * ξ[t]
        zt = (y[t] - ϕ*y[t-1]) / one_minus_ϕ
        sumξ  += ξt
        sumξz += ξt * zt
    end

    σₜ² = 1 / (sumξ / σ²ₙ + 1 / σ₀^2)
    v = 1 - (1 / σ₀^2) * σₜ²

    μ̂ = sumξz / sumξ
    μₜ = v * μ̂ + (1 - v) * m₀

    return rand(Normal(μₜ, sqrt(σₜ²)))
end

### Compute exponentiated variances
LogVol2Covs(H) = PDMat.([diagm(exp.(H[t,:])) for t in 1:size(H,1)])
Vol2Covs(H) = PDMat(diagm(H))