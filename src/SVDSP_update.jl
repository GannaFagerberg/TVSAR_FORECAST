#UpdateErrorVolatility_DSP
#update_dsp_sv_h

### SV DSP (refactored, faster, same API/outputs)
#using LinearAlgebra
#using SparseArrays
#using BandedMatrices

# assumes you already have:
# - PDSparseMat
# - MvNormalCanon
# - UpdateMixAlloc!
# - Updateξ, Updateϕ, Updateμ
# - mixLogχ²₁, postDist etc.

# ---------------------------
# Small caches (keyed by T)
# ---------------------------
const _SV_CACHE_D      = IdDict{Int, SparseMatrixCSC{Float64, Int}}()
const _SV_CACHE_Dphi   = IdDict{Int, BandedMatrix{Float64}}()
const _SV_CACHE_VEC_H0 = IdDict{Int, NamedTuple}()

@inline function _get_D(T::Int)
    get!(_SV_CACHE_D, T) do
        # D is T × (T+1), row t: [ ... 1, -1 ... ]
        D = spzeros(Float64, T, T + 1)
        @inbounds for t in 1:T
            D[t, t] = 1.0
            D[t, t + 1] = -1.0
        end
        D
    end
end

@inline function _get_Dphi(T::Int)
    get!(_SV_CACHE_Dphi, T) do
        # BandedMatrix with diag = 1, subdiag = 0 (we overwrite to -ϕ each call)
        BandedMatrix(-1 => zeros(Float64, T - 1), 0 => ones(Float64, T))
    end
end

@inline function _get_ws_h0(T::Int)
    get!(_SV_CACHE_VEC_H0, T) do
        # workspace arrays for Update_h_with_h0 (length T+1)
        T̃ = T + 1
        (
            diagQ  = zeros(Float64, T̃),
            offQ   = zeros(Float64, T̃ - 1), # for ±1 diagonals
            ℓ      = zeros(Float64, T̃),
            invx   = zeros(Float64, T),      # ξ ./ σ²ₙ
            invv   = zeros(Float64, T),      # 1 ./ v
        )
    end
end




# -------------------------------------------------------
# Main function (same signature/returns)
# -------------------------------------------------------


function UpdateErrorVolatility_DSP(
    errors, h̄, hstar, h̃, ξ̄, ϕ̄, μ̄,
    ϕ̄₀, κ̄₀, m̄₀, σ̄₀, ν̄₀, ψ̄₀,
    mixLogχ²₁, postDist, m, v, Ssv, σ02;
    σ̄²ₙ = 1, d_order = 1, offsetSV = eps()
)

    # -------------------------------------------------------
    # Dimensions
    # -------------------------------------------------------
    T = length(errors)

    # -------------------------------------------------------
    # 1. Log-squared observations:
    #    y_t = log(errors_t^2 + offset)
    #    (measurement equation for h*_t)
    # -------------------------------------------------------
    ȳstar = similar(errors, Float64)
    @inbounds @simd for i in eachindex(errors)
        ȳstar[i] = log(errors[i]^2 + offsetSV)
    end

    # -------------------------------------------------------
    # 2. Mixture allocation for log χ²₁:
    #    y_t - h*_t ≈ m_{s_t} + ε_t,  ε_t ~ N(0, v_{s_t})
    # -------------------------------------------------------
    @views h_prev = hstar[2:end]          # h*_1,…,h*_T
    s̄ = UpdateMixAlloc(ȳstar, h_prev, mixLogχ²₁)

    # -------------------------------------------------------
    # 3. Update h*_0:T (random walk with time-varying variance):
    #    h*_t - h*_{t-1} ~ N(0, exp(h_t))
    # -------------------------------------------------------
    #m̄₀_emp = median(ȳstar[1:10]) 
    #m̄₀_emp = ȳstar[1]

    #m̄₀_emp = 0.0#
    #h̄_clamped = @. clamp.(h̄, -Inf, 700)
    #precision  = @. exp(-h̄_clamped) 
    h̄ .= min.(h̄, 700)

    hstar = Update_h_with_h0(
        ȳstar,                  # log-squared observations
        m[s̄],                   # mixture means
        v[s̄],                   # mixture variances
        1.0 ./ exp.(h̄),         # precision = exp(-h_t)
        #1.0 ./ (exp.(h̄) .+ 0.01); # 0.00001 
        #1.0 ./ (exp.(h̄) .+ 0.1^15); 
        #precision,
        σ²ₙ = 1.0,
        m̄₀ = log(σ02) + log(0.5) ,
        #+ log(0.5),               # prior mean for h*_0
        #+ log(0.05)
        σ̄₀ = 0.5,
        diff_order = d_order                 # prior sd for h*_0
    )

    # -------------------------------------------------------
    # 4. Innovations of the random walk:
    #    ω_t = h*_t - h*_{t-1}
    # -------------------------------------------------------
    
    if d_order == 1
        omega = diff(hstar)
    else
        omega = diff(diff(hstar))
    end

    # -------------------------------------------------------
    # 5. SV block on ω_t:
    #    log(ω_t^2) = h_t + log(χ²₁)
    #    h_t follows DSP-SV dynamics
    # -------------------------------------------------------

    h̄, h̃, Ssv, ξ̄, ϕ̄, μ̄ = update_dsp_sv_h(
        omega, Ssv, h̄, h̃, ξ̄, ϕ̄, μ̄,
        ϕ̄₀, κ̄₀, m̄₀, σ̄₀, ν̄₀, ψ̄₀,
        mixLogχ²₁, vec(m), vec(v), postDist;
        offset = eps(), α = 1/2, β = 1/2, σ²ₙ = 1.0
    )

    # -------------------------------------------------------
    # Return updated latent states and parameters
    # -------------------------------------------------------
    return hstar, h̄, h̃, Ssv, ξ̄, ϕ̄, μ̄
end


# ---------------------------------------------
# Update hstar including h0 (same signature)
# ---------------------------------------------

function Update_h_with_h0(
    ỹ, m, v, ξ;
    σ²ₙ = 1.0,
    m̄₀ = -15.0,
    σ̄₀ = 3.0,
    diff_order::Int = 1
)

    T  = length(ỹ)
    T̃ = T + diff_order

    invv = 1.0 ./ v
    invx = ξ ./ σ²ₙ

    ℓ = zeros(Float64, T̃)

    if diff_order == 1

        # ----------------------------
        # RW1  → Tridiagonal
        # ----------------------------

        main = zeros(Float64, T̃)
        sub  = zeros(Float64, T̃ - 1)

        # RW contribution
        main[1] = invx[1] + 1/(σ̄₀^2)
        for i in 2:T
            main[i] = invx[i-1] + invx[i]
        end
        main[T̃] = invx[T]

        for i in 1:T
            sub[i] = -invx[i]
        end

        # Observation contribution
        for t in 1:T
            main[t+1] += invv[t]
            ℓ[t+1] = invv[t]*(ỹ[t] - m[t])
        end

        ℓ[1] += m̄₀/(σ̄₀^2)

        Q = spdiagm(-1 => sub, 0 => main, 1 => sub)
        Q = PDSparseMat(Q)

    elseif diff_order == 2

        # ----------------------------
        # RW2 → Pentadiagonal
        # ----------------------------

        main  = zeros(Float64, T̃)
        sub1  = zeros(Float64, T̃-1)
        sub2  = zeros(Float64, T̃-2)

        # RW2 structure
        # Each invx[t] contributes:
        #  [ 1  -2   1 ] outer product

        for t in 1:T
            i = t
            w = invx[t]

            main[i]     += w
            main[i+1]   += 4w
            main[i+2]   += w

            sub1[i]     += -2w
            sub1[i+1]   += -2w

            sub2[i]     += w
        end

        # Prior on first two states
        main[1] += 1/(σ̄₀^2)
        main[2] += 1/(σ̄₀^2)

        # Observation contribution
        for t in 1:T
            main[t+2] += invv[t]
            ℓ[t+2] = invv[t]*(ỹ[t] - m[t])
        end

        ℓ[1] += m̄₀/(σ̄₀^2)
        ℓ[2] += m̄₀/(σ̄₀^2)

        Q = spdiagm(
            -2 => sub2,
            -1 => sub1,
             0 => main,
             1 => sub1,
             2 => sub2
        )

        Q = PDSparseMat(Q)

    else
        error("diff_order must be 1 or 2")
    end

    return rand(MvNormalCanon(ℓ, Q))
end


# only RW in hstar
function Update_h_with_h0_ref(
    ỹ, m, v, ξ;
    σ²ₙ = 1.0,              # scale parameter for RW innovations (usually 1)
    m̄₀ = -15.0,            # prior mean for initial log-variance h*_0
    σ̄₀ = 3.0               # prior sd for initial log-variance h*_0
)

    # -------------------------------------------------------
    # Dimensions:
    # ỹ is length T (observations for h*_1,…,h*_T)
    # We sample h*_0,…,h*_T jointly → length T+1
    # -------------------------------------------------------
    T  = length(ỹ)
    T̃ = T + 1              # include initial state h*_0

    # -------------------------------------------------------
    # Workspace buffers (preallocated, keyed by T)
    # -------------------------------------------------------
    ws    = _get_ws_h0(T)
    diagQ = ws.diagQ        # main diagonal of precision matrix
    offQ  = ws.offQ         # first off-diagonals of precision matrix
    ℓ     = ws.ℓ            # canonical linear term
    invx  = ws.invx         # RW innovation precisions: ξ_t / σ²ₙ
    invv  = ws.invv         # observation precisions: 1 / v_t

    # -------------------------------------------------------
    # Compute precisions
    # RW innovation precision:
    #   Var(h*_t − h*_{t−1}) = σ²ₙ / ξ_t
    #   ⇒ precision = ξ_t / σ²ₙ
    #
    # Observation precision:
    #   y_t = h*_t + m_t + ε_t,  ε_t ~ N(0, v_t)
    # -------------------------------------------------------
    @inbounds @simd for t in 1:T
        invx[t] = ξ[t] / σ²ₙ
        invv[t] = 1.0 / v[t]
    end

    # -------------------------------------------------------
    # Build precision matrix Q for h* = (h*_0,…,h*_T)
    #
    # Q = D' * diag(invx) * D     (random-walk prior)
    #   + diag(0, invv[1],…,invv[T])  (observation likelihood)
    #   + prior on h*_0
    # -------------------------------------------------------
    fill!(diagQ, 0.0)
    fill!(offQ, 0.0)

    @inbounds begin
        # Diagonal terms from RW prior
        diagQ[1] = invx[1] + 1.0 / (σ̄₀^2)   # h*_0: RW + prior
        for i in 2:T
            diagQ[i] = invx[i - 1] + invx[i]
        end
        diagQ[T̃] = invx[T]

        # Off-diagonal terms from RW prior
        for i in 1:T
            offQ[i] = -invx[i]
        end

        # Add observation precision to h*_1,…,h*_T
        for t in 1:T
            diagQ[t + 1] += invv[t]
        end
    end

    # -------------------------------------------------------
    # Sparse symmetric tridiagonal precision matrix
    # -------------------------------------------------------
    Q = spdiagm(-1 => offQ, 0 => diagQ, 1 => offQ)
    Q = PDSparseMat(Q)

    # -------------------------------------------------------
    # Canonical linear term ℓ
    #
    # From likelihood:
    #   ℓ_{t} = (y_t − m_t) / v_t   for h*_t
    #
    # From prior on h*_0:
    #   ℓ_0 = m̄₀ / σ̄₀²
    # -------------------------------------------------------
    fill!(ℓ, 0.0)
    @inbounds begin
        ℓ[1] = m̄₀ / (σ̄₀^2)
        for t in 1:T
            ℓ[t + 1] = invv[t] * (ỹ[t] - m[t])
        end
    end

    # -------------------------------------------------------
    # Joint Gaussian draw:
    #   h* ~ N(Q^{-1}ℓ, Q^{-1})
    # -------------------------------------------------------
    h_all = rand(MvNormalCanon(ℓ, Q))

    # -------------------------------------------------------
    # Return (h*_0, h*_1,…,h*_T)
    # -------------------------------------------------------
    return h_all
end

# -------------------------------------------------------
# Update h in SV block 
# -------------------------------------------------------
function UpdateH_sv(ỹ, m, v, Dᵩ, ξ, ϕ, μ; σ²ₙ = 1.0)

    # -------------------------------------------------------
    # Dimensions:
    # ỹ_t are log-squared innovations (length T)
    # h_t is sampled for t = 1,…,T (no h_0 sampled here)
    # -------------------------------------------------------
    T = length(ỹ)

    # -------------------------------------------------------
    # AR(1) transition operator:
    # h_t − ϕ h_{t−1} = μ(1−ϕ) + η_t
    # Here h_1 = μ + η_1 (no explicit h_0)
    # -------------------------------------------------------
    @inbounds Dᵩ[band(-1)] .= -ϕ

    # -------------------------------------------------------
    # Innovation precisions:
    # Var(η_t) = σ²ₙ / ξ_t
    # ⇒ precision = ξ_t / σ²ₙ
    # -------------------------------------------------------
    invσ²ₙ = 1.0 / σ²ₙ
    x = ξ .* invσ²ₙ

    # -------------------------------------------------------
    # Tridiagonal precision matrix components
    # -------------------------------------------------------
    main = Vector{Float64}(undef, T)     # diagonal
    sub  = Vector{Float64}(undef, T - 1) # off-diagonal

    # -------------------------------------------------------
    # t = 1:
    # Contribution from:
    #  - observation likelihood
    #  - innovation at t = 1: h_1 − μ
    #  - innovation at t = 2: h_2 − ϕ h_1
    # -------------------------------------------------------
    main[1] = 1.0 / v[1] + x[1] + ϕ^2 * x[2]

    # -------------------------------------------------------
    # t = 2,…,T−1:
    # Contribution from:
    #  - observation likelihood
    #  - innovations at t and t+1
    # -------------------------------------------------------
    @inbounds for t in 2:T-1
        main[t]  = 1.0 / v[t] + x[t] + ϕ^2 * x[t+1]
        sub[t-1] = -ϕ * x[t]
    end

    # -------------------------------------------------------
    # t = T:
    # Contribution from:
    #  - observation likelihood
    #  - innovation at t = T
    # -------------------------------------------------------
    main[T] = 1.0 / v[T] + x[T]
    sub[T-1] = -ϕ * x[T]

    # -------------------------------------------------------
    # Sparse symmetric tridiagonal precision matrix
    # -------------------------------------------------------
    Qh̃ = PDSparseMat(spdiagm(-1 => sub, 0 => main, 1 => sub))

    # -------------------------------------------------------
    # Canonical linear term from observation equation:
    # ỹ_t − m_t − μ = h_t + ε_t,   ε_t ~ N(0, v_t)
    # -------------------------------------------------------
    lh̃ = Vector{Float64}(undef, T)
    @inbounds for t in 1:T
        lh̃[t] = (ỹ[t] - m[t] - μ) / v[t]
    end

    # -------------------------------------------------------
    # Sample centered h_t and shift back by μ
    # -------------------------------------------------------
    return rand(MvNormalCanon(lh̃, Qh̃)) .+ μ
end

# -------------------------------------------------------
# SV DSP update step (same signature/returns)
# -------------------------------------------------------

function update_dsp_sv_h(
    ν, S, H, H̃, ξ, ϕ, μ,
    ϕ₀, κ₀, m₀, σ₀, ν₀, ψ₀,
    mixLogχ²₁, m, v, postDist;
    offset = eps(), α = 1/2, β = 1/2, σ²ₙ = 1.0
)

    # -------------------------------------------------------
    # Dimensions:
    # ν_t are innovations from h*_t − h*_{t−1}
    # length(ν) = T
    # -------------------------------------------------------
    T = length(ν)

    # -------------------------------------------------------
    # 1. Log-squared innovations:
    #    log(ν_t^2) = h_t + log(χ²_1)
    # -------------------------------------------------------
    Ỹ = similar(ν, Float64)
    @inbounds @simd for i in eachindex(ν)
        Ỹ[i] = log(ν[i]^2 + offset)
    end

    # -------------------------------------------------------
    # 2. State transition operator for AR(1):
    #    h_t − ϕ h_{t−1} = μ (1−ϕ) + η_t
    #    (banded matrix, subdiagonal overwritten each call)
    # -------------------------------------------------------
    Dᵩ = _get_Dphi(T)

    # -------------------------------------------------------
    # 3. Mixture allocation for log χ²₁ approximation
    #    p(S_t=j | Ỹ_t, h_t)
    # -------------------------------------------------------
    S = UpdateMixAlloc(Ỹ,  H̃ .+ μ , mixLogχ²₁)

    # -------------------------------------------------------
    # 4. Update latent log-variance path h₁,…,h_T
    #
    # Observation:
    #   Ỹ_t − m_{S_t} = h_t + ε_t,  ε_t ~ N(0, v_{S_t})
    #
    # State evolution:
    #   h_t = μ + ϕ(h_{t−1} − μ) + η_t,
    #   η_t ~ N(0, σ²ₙ / ξ_t)
    # -------------------------------------------------------
    H = UpdateH_sv(
        Ỹ,
        m[S],
        v[S],
        Dᵩ,
        ξ,
        ϕ,
        μ;
        σ²ₙ = σ²ₙ
    )

    # -------------------------------------------------------
    # 5. Update latent scale variables ξ_t
    #    (DSP / global–local shrinkage layer)
    #
    # This renders η_t conditionally Gaussian
    # -------------------------------------------------------
    ξ = Updateξ_sv(
        H,
        ϕ,
        σ²ₙ,
        μ,
        α,
        β;
        δ = 0.01  # 0.01
    )

    # -------------------------------------------------------
    # 6. Update AR(1) coefficient ϕ
    # -------------------------------------------------------
    ϕ = Updateϕ(H, ξ, μ, σ²ₙ, ϕ₀, κ₀)

    # -------------------------------------------------------
    # 7. Update long-run mean μ (centered parameterization)
    # -------------------------------------------------------
    μ = Updateμ(H, ξ, ϕ, σ²ₙ, m₀, σ₀)

    # -------------------------------------------------------
    # 8. Non-centered transform (for storage / diagnostics)
    # -------------------------------------------------------
    H̃ = H .- μ

    # -------------------------------------------------------
    # Return updated latent states and parameters
    # -------------------------------------------------------
    return H, H̃, S, ξ, ϕ, μ
end

# -------------------------------------------------------
# PG with shoulders 
# -------------------------------------------------------
function Updateξ_sv(y, ϕ, σ²ₙ, μ, α, β; δ = 0.00)
    η = (y[2:end] .- μ .- ϕ .* (y[1:end-1] .- μ)) ./ sqrt.(σ²ₙ)
    ξ = rand.(PolyaGammaPSWSampler.(Int(α+β), [(y[1]-μ)/sqrt.(σ²ₙ); η]))
    return ξ .+ δ
end

