

# ------------------------------------------------------------------------------
# ONE-TIME setup outside MCMC
# ------------------------------------------------------------------------------

#
# # inside each MCMC iteration:
# init_y, Z = build_AR_init_opt_orig2!(
#     x0_buf, int_exp, m0_buf, group_map, ws,
#     Y, state, ϕ_expanded, activeLags_ar, p1, s1, p_max, σₑ², σy;
#     INTERCEPT = INTERCEPT,
#     cond_sma = cond_sma,
#     Z = Z,
#     l = l,
#     rng = rng
# )


mutable struct SMAWorkspace{T}
    Q::Matrix{T}
    bvec::Vector{T}
    Gbuf::Matrix{T}
    bbuf::Vector{T}
    g::Vector{T}
end

function build_SMA_workspace(q::Int; T=Float64)
    return SMAWorkspace(
        zeros(T, q, q),   # Q
        zeros(T, q),      # bvec
        Matrix{T}(LinearAlgebra.I, q, q), # Gbuf
        zeros(T, q),      # bbuf
        zeros(T, q)       # g
    )
end


function build_MA_errors_banded(
    errors::Vector{Float64},
    Y::Vector{Float64},
    state::AbstractMatrix,
    ψ_expanded::Matrix{Float64},
    activeLags_ma::Vector{Int},
    p2, s2,
    p_max,
    σₑ²,
    σ0,
    T_use;
    ws_sma::SMAWorkspace,
    INTERCEPT::Bool = true,
    ztrans = "partials",
    rng::AbstractRNG = Random.default_rng(),
    presample_mode::Symbol = :posterior,   # :posterior or :simple
    use_σ0_for_presample::Bool = true,
)
    T     = size(state, 1) - 1
    T_all = T * l
    q     = p_max[2]

    # --------------------------------------------------
    # Detrend data
    # --------------------------------------------------
    y_detrended = similar(Y)

    if INTERCEPT
        intercept_grouped  = reshape(Float64.(state[2:end, 1]), 1, :)
        intercept_expanded = Matrix{Float64}(undef, 1, T_all)
        expand_grouped_states_fast!(intercept_expanded, intercept_grouped, l, T_all)

        @inbounds for t in 1:T_all
            y_detrended[t] = Y[t] - intercept_expanded[1, t]
        end
    else
        copyto!(y_detrended, Y)
    end

    # --------------------------------------------------
    # Build σe2
    # --------------------------------------------------
    σe2 = σₑ² isa Number  ? Float64(σₑ²) :
          σₑ² isa Vector  ? Float64.(σₑ²) :
          σₑ² isa Matrix  ? vec(Float64.(σₑ²)) :
          error("σₑ² must be scalar, vector, or 1×T matrix")

    # --------------------------------------------------
    # Sample presample ε_{1−q:0}
    # --------------------------------------------------
    εprefix = view(errors, 1:q)
    u_out   = εprefix

    if presample_mode === :posterior
        sample_SMA_presample_u_info!(
            u_out,
            εprefix,
            view(y_detrended, 1:T_use),
            view(ψ_expanded, :, 1:T_use),
            activeLags_ma,
            σe2,
            σ0^2,
            q,
            T_use,
            ws_sma;
            rng = rng,
        )
    elseif presample_mode === :simple
        sample_SMA_presample_u_simple!(
            u_out,
            εprefix,
            σe2,
            σ0^2,
            q;
            rng = rng,
            use_σ0 = use_σ0_for_presample,
        )
    else
        error("Unknown presample_mode = $presample_mode. Use :posterior or :simple.")
    end

    # convert from (ε0, ε-1, ..., ε_{1-q}) to (ε_{1-q}, ..., ε0)
    reverse!(u_out)

    # --------------------------------------------------
    # Deterministic MA recursion for ε_t, t ≥ 1
    # --------------------------------------------------
    k = length(activeLags_ma)

    @inbounds for t in 1:T_all
        j = q + t
        acc = 0.0

        ψt = @view ψ_expanded[:, t]

        @simd for i in 1:k
            lag = activeLags_ma[i]
            acc += ψt[i] * errors[j - lag]
        end

        errors[j] = y_detrended[t] - acc
    end

    return errors
end

function build_MA_errors_banded_ref(
    errors::Vector{Float64},
    Y::Vector{Float64},
    state::AbstractMatrix,
    ψ_expanded::Matrix{Float64},
    activeLags_ma::Vector{Int},
    p2, s2,
    p_max,
    σₑ²,
    σ0,
    T_use;
    ws_sma::SMAWorkspace,              # new
    INTERCEPT::Bool = true,
    ztrans = "partials"
)
    T     = size(state, 1) - 1
    T_all = T * l
    q     = p_max[2]

    # --------------------------------------------------
    # Detrend data
    # --------------------------------------------------
    y_detrended = similar(Y)

    if INTERCEPT
        intercept_grouped  = reshape(Float64.(state[2:end, 1]), 1, :)
        intercept_expanded = Matrix{Float64}(undef, 1, T_all)
        expand_grouped_states_fast!(intercept_expanded, intercept_grouped, l, T_all)

        @inbounds for t in 1:T_all
            y_detrended[t] = Y[t] - intercept_expanded[1, t]
        end
    else
        copyto!(y_detrended, Y)
    end

    # --------------------------------------------------
    # Build σe2
    # --------------------------------------------------
    σe2 = σₑ² isa Number      ? Float64(σₑ²) :
          σₑ² isa Vector     ? Float64.(σₑ²) :
          σₑ² isa Matrix     ? vec(Float64.(σₑ²)) :
          error("σₑ² must be scalar, vector, or 1×T matrix")

    # --------------------------------------------------
    # Sample presample ε_{1−q:0}
    # --------------------------------------------------
   u_tmp = similar(errors, q)

   sample_SMA_presample_u_info!(
        u_tmp,
        u_tmp,
        view(y_detrended, 1:T_use),          # ✔ correct data
        view(ψ_expanded, :, 1:T_use),        # ✔ correct ψ
        activeLags_ma,
        σe2,
        σ0^2,
        q,
        T_use,
        ws_sma
    )

    #reverse!(u_out)
    errors[1:q] .= reverse(u_tmp)
    #errors[1:q] .= u_tmp
    # --------------------------------------------------
    # Deterministic MA recursion for ε_t, t ≥ 1
    # --------------------------------------------------
    k = length(activeLags_ma)

    @inbounds for t in 1:T_all
        
        j = q + t
        acc = 0.0

        ψt = @view ψ_expanded[:, t] # start t=1

        @simd for i in 1:k
            lag = activeLags_ma[i]
            acc += ψt[i] * errors[j - lag]
        end

        errors[j] = y_detrended[t] - acc
    end

    return errors
end


# u = (ε0, ε-1, ..., ε_{1-q})

function sample_SMA_presample_u_info!(
    u_out::AbstractVector{<:Real},
    εprefix::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    ψ::AbstractMatrix{<:Real},
    activeLags_ma::AbstractVector{<:Integer},
    σe2::Union{Real,AbstractVector{<:Real}},
    σ0²::Real,
    q::Int,
    T_use::Int,
    ws::SMAWorkspace;   # ← NEW
    rng::AbstractRNG = Random.default_rng(),
)

    @assert length(u_out) == q
    @assert length(εprefix) ≥ q
    @assert size(ψ,1) == length(activeLags_ma)
    @assert size(ψ,2) ≥ T_use
    @assert maximum(activeLags_ma) <= q
    @assert σ0² > 0

    k = length(activeLags_ma)
    σe2_is_scalar = σe2 isa Number

    if σe2_is_scalar
        @assert σe2 > 0
    else
        @assert length(σe2) ≥ T_use
        @assert all(@view(σe2[1:T_use]) .> 0)
    end

    invσ0² = 1.0 / σ0²
    invσe2_scalar = σe2_is_scalar ? (1.0 / σe2) : 0.0

    # --------------------------------------------------
    # Workspace buffers
    # --------------------------------------------------
    Q    = ws.Q
    bvec = ws.bvec
    Gbuf = ws.Gbuf
    bbuf = ws.bbuf
    g    = ws.g

    # Reset buffers
    fill!(Q, 0.0)
    fill!(bvec, 0.0)
    fill!(bbuf, 0.0)
    fill!(g, 0.0)

    # Reset Gbuf = I
    fill!(Gbuf, 0.0)
    @inbounds for i in 1:q
        Gbuf[i,i] = 1.0
    end

    # Prior contribution
    @inbounds for i in 1:q
        Q[i,i] = invσ0²
    end

    # --------------------------------------------------
    # Ring buffer
    # --------------------------------------------------
    ring = 0
    @inline pos_for_offset(offset::Int) = mod(ring - offset, q) + 1

    # --------------------------------------------------
    # Forward recursion
    # --------------------------------------------------
    @inbounds for t in 1:T_use
        ring = mod(ring, q) + 1

        bt = y[t]
        fill!(g, 0.0)

        ψt = @view ψ[:, t]

        for ii in 1:k
            lag   = activeLags_ma[ii]
            coeff = ψt[ii]

            if coeff == 0.0
                continue
            end

            if t > lag
                slot = pos_for_offset(lag)

                bt -= coeff * bbuf[slot]

                col = @view Gbuf[:, slot]
                for j in 1:q
                    g[j] -= coeff * col[j]
                end
            else
                idx = 1 - (t - lag)
                @assert 1 <= idx <= q
                g[idx] -= coeff
            end
        end

        # Store affine form
        bbuf[ring] = bt
        @views Gbuf[:, ring] .= g

        invσe2 = σe2_is_scalar ? invσe2_scalar : (1.0 / σe2[t])

        # Precision update
        for j in 1:q
            gj = g[j]
            bvec[j] -= invσe2 * bt * gj
            for i in 1:j
                Q[i,j] += invσe2 * g[i] * gj
            end
        end
    end

    # --------------------------------------------------
    # Symmetrise Q
    # --------------------------------------------------
    @inbounds for j in 1:q
        for i in 1:(j-1)
            Q[j,i] = Q[i,j]
        end
    end

    # --------------------------------------------------
    # Sample
    # --------------------------------------------------
    F = cholesky(Symmetric(Q))
    μ = F \ bvec

    z = randn(rng, q)

    copyto!(u_out, μ)
    u_out .+= (F.U \ z)

    εprefix[1:q] .= u_out

    return nothing
end

# Update weather beta paarmeter
function update_beta!(r, X,σₑ², b0, B0)
    
    σ2e = σₑ²[1,1]

    prec = dot(X, X) / σ2e + 1 / B0
    var  = 1 / prec

    mean = var * (dot(X, r) / σ2e + b0 / B0)

    β = mean + sqrt(var) * randn()

    return mean + sqrt(var) * randn()
end

function update_beta_tvvar!(r, X, σ2e_vec, b0, B0)

    @inbounds begin
        sum_x2 = 0.0
        sum_xr = 0.0

        for t in eachindex(X)
            w = 1.0 / σ2e_vec[t]
            xt = X[t]

            sum_x2 += xt * xt * w
            sum_xr += xt * r[t] * w
        end
    end

    prec = sum_x2 + 1 / B0
    var  = 1 / prec
    mean = var * (sum_xr + b0 / B0)

    return mean + sqrt(var) * randn()
end




function update_beta_hd_cd_tvvar!(
    r, HD, CD, σ2e_vec,
    bh, bc,      # prior means
    Bh, Bc       # prior variances
)
    q11 = 0.0
    q22 = 0.0
    q12 = 0.0
    s1  = 0.0
    s2  = 0.0

    @inbounds for t in eachindex(r)
        w  = 1.0 / σ2e_vec[t]
        ht = HD[t]
        ct = CD[t]
        rt = r[t]

        q11 += ht * ht * w
        q22 += ct * ct * w
        q12 += ht * ct * w

        s1  += ht * rt * w
        s2  += ct * rt * w
    end

    # prior contribution
    q11 += 1 / Bh
    q22 += 1 / Bc
    s1  += bh / Bh
    s2  += bc / Bc

    # precision determinant
    detQ = q11 * q22 - q12^2
    detQ <= 0 && error("Posterior precision is not positive definite.")

    # covariance = Q^{-1}
    V11 =  q22 / detQ
    V22 =  q11 / detQ
    V12 = -q12 / detQ

    # posterior mean
    m1 = V11 * s1 + V12 * s2
    m2 = V12 * s1 + V22 * s2

    # full correlated Gaussian draw
    V = Symmetric([V11 V12; V12 V22])
    L = cholesky(V).L
    z = randn(2)
    β = [m1, m2] .+ L * z

    return β[1], β[2]
end

function update_beta_hd_cd_tvvar_no_cov!(
    r, HD, CD, σ2e_vec,
    bh, bc,      # prior means
    Bh, Bc       # prior variances
)
    xx11 = 0.0
    xx22 = 0.0
    xx12 = 0.0
    xy1  = 0.0
    xy2  = 0.0

    @inbounds for t in eachindex(r)
        w = 1.0 / σ2e_vec[t]

        ht = HD[t]
        ct = CD[t]
        rt = r[t]

        xx11 += ht * ht * w
        xx22 += ct * ct * w
        xx12 += ht * ct * w

        xy1  += ht * rt * w
        xy2  += ct * rt * w
    end

    # add prior (DIAGONAL but different)
    xx11 += 1 / Bh
    xx22 += 1 / Bc

    # determinant
    det = xx11 * xx22 - xx12^2

    # inverse of precision
    V11 =  xx22 / det
    V22 =  xx11 / det
    V12 = -xx12 / det

    # posterior mean
    m1 = V11 * (xy1 + bh / Bh) + V12 * (xy2)
    m2 = V12 * (xy1) + V22 * (xy2 + bc / Bc)

    # sample (still diagonal approx)
    z1 = randn()
    z2 = randn()

    βh = m1 + sqrt(V11) * z1
    βc = m2 + sqrt(V22) * z2

    return βh, βc
end


function update_beta_hd_cd_mean(
    r, HD, CD, σ2e_vec,
    bh, bc,
    Bh, Bc
)
    q11 = 0.0
    q22 = 0.0
    q12 = 0.0
    s1  = 0.0
    s2  = 0.0

    @inbounds for t in eachindex(r)
        w  = 1.0 / σ2e_vec[t]
        ht = HD[t]
        ct = CD[t]
        rt = r[t]

        q11 += ht * ht * w
        q22 += ct * ct * w
        q12 += ht * ct * w

        s1  += ht * rt * w
        s2  += ct * rt * w
    end

    q11 += 1 / Bh
    q22 += 1 / Bc
    s1  += bh / Bh
    s2  += bc / Bc

    detQ = q11 * q22 - q12^2

    m1 = ( q22 * s1 - q12 * s2 ) / detQ
    m2 = ( -q12 * s1 + q11 * s2 ) / detQ

    return m1, m2
end


function init_beta_hd_cd(y::AbstractVector,
                        HD::AbstractVector,
                        CD::AbstractVector)

    @assert length(y) == length(HD) == length(CD)

    # X'X components
    xx11 = dot(HD, HD)      # HD'HD
    xx22 = dot(CD, CD)      # CD'CD
    xx12 = dot(HD, CD)      # HD'CD

    # X'y components
    xy1 = dot(HD, y)
    xy2 = dot(CD, y)

    # determinant
    det = xx11 * xx22 - xx12^2

    if det ≤ 1e-12
        error("Design matrix nearly singular (HD/CD collinear).")
    end

    # solve 2×2 system
    βh = ( xy1 * xx22 - xy2 * xx12) / det
    βc = (-xy1 * xx12 + xy2 * xx11) / det

    return βh, βc
end




function build_MA_errors_banded_gated(
    errors::Vector{Float64},
    Y::Vector{Float64},
    state::AbstractMatrix,
    ψ_expanded::Matrix{Float64},
    activeLags_ma::Vector{Int},
    p2, s2,
    p_max,
    σₑ²,
    σ0,
    T_use;
    ws_sma::SMAWorkspace,
    INTERCEPT::Bool = true,
    ztrans = "partials",
    rng::AbstractRNG = Random.default_rng(),
    presample_mode::Symbol = :posterior,
    use_σ0_for_presample::Bool = true,
    stopcol=stopcol,
    startcol=startcol
    
)
    T     = size(state, 1) - 1
    T_all = T * l
    q     = p_max[2]
    k     = length(activeLags_ma)

    # ----------------------------------
    # Baseline damping from MA dimension
    # ----------------------------------
    #r = k   # number of primitive MA modes
    r = stopcol - startcol + 1
    #w_base = 1.0 / sqrt(1.0 + 2 / 3.0)
    w_base= 0.5

    # ----------------------------------
    # Detrend
    # ----------------------------------
    y_detrended = similar(Y)

    if INTERCEPT
        intercept_grouped  = reshape(Float64.(state[2:end, 1]), 1, :)
        intercept_expanded = Matrix{Float64}(undef, 1, T_all)
        expand_grouped_states_fast!(intercept_expanded, intercept_grouped, l, T_all)

        @inbounds for t in 1:T_all
            y_detrended[t] = Y[t] - intercept_expanded[1, t]
        end
    else
        copyto!(y_detrended, Y)
    end

    # ----------------------------------
    # σ handling
    # ----------------------------------
    σe2 = σₑ² isa Number  ? Float64(σₑ²) :
          σₑ² isa Vector  ? Float64.(σₑ²) :
          σₑ² isa Matrix  ? vec(Float64.(σₑ²)) :
          error("σₑ² must be scalar, vector, or 1×T matrix")

    # ----------------------------------
    # Presample
    # ----------------------------------
    εprefix = view(errors, 1:q)
    u_out   = εprefix

    if presample_mode === :posterior
        sample_SMA_presample_u_info!(
            u_out,
            εprefix,
            view(y_detrended, 1:T_use),
            view(ψ_expanded, :, 1:T_use),
            activeLags_ma,
            σe2,
            σ0^2,
            q,
            T_use,
            ws_sma;
            rng = rng,
        )
    elseif presample_mode === :simple
        sample_SMA_presample_u_simple!(
            u_out,
            εprefix,
            σe2,
            σ0^2,
            q;
            rng = rng,
            use_σ0 = use_σ0_for_presample,
        )
    else
        error("Unknown presample_mode")
    end

    reverse!(u_out)

    # ----------------------------------
    # MAIN RECURSION WITH GATING
    # ----------------------------------
    @inbounds for t in 1:T_all
        j = q + t
        acc = 0.0
        ψt = @view ψ_expanded[:, t]

        # --- MA contribution ---
        @simd for i in 1:k
            lag = activeLags_ma[i]
            acc += ψt[i] * errors[j - lag]
        end

        # ----------------------------------
        # 1. Working residual
        # ----------------------------------
        r_work = y_detrended[t] - acc

        # ----------------------------------
        # 2. Standardize
        # ----------------------------------
        σt = σe2 isa Number ? sqrt(σe2) : sqrt(σe2[min(t, end)])
        #zt = r_work / max(σt, 1e-8)

        zt = errors_reg[t] / max(σt, 1e-8)

        # ----------------------------------
        # 3. Soft gating
        # ----------------------------------
        c = 2.0        # threshold
        #λ = 0.5        # strength

        #c = 0.0        # threshold
        λ = 1.0        # strength

        excess = max(0.0, abs(zt) - c)
        w_gate = 1.0 / (1.0 + λ * excess^2)

        # ----------------------------------
        # 4. Final weight (hybrid)
        # ----------------------------------
        #w_t = w_base * w_gate
        w_t = w_gate

        # ----------------------------------
        # 5. Update
        # ----------------------------------
        errors[j] = y_detrended[t] - w_t * acc
    end

    return errors
end




function sample_SMA_presample_u_simple!(
    u_out::AbstractVector{<:Real},
    εprefix::AbstractVector{<:Real},
    σe2::Union{Real,AbstractVector{<:Real}},
    σ0²::Real,
    q::Int;
    rng::AbstractRNG = Random.default_rng(),
    use_σ0::Bool = true,
)

    @assert length(u_out) == q
    @assert length(εprefix) >= q
    @assert σ0² > 0

    if use_σ0
        σ = sqrt(σ0²)
        @inbounds for i in 1:q
            u_out[i] = σ * randn(rng)
        end
    else
        if σe2 isa Number
            σ = sqrt(float(σe2))
            @inbounds for i in 1:q
                u_out[i] = σ * randn(rng)
            end
        else
            @assert length(σe2) >= q
            @inbounds for i in 1:q
                u_out[i] = sqrt(float(σe2[i])) * randn(rng)
            end
        end
    end

    εprefix[1:q] .= u_out
    return nothing
end
