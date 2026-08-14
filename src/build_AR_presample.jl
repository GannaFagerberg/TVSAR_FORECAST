using SparseArrays, LinearAlgebra, Random
using SparseArrays, LinearAlgebra, Random

struct PresampleWorkspace
    Q::SparseMatrixCSC{Float64,Int}
    posmap::Dict{Tuple{Int,Int},Int}
    b::Vector{Float64}
    idxs::Vector{Int}
    coefs::Vector{Float64}
    z::Vector{Float64}
    u::Vector{Float64}
end

function build_presample_workspace(maxlag::Int, activeLags::Vector{Int})
    bw = maxlag
    k  = length(activeLags)

    I = Int[]
    J = Int[]

    # always keep diagonal in the structure
    for i in 1:bw
        push!(I, i)
        push!(J, i)
    end

    # add all possible upper-triangular couplings
    for t in 1:bw
        unknown = Int[]

        for lag in activeLags
            tlag = t - lag
            if tlag <= 0
                push!(unknown, 1 - tlag)
            end
        end

        sort!(unknown)
        unique!(unknown)

        m = length(unknown)
        for a in 1:m
            ia = unknown[a]
            for b in a:m
                ib = unknown[b]
                push!(I, ia)
                push!(J, ib)
            end
        end
    end

    # use ones so entries definitely exist structurally
    Q = sparse(I, J, ones(Float64, length(I)), bw, bw)

    # enforce diagonal structurally
    for i in 1:bw
        Q[i,i] = 1.0
    end

    # zero numeric values, keep structure
    fill!(Q.nzval, 0.0)

    posmap = Dict{Tuple{Int,Int},Int}()
    for col in 1:size(Q,2)
        for ptr in Q.colptr[col]:(Q.colptr[col+1]-1)
            row = Q.rowval[ptr]
            posmap[(row,col)] = ptr
        end
    end

    return PresampleWorkspace(
        Q,
        posmap,
        zeros(bw),
        Vector{Int}(undef, k),
        Vector{Float64}(undef, k),
        zeros(bw),
        zeros(bw)
    )
end

function sample_AR_presample!(
    ws::PresampleWorkspace,
    y_obs,
    phi_mat,
    activeLags,
    c,
    σ2;
    m0,
    P0_diag,
    rng = Random.default_rng(),
)

    p  = length(y_obs)
    k  = length(activeLags)

    u      = ws.u
    bw     = length(u)
    Q      = ws.Q
    b      = ws.b
    idxs   = ws.idxs
    coefs  = ws.coefs
    posmap = ws.posmap

    fill!(Q.nzval, 0.0)
    fill!(b, 0.0)

    invP0 = 1.0 / float(P0_diag)

    # --------------------------------------------------
    # Prior contribution
    # --------------------------------------------------
    @inbounds for i in 1:bw
        pos = posmap[(i,i)]
        Q.nzval[pos] += invP0
        b[i] += invP0 * float(m0[i])
    end

    σ2_is_scalar = σ2 isa Real
    invσ2_scalar = σ2_is_scalar ? 1.0 / float(σ2) : 0.0

    @inline idx_from_tlag(tlag::Int) = 1 - tlag

    # --------------------------------------------------
    # Likelihood contribution
    # --------------------------------------------------
    @inbounds for t in 1:p

        invσ2 = σ2_is_scalar ? invσ2_scalar : 1.0 / float(σ2[t])
        rhs   = float(y_obs[t]) - float(c[t])

        m = 0

        for j in 1:k
            lag  = activeLags[j]
            ϕ    = float(phi_mat[j,t])
            tlag = t - lag

            if tlag >= 1
                rhs -= ϕ * float(y_obs[tlag])
            else
                m += 1
                ia = idx_from_tlag(tlag)
                idxs[m]  = ia
                coefs[m] = ϕ
            end
        end

        for a in 1:m

            ia = idxs[a]
            ca = coefs[a]

            b[ia] += invσ2 * ca * rhs

            pos = posmap[(ia,ia)]
            Q.nzval[pos] += invσ2 * ca * ca

            for b2 in (a+1):m

                ib = idxs[b2]
                cb = coefs[b2]

                iU = ia <= ib ? ia : ib
                jU = ia <= ib ? ib : ia

                pos = posmap[(iU,jU)]

                Q.nzval[pos] += invσ2 * ca * cb

            end
        end
    end

    # --------------------------------------------------
    # Sparse Cholesky factorization
    # --------------------------------------------------
    F = cholesky(Symmetric(Q, :U))

    # --------------------------------------------------
    # Posterior mean
    # --------------------------------------------------
    u .= F \ b

    # --------------------------------------------------
    # Gaussian perturbation
    # --------------------------------------------------
    #randn!(rng, ws.z)
    #u .+= F' \ ws.z

    # Gaussian perturbation: covariance Q^{-1}
    randn!(rng, ws.z)
    u .+= F.U \ ws.z

    return u
end

function build_AR_init_opt_orig!(
    x0_out::AbstractVector,                 # length maxlag
    int_exp::AbstractVector,                # length maxlag
    m0_buf::Vector{Float64},                # length maxlag (reused)
    group_map::AbstractVector{Int},         # length maxlag
    ws::PresampleWorkspace,                 # workspace
    Y::AbstractVector,
    state::AbstractMatrix,
    ϕ_expanded::AbstractMatrix{Float64},    # k × T
    activeLags_ar,
    p1, s1,
    p_max,
    σₑ²,
    σy;
    INTERCEPT::Bool = true,
    cond_sma = nothing,
    Z::AbstractMatrix,
    l::Int,
    rng::AbstractRNG = Random.default_rng(),
)

    maxlag = maximum(activeLags_ar)
    k      = length(activeLags_ar)
    p      = maxlag

    @assert length(x0_out) >= p
    @assert length(int_exp) >= p
    @assert length(m0_buf) >= p
    @assert length(group_map) >= p
    @assert size(ϕ_expanded,1) == k
    @assert size(ϕ_expanded,2) >= p
    @assert length(Y) >= p
    @assert size(Z,2) == k

    # --------------------------------------------------
    # 1) Intercept vector for presample
    # --------------------------------------------------
    if INTERCEPT

        ng = group_map[p]

        int_grouped_all = @view state[2:end,1]

        @assert length(int_grouped_all) >= ng "Need $ng grouped intercepts"

        int_grouped = @view int_grouped_all[1:ng]

        expand_grouped_intercept!(
            int_exp,
            int_grouped,
            group_map,
            p
        )

        fill!(m0_buf, float(Y[1]))

    else

        fill!(int_exp,0.0)
        fill!(m0_buf,0.0)

    end

    c = @view int_exp[1:p]

    # --------------------------------------------------
    # 2) Innovation variances
    # --------------------------------------------------
    σ2 = σₑ² isa Number ? σₑ² : @view σₑ²[1:p]

    # --------------------------------------------------
    # 3) AR coefficients
    # --------------------------------------------------
    phi_first = @view ϕ_expanded[:,1:p]

    # --------------------------------------------------
    # 4) Sample presample vector
    # --------------------------------------------------
    u = sample_AR_presample!(
        ws,
        @view(Y[1:p]),
        phi_first,
        activeLags_ar,
        c,
        σ2;
        m0 = @view(m0_buf[1:p]),
        P0_diag = σy,
        rng = rng
    )

    # --------------------------------------------------
    # 5) Copy into output
    # --------------------------------------------------
    @inbounds copyto!(view(x0_out,1:p),u)

    if cond_sma !== nothing

        #@assert length(cond_sma) >= p

        @inbounds for i in 1:p
            x0_out[i] += cond_sma[i]
        end

    end

    # --------------------------------------------------
    # 6) Build first regression row
    # --------------------------------------------------
    @inbounds Z[1,:] .= @view x0_out[activeLags_ar]

    return reverse!(view(x0_out,1:p)), Z

end
