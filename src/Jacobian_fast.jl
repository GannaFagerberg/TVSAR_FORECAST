
mutable struct JacobianWorkspace{T,CFG,F}
    ψtmp::Vector{T}
    cfg::CFG
    g!::F
end


function build_jacobian_workspace(cache_ar, pfit::Int;
    T = Float64,
    ztrans = "partials",
    negative_signs = true
)
    ψtmp = zeros(T, length(cache_ar.activeLags))

    function g!(out, θ)
        cache_tmp = build_sarma_cache(
            cache_ar.p, cache_ar.s, cache_ar.activeLags;
            T = eltype(θ)
        )
        MultiSARMAtoReg_cached!(
            out,
            θ,
            cache_tmp;
            ztrans = ztrans,
            negative_signs = negative_signs
        )
        return out
    end

    θ0 = zeros(T, pfit)
    y0 = zeros(T, length(cache_ar.activeLags))

    cfg = ForwardDiff.JacobianConfig(g!, y0, θ0)

    return JacobianWorkspace(ψtmp, cfg, g!)
end


function jacobian_reg_terms!(
    Jg,
    θ_work,
    cache;
    ztrans="partials",
    negative_signs=true
)
    cache_ref = Ref{Any}(nothing)

    function g!(out, θ)
        Tθ = eltype(θ)

        c = cache_ref[]
        if c === nothing || !(c isa SARMARegCache{Tθ})
            cache_ref[] = build_sarma_cache(cache.p, cache.s, cache.activeLags; T=Tθ)
        end

        MultiSARMAtoReg_cached!(
            out,
            θ,
            cache_ref[];
            ztrans=ztrans,
            negative_signs=negative_signs
        )
        return out
    end

    ψtmp = similar(cache.activeLags, eltype(θ_work))
    ForwardDiff.jacobian!(Jg, g!, ψtmp, θ_work)

    return nothing
end

function jacobian_C_fast!(
    C̄,
    reg_terms,
    Jg,
    μ̄,
    Cargs,
    cache;
    INTERCEPT::Bool,
    ztrans::AbstractString,
    startcol::Int,
    negative_signs::Bool=true
)
    ny = length(Cargs)
    n  = length(μ̄)

    θ_work = INTERCEPT ? @view(μ̄[startcol:end]) : μ̄
    pfit   = length(θ_work)

    jacobian_reg_terms!(Jg, θ_work, cache; ztrans=ztrans, negative_signs=negative_signs)

    @assert size(Jg, 2) == pfit
    @assert size(C̄, 1) == ny
    @assert size(C̄, 2) == n
    @assert length(reg_terms) == size(Jg, 1)

    fill!(C̄, zero(eltype(C̄)))

    if INTERCEPT
        @inbounds for j in 1:ny
            C̄[j,1] = one(eltype(C̄))
        end
    end

    col0 = INTERCEPT ? startcol : 1

    @inbounds for j in 1:ny
        z = Cargs[j]
        @assert length(z) == size(Jg, 1)

        for k in 1:pfit
            s = zero(eltype(C̄))
            @simd for i in axes(Jg, 1)
                s += z[i] * Jg[i,k]
            end
            C̄[j, col0 + k - 1] = s
        end
    end

    return nothing
end

