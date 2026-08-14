# ------------------------------------------------------------------------------
# Build group map ONCE for given (p, l): group_map[t] = (t-1)÷l + 1
# ------------------------------------------------------------------------------
function build_group_map(p::Int, nPerGroup::Int)
    group_map = Vector{Int}(undef, p)
    @inbounds for t in 1:p
        group_map[t] = (t - 1) ÷ nPerGroup + 1
    end
    return group_map
end


#### Group vector into chunks of size k
function group_vector(vec, k)
    n = length(vec)
    ng = cld(n, k)
    groups = Vector{Vector{eltype(vec)}}(undef, ng)

    idx = 1
    @inbounds for i in 1:k:n
        groups[idx] = vec[i:min(i+k-1, n)]   # still makes a copy
        idx += 1
    end

    return groups
end


function group_vector_view(vec, k)
    n  = length(vec)
    ng = cld(n, k)

    groups = Vector{SubArray{eltype(vec),1,typeof(vec),Tuple{UnitRange{Int}},true}}(undef, ng)

    @inbounds for (idx, i) in enumerate(1:k:n)
        groups[idx] = @view vec[i:min(i+k-1, n)]
    end

    return groups
end


# ------------------------------------------------------------------------------
# Expand grouped intercepts into first p values (in-place, no alloc)
# ------------------------------------------------------------------------------
function expand_grouped_intercept!(
    out::AbstractVector,
    int_grouped::AbstractVector,
    group_map::AbstractVector{Int},
    p::Int
)
    @assert length(out) >= p
    @assert length(group_map) >= p
    ng = group_map[p]
    @assert length(int_grouped) >= ng  "Need $ng groups, got $(length(int_grouped))"
    @inbounds for t in 1:p
        out[t] = int_grouped[group_map[t]]
    end
    return nothing
end



function compute_resid!(
    resid::AbstractVector{Float64},
    y::AbstractVector{Float64},
    reg_terms::AbstractVector{Float64},
    Cargs,
    θ₀::Float64
)
    ny = length(y)
    @inbounds for j in 1:ny
        z = Cargs[j]
        s = 0.0
        @simd for i in eachindex(reg_terms)
            s += reg_terms[i] * z[i]
        end
        resid[j] = y[j] - θ₀ - s
    end
    return resid
end

function clamp_partials!(μ, startcol, p_threshold)
    @inbounds for i in startcol:length(μ)
        x = μ[i]
        if x > p_threshold
            μ[i] = p_threshold
        elseif x < -p_threshold
            μ[i] = -p_threshold
        else
            μ[i] = x
        end
    end
    return μ
end


function expand_grouped_states_fast!(
    out::AbstractMatrix,
    ϕ_mat::AbstractMatrix,
    l::Int,
    T_all::Int
)
    k, T = size(ϕ_mat)

    col = 1
    @inbounds for t in 1:T
        for _ in 1:l
            col > T_all && return out

            @inbounds @simd for j in 1:k
                out[j, col] = ϕ_mat[j, t]
            end

            col += 1
        end
    end

    return out
end


function compute_conditional_mean!(
    cond_mean::Vector{Float64},
    Z::AbstractMatrix,
    ϕ_expanded::AbstractMatrix,
    state::AbstractMatrix,
    group_map::Vector{Int};
    INTERCEPT::Bool = true
)
    T, k = size(Z)

    if INTERCEPT
        intercept = @view state[2:end, 1]

        @inbounds for t in 1:T
            s = 0.0
            @simd for j in 1:k
                s += Z[t, j] * ϕ_expanded[j, t]
            end
            cond_mean[t] = s + intercept[group_map[t]]
        end
    else
        @inbounds for t in 1:T
            s = 0.0
            @simd for j in 1:k
                s += Z[t, j] * ϕ_expanded[j, t]
            end
            cond_mean[t] = s
        end
    end

    return cond_mean
end

function compute_residuals!(residuals::Vector{Float64},
                            Y::Vector{Float64},
                            cond_mean::Vector{Float64})
    @inbounds for t in eachindex(Y, cond_mean, residuals)
        residuals[t] = Y[t] - cond_mean[t]
    end
    return residuals
end

function compute_noise_SARMA(alpha_sigma_hat, beta_sigma, errors)
    # Sample sigma² given beta
    beta_sigma_hat = beta_sigma + 0.5 * sum(errors .^ 2)
    sigma2 = 1 / rand(Gamma(alpha_sigma_hat, 1 / beta_sigma_hat))
    return sqrt(sigma2)
end

### R ARIMA fit
function Arima(y; order = [0,0,0], seasonal = [0,0,0], xreg = nothing, include_mean = false,
    include_drift = true, include_constant = true, frequency = 1, deltat = 1)
    R"""
    suppressMessages(library(forecast))
    fittedModel = Arima(ts($y, frequency = $frequency, deltat = $deltat), order = $order, seasonal = $seasonal, xreg = $xreg, 
        include.mean = $include_mean, include.drift = $include_drift, include.constant = $include_constant)
    """
    @rget fittedModel
    return fittedModel
end

