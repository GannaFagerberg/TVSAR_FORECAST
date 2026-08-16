############################################################
# Active Lags for a General Multi-Seasonal SAR Process
############################################################

# Find the active lags in a general multi-seasonal SAR(p,s) process.
#
# s : vector containing the seasonal periods
# p : vector containing the AR order for each seasonal polynomial
#
# Example:
#   p = [1, 1, 1]
#   s = [1, 24, 168]
#
# The function constructs each seasonal AR polynomial, multiplies
# them, and returns the lags corresponding to non-zero coefficients.
function FindActiveLagsMultiSAR(p::AbstractVector, s::AbstractVector)

    ARpolynomials = Array{Polynomial}(undef, length(p))

    for (j, pⱼ) ∈ enumerate(p)

        ARpolymat = [
            zeros(s[j] - 1, pⱼ)
            -ones(1, pⱼ)
        ]

        ARpolynomials[j] = Polynomial(
            [1; ARpolymat[:]],
            :z
        )
    end

    return (findall(coeffs(prod(ARpolynomials)) .!= 0) .- 1)[2:end]

    # return (findall(Polynomials.coeffs(prod(ARpolynomials)) .!= 0) .- 1)[2:end]
end



############################################################
# Construct AR Regression Matrix Using Only Active Lags
############################################################

# Construct the response vector y and regression matrix Z
# using only the lags specified in activeLags.
#
# The first maxlag observations are treated as presample values.
function SetupARReg_active(x, activeLags)

    maxlag = maximum(activeLags)
    T_full = length(x)

    # Effective sample size after accounting for the largest lag
    T = T_full - maxlag

    # Response vector
    y = x[(maxlag + 1):end]

    # Number of active regressors
    p = length(activeLags)

    # Regression matrix
    Z = zeros(T, p)

    @inbounds for (j, lag) in enumerate(activeLags)

        Z[:, j] .= x[
            (maxlag - lag + 1):(T_full - lag)
        ]

    end

    return y, Z, T
end



function SetupARReg(x, p)
    T = length(x)

    # Set up lags
    y = x[p+1:end]
    Z = zeros(T, p)
    for i = 1:p
        Z[:,i] = [zeros(i); x[1:(end-i)]]
    end
    Z = Z[(p+1):end,:]
    T = T - p # Redefining time here!()

    return y, Z, T
end


############################################################
# Cache for Multi-Seasonal SARMA Reparameterization
############################################################

# Stores quantities that are repeatedly used when converting
# unconstrained SARMA parameters into regression coefficients.
#
# The buffers avoid repeated memory allocation inside MCMC,
# filtering, or likelihood evaluations.
struct SARMARegCache{T}

    # Seasonal periods
    s::Vector{Int}

    # Orders corresponding to each seasonal polynomial
    p::Vector{Int}

    # Active lags in the full multiplicative polynomial
    activeLags::Vector{Int}

    # Locations of coefficients for each seasonal polynomial
    positions::Vector{Vector{Int}}

    # Maximum lag in the complete multiplicative polynomial
    L::Int

    # Polynomial coefficient buffers
    coeffs::Vector{T}
    newcoeffs::Vector{T}

    # Temporary buffers used by the AR reparameterization
    φtmp::Vector{T}
    Ptmp::Vector{T}
    ϕtmp::Matrix{T}
end



############################################################
# Build SARMA Regression Cache
############################################################

function build_sarma_cache(p, s, activeLags; T = Float64)

    # Maximum lag implied by all seasonal polynomials
    L = sum(p .* s)

    # Positions of coefficients associated with each
    # seasonal polynomial
    positions = Vector{Vector{Int}}(undef, length(s))

    for l in 1:length(s)

        pos = Vector{Int}(undef, p[l])

        for j in 1:p[l]
            pos[j] = j * s[l]
        end

        positions[l] = pos
    end

    # Largest AR order across the seasonal components
    maxp = maximum(p)

    return SARMARegCache{T}(
        s,
        p,
        activeLags,
        positions,
        L,

        zeros(T, L + 1),         # coeffs buffer
        zeros(T, L + 1),         # newcoeffs buffer

        zeros(T, maxp),          # φtmp
        zeros(T, maxp),          # Ptmp
        zeros(T, maxp, maxp)     # ϕtmp
    )
end

############################################################
# AR Reparameterization
############################################################

# Transform unconstrained parameters x into stable AR
# coefficients using the selected transformation.
#
# Available transformations:
#   "sigmoid"
#   "monahan"
#   "partials"
#   "linear"
#
# The Durbin-Levinson recursion is then used to map the
# partial autocorrelations into AR coefficients.
function arma_reparam!(
    φout,
    P,
    ϕmat,
    x;
    ztrans = "monahan",
    threshold = nothing,
    negative_signs = true
)

    p = length(x)


    # ------------------------------------------------------
    # Transform unconstrained parameters
    # ------------------------------------------------------

    if ztrans == "sigmoid"

        @inbounds for i in 1:p
            P[i] = tanh(x[i] / 2)
        end

    elseif ztrans == "monahan"

        @inbounds for i in 1:p
            xi = x[i]
            P[i] = xi / sqrt(1 + xi * xi)
        end

    elseif ztrans == "partials"

        @inbounds for i in 1:p
            P[i] = clamp(x[i], -0.9999, 0.9999)
        end

    elseif ztrans == "linear"

        copyto!(P, x)

    else

        error("ztrans must be sigmoid/monahan/partials/linear")

    end


    # ------------------------------------------------------
    # Optional additional thresholding
    # ------------------------------------------------------

    if threshold !== nothing

        @inbounds for i in 1:p
            P[i] = clamp(P[i], -threshold, threshold)
        end

    end


    # ------------------------------------------------------
    # Sign convention
    # ------------------------------------------------------

    if negative_signs

        @inbounds for i in 1:p
            P[i] = -P[i]
        end

    end


    # ------------------------------------------------------
    # Durbin-Levinson recursion
    # ------------------------------------------------------

    fill!(ϕmat, zero(eltype(P)))

    ϕmat[1, 1] = P[1]

    @inbounds for k = 2:p

        pk = P[k]

        # Inner recursion
        @simd for j = 1:k-1

            ϕmat[k, j] =
                ϕmat[k-1, j] +
                pk * ϕmat[k-1, k-j]

        end

        # Final element
        ϕmat[k, k] = pk

    end


    # ------------------------------------------------------
    # Store final AR coefficients
    # ------------------------------------------------------

    if negative_signs

        @inbounds @simd for j in 1:p
            φout[j] = -ϕmat[p, j]
        end

    else

        @inbounds @simd for j in 1:p
            φout[j] = ϕmat[p, j]
        end

    end

    return nothing
end



############################################################
# Allocating Wrapper for AR Reparameterization
############################################################

function arma_reparam(
    x;
    ztrans = "monahan",
    threshold = nothing,
    negative_signs = true
)

    p = length(x)

    φout = similar(x)
    P    = similar(x)
    ϕmat = similar(x, p, p)

    arma_reparam!(
        φout,
        P,
        ϕmat,
        x;
        ztrans = ztrans,
        threshold = threshold,
        negative_signs = negative_signs
    )

    return φout, P
end



############################################################
# Multi-Seasonal SARMA Parameters -> Regression Coefficients
# Cached / In-Place Version
############################################################

# Convert the parameter vector θ into the regression
# coefficients implied by the complete multiplicative
# multi-seasonal SARMA polynomial.
#
# The cache contains all temporary buffers required for the
# polynomial construction, avoiding repeated allocations.
function MultiSARMAtoReg_cached!(
    out::AbstractVector,
    θ::AbstractVector,
    cache::SARMARegCache;
    ztrans::AbstractString = "partials",
    threshold = nothing,
    negative_signs::Bool = true
)

    # ------------------------------------------------------
    # Cached buffers
    # ------------------------------------------------------

    coeffs    = cache.coeffs
    newcoeffs = cache.newcoeffs
    L         = cache.L

    φtmp = cache.φtmp
    Ptmp = cache.Ptmp
    ϕtmp = cache.ϕtmp


    # ------------------------------------------------------
    # Initialize polynomial
    # ------------------------------------------------------

    fill!(coeffs, zero(eltype(coeffs)))
    coeffs[1] = one(eltype(coeffs))

    count = 0

    δ = eps(real(eltype(coeffs)))


    # ------------------------------------------------------
    # Loop over seasonal AR polynomials
    # ------------------------------------------------------

    @inbounds for l in eachindex(cache.s)

        pl = cache.p[l]

        pl == 0 && continue


        # Parameters belonging to seasonal polynomial l
        θview = @view θ[count+1 : count+pl]


        # Reset temporary buffers
        fill!(φtmp, zero(eltype(φtmp)))
        fill!(Ptmp, zero(eltype(Ptmp)))
        fill!(ϕtmp, zero(eltype(ϕtmp)))


        # Transform parameters into AR coefficients
        arma_reparam!(
            φtmp,
            Ptmp,
            ϕtmp,
            θview;
            ztrans = ztrans,
            threshold = threshold,
            negative_signs = negative_signs
        )


        # Match arma_reparam(...)[1] .+ eps()
        for j in 1:pl
            φtmp[j] += δ
        end


        # --------------------------------------------------
        # Build one seasonal block polynomial:
        #
        # 1 + coef1*z^s + coef2*z^(2s) + ... + coefp*z^(ps)
        # --------------------------------------------------

        copyto!(newcoeffs, coeffs)

        fill!(newcoeffs, zero(eltype(newcoeffs)))

        for i in 1:(L + 1)

            ci = coeffs[i]

            ci == 0 && continue

            # Multiply by the constant term 1
            newcoeffs[i] += ci


            # Add seasonal polynomial terms
            for j in 1:pl

                lag  = cache.positions[l][j]
                coef = negative_signs ? -φtmp[j] : φtmp[j]

                if i + lag <= L + 1
                    newcoeffs[i + lag] += ci * coef
                end

            end

        end


        # Store updated multiplicative polynomial
        copyto!(coeffs, newcoeffs)

        count += pl

    end


    # ------------------------------------------------------
    # Extract coefficients corresponding to active lags
    # ------------------------------------------------------

    @inbounds for i in eachindex(cache.activeLags)

        idx = 1 + cache.activeLags[i]

        out[i] =
            negative_signs ? -coeffs[idx] : coeffs[idx]

    end

    return nothing
end

############################################################
# Allocating Wrapper for Multi-Seasonal SARMA Conversion
############################################################

function MultiSARMAtoReg_cached(
    θ::AbstractVector,
    cache::SARMARegCache;
    ztrans = "partials",
    threshold = nothing,
    negative_signs = true
)

    out = similar(
        θ,
        length(cache.activeLags)
    )

    MultiSARMAtoReg_cached!(
        out,
        θ,
        cache;
        ztrans = ztrans,
        threshold = threshold,
        negative_signs = negative_signs
    )

    return out
end