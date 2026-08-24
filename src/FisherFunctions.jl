function FisherInfo_full_global_gaussian(
    μ,
    σ²,
    Cargs,
    cache,
    ws;
    INTERCEPT::Bool = true,
    startcol::Int = INTERCEPT ? 2 : 1,
    ztrans::AbstractString = "partials",
    negative_signs::Bool = true
)

    Tμ = eltype(μ)
    p  = length(μ)

    FI = zeros(Tμ, p, p)

    # ----------------------------------------------------------
    # Reuse exactly the same buffers as in the IEKF
    # ----------------------------------------------------------
    C̄        = ws.C̄
    reg_terms = ws.reg_terms
    Jg        = ws.Jg

    # ----------------------------------------------------------
    # State components entering the SAR transformation
    #
    # RW intercept:
    #   μ = [c, θ_AR...],       startcol = 2
    #
    # no intercept:
    #   μ = [θ_AR...],          startcol = 1
    #
    # local-linear intercept:
    #   μ = [c, d, θ_AR...],    startcol = 3
    # ----------------------------------------------------------
    θ_work = INTERCEPT ? @view(μ[startcol:end]) : μ

    # ----------------------------------------------------------
    # Transform latent SAR states to expanded AR coefficients
    #
    # This is exactly the same transformation used in FFBSx.
    # ----------------------------------------------------------
    MultiSARMAtoReg_cached!(
        reg_terms,
        θ_work,
        cache;
        ztrans = ztrans,
        negative_signs = negative_signs
    )

    invσ² = one(Tμ) / Tμ(σ²)

    # ----------------------------------------------------------
    # Sum Fisher contributions over ALL groups / observations
    # ----------------------------------------------------------
    for t in eachindex(Cargs)

        Cargs_t = Cargs[t]

        # ------------------------------------------------------
        # Construct
        #
        #       C̄[i,:] = ∂η_{t,i} / ∂μ
        #
        # using exactly the same Jacobian routine as the IEKF.
        # ------------------------------------------------------
        jacobian_C_fast!(
            C̄,
            reg_terms,
            Jg,
            μ,
            Cargs_t,
            cache;
            INTERCEPT = INTERCEPT,
            ztrans = ztrans,
            startcol = startcol,
            negative_signs = negative_signs
        )

        ng = length(Cargs_t)

        # Only rows corresponding to observations in this group
        Cg = @view C̄[1:ng, :]

        # ------------------------------------------------------
        # Gaussian Fisher:
        #
        #    FI_t = (1/σ²) Cg' Cg
        #
        # Accumulate without constructing Cg' * Cg explicitly.
        # ------------------------------------------------------
        mul!(
            FI,
            transpose(Cg),
            Cg,
            invσ²,
            one(Tμ)
        )
    end

    return Symmetric(FI)
end

### Full local Fisher information: Gaussian TVSAR
function FisherInfo_full_local_gaussian(
    μ,
    σ²,
    t,
    Cargs,
    cache,
    ws;
    INTERCEPT::Bool = true,
    startcol::Int = INTERCEPT ? 2 : 1,
    ztrans::AbstractString = "partials",
    negative_signs::Bool = true
)

    Tμ = eltype(μ)
    p  = length(μ)

    FI = zeros(Tμ, p, p)

    # ----------------------------------------------------------
    # Reuse exactly the same buffers as in the IEKF
    # ----------------------------------------------------------
    C̄        = ws.C̄
    reg_terms = ws.reg_terms
    Jg        = ws.Jg

    # ----------------------------------------------------------
    # State components entering the SAR transformation
    #
    # RW intercept:
    #   μ = [c, θ_AR...],       startcol = 2
    #
    # no intercept:
    #   μ = [θ_AR...],          startcol = 1
    #
    # local-linear intercept:
    #   μ = [c, d, θ_AR...],    startcol = 3
    # ----------------------------------------------------------
    θ_work = INTERCEPT ? @view(μ[startcol:end]) : μ

    # ----------------------------------------------------------
    # Transform latent SAR states to expanded AR coefficients
    #
    # Exactly the same transformation as in FFBSx.
    # ----------------------------------------------------------
    MultiSARMAtoReg_cached!(
        reg_terms,
        θ_work,
        cache;
        ztrans = ztrans,
        negative_signs = negative_signs
    )

    # ----------------------------------------------------------
    # Current group only
    # ----------------------------------------------------------
    Cargs_t = Cargs[t]

    # ----------------------------------------------------------
    # Construct
    #
    #       C̄[i,:] = ∂η_{t,i} / ∂μ
    #
    # for observations belonging to group t.
    # ----------------------------------------------------------
    jacobian_C_fast!(
        C̄,
        reg_terms,
        Jg,
        μ,
        Cargs_t,
        cache;
        INTERCEPT = INTERCEPT,
        ztrans = ztrans,
        startcol = startcol,
        negative_signs = negative_signs
    )

    ng = length(Cargs_t)

    # Only rows corresponding to observations in this group
    Cg = @view C̄[1:ng, :]

    # ----------------------------------------------------------
    # Gaussian LOCAL Fisher:
    #
    #       FI_t = (1/σ²) Cg' Cg
    #
    # This is the TOTAL information in group t.
    # ----------------------------------------------------------
    invσ² = one(Tμ) / Tμ(σ²)

    mul!(
        FI,
        transpose(Cg),
        Cg,
        invσ²,
        zero(Tμ)
    )

    return Symmetric(FI)
end


### Full initial Fisher information: Gaussian TVSAR
function FisherInfo_full_initial_gaussian(
    μ,
    X,
    σ²,
    cache;
    INTERCEPT::Bool = true,
    startcol::Int = INTERCEPT ? 2 : 1,
    ztrans::AbstractString = "partials",
    negative_signs::Bool = true
)

    Tμ = eltype(μ)
    p  = length(μ)

    nobs      = size(X, 1)
    nregterms = size(X, 2)

    # ----------------------------------------------------------
    # State components entering SAR transformation
    #
    # RW intercept:
    #   μ = [c, θ_AR...]       startcol = 2
    #
    # LL intercept:
    #   μ = [c, d, θ_AR...]    startcol = 3
    #
    # No intercept:
    #   μ = [θ_AR...]          startcol = 1
    # ----------------------------------------------------------

    θ_work = INTERCEPT ?
             @view(μ[startcol:end]) :
             μ

    nsar_state = length(θ_work)


    # ----------------------------------------------------------
    # Allocate buffers with the EXACT required dimensions
    # ----------------------------------------------------------

    # Expanded SAR coefficients
    reg_terms = zeros(Tμ, nregterms)

    # Jacobian of expanded SAR coefficients wrt latent SAR states
    #
    #     Jg = ∂ reg_terms / ∂ θ_work'
    #
    Jg = zeros(Tμ, nregterms, nsar_state)

    # Jacobian of observation means wrt complete state μ
    #
    #     C̄[i,:] = ∂η_i / ∂μ
    #
    C̄ = zeros(Tμ, nobs, p)


    # ----------------------------------------------------------
    # Transform latent SAR states to expanded AR coefficients
    # ----------------------------------------------------------

    MultiSARMAtoReg_cached!(
        reg_terms,
        θ_work,
        cache;
        ztrans = ztrans,
        negative_signs = negative_signs
    )


    # ----------------------------------------------------------
    # Observation-specific regressors
    # ----------------------------------------------------------

    Cargs_initial = [
        @view(X[i, :]) for i in axes(X, 1)
    ]


    # ----------------------------------------------------------
    # Construct observation Jacobian
    #
    # Internally:
    #
    #     Jg = ∂ reg_terms / ∂ θ_work'
    #
    # followed by
    #
    #     C̄[i,:] = ∂η_i / ∂μ
    # ----------------------------------------------------------

    jacobian_C_fast!(
        C̄,
        reg_terms,
        Jg,
        μ,
        Cargs_initial,
        cache;
        INTERCEPT = INTERCEPT,
        ztrans = ztrans,
        startcol = startcol,
        negative_signs = negative_signs
    )


    # ----------------------------------------------------------
    # Gaussian Fisher information
    #
    #     FI = (1/σ²) C̄' C̄
    # ----------------------------------------------------------

    FI = zeros(Tμ, p, p)

    invσ² = one(Tμ) / Tμ(σ²)

    mul!(
        FI,
        transpose(C̄),
        C̄,
        invσ²,
        zero(Tμ)
    )

    return Symmetric(FI)
end

function prior_t0_gaussian(
    priorparam_μ,
    κ₀,
    nparams,
    X,
    σ²,
    cache;
    INTERCEPT::Bool = true,
    startcol::Int = INTERCEPT ? 2 : 1,
    ztrans::AbstractString = "partials",
    negative_signs::Bool = true,
    ridge::Float64 = 1e-8,
    FisherInfo_initial = FisherInfo_full_initial_gaussian
)

    # ----------------------------------------------------------
    # Prior mean
    #
    # Gaussian TVSAR:
    #
    #     y_t = η_t + ε_t
    #
    # so the intercept is already on the observation scale.
    #
    # With latent SAR states equal to zero, the transformed
    # AR coefficients are zero under the usual parametrization.
    # ----------------------------------------------------------

    if INTERCEPT

        μ₀ = zeros(Float64, nparams)
        μ₀[1] = priorparam_μ

    else

        μ₀ = zeros(Float64, nparams)

    end


    # ----------------------------------------------------------
    # Initial Fisher information
    #
    #     FI = (1/σ²) Σ_i J_i J_i'
    # ----------------------------------------------------------

    FI = FisherInfo_initial(
        μ₀,
        X,
        σ²,
        cache;
        INTERCEPT = INTERCEPT,
        startcol = startcol,
        ztrans = ztrans,
        negative_signs = negative_signs
    )


    # ----------------------------------------------------------
    # Average Fisher information per observation
    #
    #     FI_bar = FI / nobs
    # ----------------------------------------------------------

    if FI isa AbstractVector
        FI = Diagonal(FI ./ size(X, 1))
    else
        FI = Matrix(FI) ./ size(X, 1)
    end


    # ----------------------------------------------------------
    # Optional numerical regularization
    # ----------------------------------------------------------

    if ridge > 0.0
        @inbounds for k in axes(FI, 1)
            FI[k, k] += ridge
        end
    end

    # ----------------------------------------------------------
    # Fisher-based prior covariance
    #
    #     Σ₀ = (1/κ₀) FI_bar^{-1}
    # ----------------------------------------------------------

    Σ₀ =
        (1 / κ₀) .*
        inv(Symmetric(FI))

    return μ₀, Hermitian(Σ₀)
end