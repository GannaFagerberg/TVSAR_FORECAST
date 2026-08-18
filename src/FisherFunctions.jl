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