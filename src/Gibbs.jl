

# ============================================================
# IEKF workspace (UPDATED)
# - Keeps your old (S, K, Joseph) path
# - Adds extra buffers for the "ny-large" exact information-form path
# ============================================================

struct IEKFWorkspace{T}
    # --- old path buffers ---
    C̄::Matrix{T}          # ny×n
    resid::Vector{T}       # ny
    K::Matrix{T}           # n×ny
    tmp_ny_n::Matrix{T}    # ny×n
    tmp_n_ny::Matrix{T}    # n×ny
    tmp_n_n::Matrix{T}     # n×n
    tmp_n_n2::Matrix{T}    # n×n
    S::Matrix{T}           # ny×ny
    U::Matrix{T}           # ny×ny
    I_n::Matrix{T}         # n×n

    reg_terms::Vector{T}   # nreg
    Jg::Matrix{T}          # nreg×pfit  (your jacobian buffer)

    # --- new path buffers (ny-large info-form, exact) ---
    tmp_ny::Vector{T}      # ny   (R^{-1} * resid)
    b::Vector{T}           # n    (C' R^{-1} resid)
    δ::Vector{T}           # n    (state increment)
    W::Matrix{T}           # n×n  (C' R^{-1} C)
    A::Matrix{T}           # n×n  (I + Ω̄ W)
    X::Matrix{T}           # n×n  (scratch for solves/products)

     # --- NEW sequential EKF buffer ---
    v::Vector{T}           # n   (Σ * c_j)

    #v::Vector{T}      # n
    δμ::Vector{T}     # n
    
end


function IEKFWorkspace(n::Int, ny::Int, nreg::Int, pfit::Int; T=Float64)
    IEKFWorkspace{T}(
        # old
        zeros(T, ny, n),
        zeros(T, ny),
        zeros(T, n, ny),
        zeros(T, ny, n),
        zeros(T, n, ny),
        zeros(T, n, n),
        zeros(T, n, n),
        zeros(T, 1, 1), # for seq.
        zeros(T, 1, 1), # for seq.
        #zeros(T, ny, ny),
        #zeros(T, ny, ny),
        Matrix{T}(LinearAlgebra.I, n, n),

        zeros(T, nreg),
        zeros(T, nreg, pfit),

        # new
        zeros(T, ny),
        zeros(T, n),
        zeros(T, n),
        zeros(T, n, n),
        zeros(T, n, n),
        zeros(T, n, n),
        
        # --- NEW sequential EKF buffer ---
        #zeros(T, n)   # v = Σ * c_j

        # in constructor add:
        zeros(T, n),      # v
        zeros(T, n),      # δμ
    )
end


### Gibbs
function GibbsSamplerTVSARMA_full(y_g, Y, priorSettings, modelSettings, algoSettings)

    # ==================================================
    # Dimensions
    # ==================================================
    T = length(y_g)

    #intercept_dynamics = :ll   # or :rw
    #intercept_dynamics = INTERCEPT ? :ll : :none

    # ==================================================
    # Prior settings
    # ==================================================
    ϕ₀, κ₀, m₀, σ₀, ν₀, ψ₀, μ₀, Σ₀,
    σₑ, alpha_sigma, beta_sigma, alpha_sigma_hat,
    α_ukf, β_ukf, κ_ukf = priorSettings

    # ==================================================
    # Algorithm settings
    # ==================================================
    nBurn, nIter, INTERCEPT, resid_label,
    method_label, SARMA, SAR, SMA,
    SAR_conditional, SV, SVDSP, DSP_label = algoSettings 

    # ==================================================
    # Model settings
    # ==================================================
    nPerGroup, s1, p1, s2, p2, p_max, nLags, iterations,
    Cargs, Z, activeLags_ma, activeLags_ar, 
    cache_ma, cache_ar, ztrans,
    updateσₙ, nMixComp, α, β,
    ϕ̄₀, κ̄₀, m̄₀, σ̄₀, ν̄₀, ψ̄₀,
    ϕ̄v, μ̄v, σ̄²ₙ,
    intercept_dynamics, T_use = modelSettings

    # ==================================================
    # Thinning
    # ==================================================
    thin_factor = 1
    nPost = nIter - nBurn  # 13000 - 3000 = 10000
    nThin = div(nPost, thin_factor)  # 10000 / 10 = 700  
    thin_idx = 0

    # ==================================================
    # Intercept dynamics switch (NEW)
    # ==================================================
    # :rw -> old random-walk intercept
    # :ll -> new local linear trend (second difference)
    # set to :rw if you want the old behavior

    # Number of state components used by intercept
    nInterceptStates = INTERCEPT ? (intercept_dynamics === :ll ? 2 : 1) : 0

    # ==================================================
    # Derived dimensions
    # ==================================================
    T_all = T * nPerGroup

    # startcol is now automatic and consistent everywhere
    startcol = 1 + nInterceptStates
    stopcol  = nLags

    # ==================================================
    # Initialize latent processes
    # ==================================================
    #δ = fill(0.0, nLags)

    # ==================================================
    # Log-χ² mixture approximation
    # ==================================================
    ω, m, v   = SetUpLogChi2Mixture(nMixComp, 1)
    mixLogχ²₁ = prepare_logchi2_mix10(ω, m, v)

    #mixLogχ²₁, m, v = SetUpLogChi2Mixture(nMixComp, 1) # Only 5 and 10 component supported
    #S = zeros(Int, T, nLags)
    postDist  = zeros(T, nMixComp)
   
    # ==================================================
    # Initial latent state values
    # ==================================================
    σₑ = reshape(σₑ, :, 1)

    μ  = fill(m₀, nLags)
    H = fill(m₀, T, nLags)
    H_prev = fill(m₀, T, nLags)

    Hpost  = zeros(T, nLags,  nThin)
    ξ      = ones(T, nLags)
    ϕpost  = zeros(nLags,  nThin)
    μpost  = zeros(nLags,  nThin)
    intercept_true = zeros(nThin, T_all )
    
    ϕ = fill(ϕ₀, nLags)
    S  = zeros(Int, T, nLags)
    σ²ₙ = fill(updateσₙ ? 1.0 : 1.0, nLags) ## Measurement noise scaling
    
    scale_post  = zeros(nLags,  nThin)

    if INTERCEPT && intercept_dynamics === :ll
        # State ordering: [c_t, d_t, coefficients...]
        # No stochastic volatility on level c_t
        Hpost  = zeros(T, nLags-1,  nThin)
        ϕpost  = zeros(nLags-1,  nThin)
        μpost  = zeros(nLags-1,  nThin)

        μ  = fill(m₀, nLags-1)
        ϕ = fill(ϕ₀, nLags-1)
        H = fill(m₀, T, nLags-1)

        #if INTERCEPT
        #H[:,1] .= -10 + log(l)
        #end

        ξ  = ones(T, nLags-1)
        S  = zeros(Int, T, nLags-1)
        σ²ₙ = fill(updateσₙ ? 1.0 : 1.0, nLags-1)
    end

    H̃     = H .- m₀
    state = zeros(T + 1, nLags)

    # ==================================================
    # State transition matrix A (UPDATED)
    # ==================================================
    A = Matrix{Float64}(LinearAlgebra.I, nLags, nLags)

    if INTERCEPT
        if intercept_dynamics === :ll
            # local linear trend:
            # c_t = c_{t-1} + d_{t-1}
            # d_t = d_{t-1}
            A[1, 2] = 1.0
        end
        # :rw needs no change (identity already does it)
    end

    # ==================================================
    # Control input (unchanged)
    # ==================================================
    B = zeros(nLags)
    U = zeros(T, 1)

    # ==================================================
    # Storage for MCMC draws
    # ==================================================
    θpost  = zeros(T, nLags,  nThin)

    if SVDSP || SV
        σₑpost = zeros(T,  nThin)
    else
        σₑpost = zeros(nThin)
    end

    Σ_filt = zeros(T, nLags,  nThin)
    Σ_pred = zeros(T, nLags,  nThin)
    μ_filt = zeros(T, nLags,  nThin)
    μ_pred = zeros(T, nLags,  nThin)

    static_state_var = zeros(nLags,  nThin)

    # ==================================================
    # Remaining code: UNCHANGED
    # ==================================================

    total_params = size(Z, 2)
    Cargs_raw = [Vector{Float64}(undef, total_params) for _ in 1:T_all] 
    σₑ² = similar(σₑ)
    Dᵩ = BandedMatrix(-1 => repeat([-ϕ[1]], T-1),0 => Ones(T))
  
     # ==================================================
    # Observation equation setup
    # ==================================================
    
    if SMA
        σy = nothing
        σ0 = σₑ[1]
        ψ_mat = Matrix{Float64}(undef, total_params, T)
        ψ_expanded = Matrix{Float64}(undef, total_params, T_all)
        #residuals    = fill(0.0, T_all + p_max[2])
        residuals    = fill(0.0, T_all)
        errors    = fill(0.0, T_all + p_max[2])
        

        freeze_iter = 1000
        errors_med = zeros(T_all + p_max[2],  freeze_iter)
        errors_mx = zeros(T_all + p_max[2], 1, freeze_iter)
        Z_fixed = nothing
       
    elseif SARMA
         σ0 = σₑ[1]
        σy = σₑ[1]^2
        
        Z_ar = Z[:, 1:length(activeLags_ar)]
        Z_ma = Z[:, length(activeLags_ar)+1:total_params]
        σy = var(@view Y[1:min(end, 30)])
        
        errors    = fill(0.0, T_all + p_max[2])
        #errors_mx = zeros(T_all + p_max[2], 1,  nThin)
        #errors_med = zeros(T_all + p_max[2],  nIter)
        #residuals    = fill(0.0, T_all + p_max[2])
        y_mx      = zeros(p_max[1], 1,  nThin)

        ϕ_mat = Matrix{Float64}(undef, length(activeLags_ar), T)
        ϕ_expanded = Matrix{Float64}(undef, length(activeLags_ar), T_all)
        ψ_mat = Matrix{Float64}(undef, length(activeLags_ma), T)
        ψ_expanded = Matrix{Float64}(undef, length(activeLags_ma), T_all)

        nθ_ar = sum(p1) 
        nθ_ma = sum(p2)
        ar_cols = startcol : (startcol + nθ_ar - 1)
        ma_cols = (startcol + nθ_ar) : (startcol + nθ_ar + nθ_ma - 1)

        freeze_iter = 1000
        errors_med = zeros(T_all + p_max[2],  freeze_iter)
        errors_mx = zeros(T_all + p_max[2], 1, freeze_iter)
        Z_fixed = nothing


    elseif SAR
        
        σ0 = σₑ[1]
        #σy = σₑ[1]^2
  
        σy = Statistics.var(@view Y[1:min(end, 30)])
        y_mx      = zeros(p_max[1], 1,  nThin)
        residuals = fill(0.0, T_all)
        ϕ_mat = Matrix{Float64}(undef, total_params, T)
        ϕ_expanded = Matrix{Float64}(undef, total_params, T_all)  
    end

    # ==================================================
    # Stochastic volatility buffers
    # ==================================================
    if SVDSP || SV

        pre_length = SAR ? 0 : p_max[2]
        #||SARMA
        h̄ = fill(m̄₀, T_all + pre_length)
        h̃ = h̄ .- m̄₀

        hstar = zeros((T_all + 1) + pre_length); hstar .= m̄₀
        ξ̄ = ones(T_all + pre_length)

        postDistsv = zeros(T_all + pre_length, nMixComp)
        Ssv = zeros(Int, T_all + pre_length)

        #errors_mx = zeros(T_all + pre_length, 1, nIter - nBurn)
    
        # Holders for SV parameters
        μ̃post = zeros(T,  nThin)
        ϕ̃post = zeros(T,  nThin)
        σ̄²ₙpost = zeros(T,  nThin)
        #h̃post = zeros(T_all + pre_length,  nThin)
        h̃post = zeros(T_all,  nThin)

        # allocate once OUTSIDE Gibbs loop
        σₑ_full = Vector{Float64}(undef, T_all)

    end

    static_var = fill(0.0, nLags)
    
    if !DSP_label
        static_var .= exp.(H[1,:])
    end
    
    ### DSP buffers
    zprev_buf = zeros(T)
    zcurr_buf = zeros(T)
    prop_sd_phi = fill(0.05, nLags)
    acc_phi = zeros(Int, nLags)

    ### ARMA buffers
    if SAR

        p      = p_max[1]
        #maxlag = maximum(activeLags_ar)
        maxlag = p
        kl     = length(activeLags_ar)

        ws_presample = build_presample_workspace(maxlag, activeLags_ar)

        group_map = build_group_map(p, nPerGroup)

        int_exp = zeros(Float64, p)
        x0_buf  = zeros(Float64, p)
        m0_buf  = zeros(Float64, p)

        ws = IEKFWorkspace(nLags, nPerGroup, kl, pFit; T=Float64)

    elseif SMA

        q      = p_max[2]
        maxlag_ma = maximum(activeLags_ma)
        kl     = length(activeLags_ma)

        ws_sma = build_SMA_workspace(q)

        # IMPORTANT: SMA does NOT use these
        # (they belong to AR presample logic)
        int_exp = nothing
        x0_buf  = nothing
        m0_buf  = nothing

        ws = IEKFWorkspace(nLags, nPerGroup, kl, pFit; T=Float64)

    elseif SARMA
         # SARMA
        intercept_true = zeros(nThin, T )
        nma = length(ma_cols)

        if INTERCEPT
           nar_inter = length(ar_cols)+1
           nar = length(ar_cols)
        else
            nar = length(ar_cols)
            nar_inter = nar
        end

        # AR
        p = p_max[1]
        kl_ar = length(activeLags_ar)

        ws_AR_presample = build_presample_workspace(p, activeLags_ar)
        group_map_ar = build_group_map(p, l)

        int_exp = zeros(Float64, p)
        x0_buf  = zeros(Float64, p)
        m0_buf  = zeros(Float64, p)

        ws_ar = IEKFWorkspace(nar_inter, l, kl_ar, nar; T=Float64)

        # MA
        q = p_max[2]
        kl_ma = length(activeLags_ma)

        ws_sma = build_SMA_workspace(q)
        ws_ma = IEKFWorkspace(nma, l, kl_ma, nma; T=Float64)

        Z = similar(Z_ar, size(Z_ar,1), size(Z_ar,2)+size(Z_ma,2))
   
    end

    cond_mean_post = zeros(Float64, T_all)
    #residuals = zeros(Float64, T_all)

    group_map_T = build_group_map(T_all, nPerGroup)

    maxp = maximum(p1)

    φtmp = zeros(Float64, maxp)
    Ptmp = zeros(Float64, maxp)
    ϕtmp = zeros(Float64, maxp, maxp)
 
    ### Buffers for backwards sampling  #
    KG_buf   = zeros(nLags, nLags)
    tmp_vec  = zeros(nLags)
    tmp_mat  = zeros(nLags,nLags)
    tmp_mat2 = zeros(nLags, nLags)

    ######
    #LOOP
    ######


   for i in 1:nIter # i=1

    # ==================================================
    # Draw local level using FFBS
    # ==================================================
    Σₙ = DSP_label ? LogVol2Covs( H) : Vol2Covs(static_var)
    # seems pretyy fast and small allocation

    # ==================================================
    # Precompute σₑ² in-place (NO allocation)
    # ==================================================
    @inbounds @. σₑ² = σₑ .* σₑ
   
    # ==================================================
    # Update the state
    # ==================================================

    Σₑ_g  = group_vector(σₑ², nPerGroup) #500-element Vector{Vector{Float64}}
    #500-element Vector{Vector{Vector{Float64}}}:

    if SARMA

         state = FFBSx_sarma(
            U, y_g, A, B, Cargs, Σₑ_g, Σₙ, μ₀, Σ₀,
            iterations, α_ukf, β_ukf, κ_ukf, ws_ar, ws_ma;
            resid_check = resid_label,
            mode = method_label,
            startcol = startcol,
            INTERCEPT = INTERCEPT,
            ar_cols = ar_cols,
            ma_cols = ma_cols,
            nar_inter = nar_inter,
            cache_ar = cache_ar,
            cache_ma = cache_ma,
            ztrans = ztrans,
            clipped_partials = clipped_partials,
            p_threshold = p_threshold,
            intercept_dynamics = intercept_dynamics
        )
        
    else

        state = FFBSx(
            U, y_g, A, B,  Cargs, Σₑ_g, Σₙ, μ₀, Σ₀,
            iterations, α_ukf, β_ukf, κ_ukf, ws;
            resid_check = resid_label,
            mode = method_label,
            startcol = startcol,
            INTERCEPT=INTERCEPT,
            negative_signs = !SMA
        )

    end
        
    # ==================================================
    # Update log-volatility evolution
    # ==================================================
    
    if intercept_dynamics === :ll
        omega = diff(state[:, 2:end], dims = 1)
    else
        omega = diff(state, dims = 1)
          if scaled==true
            σ2_global = mean(getindex.(Σₑ_g, 1))
            #σ_global  = sqrt(σ2_global)
            Sdiag = fisher_scaling_diag_gaussian(Z[:,1:2]; σ2=σ2_global, intercept=true)
            omega./= Sdiag'
            #omega[:,1] ./= σ_global
            #omega[:,1] ./= sqrt.([Σₑ_g[t][1] for t in 1:length(Σₑ_g)])
            #omega ./= Sdiag'
          end
        
    end 

    if DSP_label

        #H_prev .= H   # save H before updating

        if nPerGroup==1
            update_dsp!(omega, S, H, H̃, ξ, ϕ, μ, σ²ₙ,ϕ₀, κ₀, m₀, σ₀, ν₀, ψ₀,mixLogχ²₁, m, v, postDist, Dᵩ;offset = eps(),updateσₙ = updateσₙ,α = α,β = β)
        else

        update_dsp_grouped!(
                omega, S, H, H̃, ξ, ϕ, μ, σ²ₙ,
                ϕ₀, κ₀, m₀, σ₀, ν₀, ψ₀,
                mixLogχ²₁, m, v, postDist, Dᵩ;
                g = nPerGroup,
                zprev_buf = zprev_buf,
                zcurr_buf = zcurr_buf,
                prop_sd_phi = prop_sd_phi,
                acc_phi = acc_phi,
                updateσₙ = updateσₙ,
                α = α,
                β = β,
                INTERCEPT=INTERCEPT
            )

        end
    else
      static_var .= compute_noise_SARMA_multi(alpha_sigma_hat, beta_sigma, omega)
    end

    ## Infer the errors given AR/MA part
    ## Infer the errors given AR/MA part
    if SARMA
    
        @inbounds Threads.@threads for t in 1:T
                row = view(state, t+1, ar_cols)
                out_ar = view(ϕ_mat, :, t)
                MultiSARMAtoReg_cached!(out_ar, row, cache_ar; ztrans=ztrans, negative_signs=true)
                #MultiSARMAtoReg!(out, row, p1, s1, activeLags_ar;ztrans=ztrans, negative_signs=true)
        end
        expand_grouped_states_fast!(ϕ_expanded, ϕ_mat, l, T_all)
    
        ## Compute expanded SMA coeffs. 
        @inbounds Threads.@threads for t in 1:T
                row = view(state, t+1, ma_cols)
                out_ma = view(ψ_mat, :, t)
                MultiSARMAtoReg_cached!(out_ma, row, cache_ma;ztrans=ztrans, negative_signs=false)
                #MultiSARMAtoReg!(out, row, p2, s2, activeLags_ma;ztrans=ztrans, negative_signs=false)
        end

        expand_grouped_states_fast!(ψ_expanded, ψ_mat, l, T_all)
        ϕ_sum = vec(sum(ϕ_expanded; dims=1))   # length T_all
    
        # ---------- AR stage ----------
        ma_sums = Vector{Float64}(undef, p_max[1])
        obs_ar  = Vector{Float64}(undef, p_max[1])

        @inbounds for t in 1:p_max[1]
            ma_sums[t] = dot(@view(Z_ma[t, :]), @view(ψ_expanded[:, t]))
            obs_ar[t]  = Y[t] - ma_sums[t]
        end

        init_y, Z_ar = build_AR_init_opt_orig!(
                x0_buf,
                int_exp,
                m0_buf,
                group_map_ar,
                ws_AR_presample,
                obs_ar,
                state,
                ϕ_expanded,
                activeLags_ar,
                p1, s1,
                p_max,
                σₑ²,
                σy;
                INTERCEPT = INTERCEPT,
                cond_sma  = ma_sums[1],
                Z = Z_ar,
                l = nPerGroup
        )

        # ---------- MA stage ----------
        obs_ma = copy(Y)
        @inbounds for t in 1:T_all# t=1
            obs_ma[t] -= dot(@view(Z_ar[t, :]), @view(ϕ_expanded[:, t]))
        end

        # ---------------------------------------------------------
        # Learn MA error regressors, then freeze them
        # ---------------------------------------------------------
        if i <= freeze_iter

            errors_reg = build_MA_errors_banded(
                errors,
                Y,
                state,
                ψ_expanded,
                activeLags_ma,
                p2,
                s2,
                p_max,
                σₑ²,
                σ0,
                T_use,
                nPerGroup;
                ws_sma = ws_sma,
                INTERCEPT = INTERCEPT,
                ztrans = ztrans,
                presample_mode = :posterior,
                use_σ0_for_presample = true
            )

            # Store current reconstructed error path
            errors_med[:, i] .= errors_reg

            # Running pointwise median error path
            med_error = dropdims(
                median(@view(errors_med[:, 1:i]), dims=2);
                dims=2
            )

            # MA regressors from median error history
            _, Z_MA, _ = SetupARReg_active(
                med_error,
                activeLags_ma
            )

            # Check AR/MA time alignment
            @assert size(Z_MA, 1) == size(Z_ar, 1)

            # Construct complete SARMA design
            @views Z[:, 1:size(Z_ar, 2)] .= Z_ar
            @views Z[:, size(Z_ar, 2)+1:end] .= Z_MA

            # Freeze only the MA part
            if i == freeze_iter
                Z_fixed = copy(Z_MA)
            end

            # Current error realization is still the response
            residuals .= errors_reg[maxlag_ma+1:end]

        else

            # -----------------------------------------------------
            # Fixed-MA-design phase
            # -----------------------------------------------------

            @assert size(Z_fixed, 1) == size(Z_ar, 1)

            @views Z[:, 1:size(Z_ar, 2)] .= Z_ar
            @views Z[:, size(Z_ar, 2)+1:end] .= Z_fixed

            compute_conditional_mean!(
                cond_mean,
                Z,
                vcat(ϕ_expanded, ψ_expanded),
                state,
                group_map_T;
                INTERCEPT = INTERCEPT
            )

            compute_residuals!(
                residuals,
                Y,
                cond_mean
            )
        end

    elseif SMA

        #adapt = i ≤ nBurn
        @inbounds Threads.@threads for t in 1:T
                row = view(state, t+1, startcol:stopcol)
                out = view(ψ_mat, :, t)
                MultiSARMAtoReg_cached!(out, row, cache_ma; ztrans=ztrans, negative_signs=false)
                #MultiSARMAtoReg!(out, row, p2, s2, activeLags_ma;ztrans=ztrans, negative_signs=false)
        end
        expand_grouped_states_fast!(ψ_expanded, ψ_mat, nPerGroup, T_all)
   
        ### Median trick
        
        # ---------------------------------------------------------
        # Learn MA error regressors, then freeze them i=1
        # ---------------------------------------------------------
        if i <= freeze_iter

            errors_reg = build_MA_errors_banded(
                errors,
                Y,
                state,
                ψ_expanded,
                activeLags_ma,
                p2,
                s2,
                p_max,
                σₑ²,
                σ0,
                T_use,
                nPerGroup;
                ws_sma = ws_sma,
                INTERCEPT = INTERCEPT,
                ztrans = ztrans,
                presample_mode = :posterior,
                use_σ0_for_presample = true
            )

            # Store current reconstructed MA-error path
            errors_med[:, i] .= errors_reg

            # Running pointwise median
            med_error = dropdims(
                median(@view(errors_med[:, 1:i]), dims=2);
                dims=2
            )

            # MA lagged-error regressors based on running median
            _, Z, _ = SetupARReg_active(
                med_error,
                activeLags_ma
            )

            # Freeze the design after adaptation
            if i == freeze_iter
                Z_fixed = copy(Z)
            end

            # Current reconstructed error path
            residuals .= errors_reg[maxlag_ma+1:end]

        else

            # -----------------------------------------------------
            # Fixed-Z phase
            # -----------------------------------------------------
            Z = Z_fixed

            compute_conditional_mean!(
                cond_mean,
                Z,
                ψ_expanded,
                state,
                group_map_T;
                INTERCEPT = INTERCEPT
            )

            compute_residuals!(
                residuals,
                Y,
                cond_mean
            )
        end

        #_, Z, _     = SetupARReg_active(errors_reg, activeLags_ma)
        #residuals = errors_reg
       
    elseif SAR
        
        # fast
        @views @inbounds for t in 1:T
            row = state[t+1, startcol:stopcol]
            out = ϕ_mat[:, t]
            MultiSARMAtoReg_cached!(out, row, cache_ar; ztrans=ztrans, negative_signs=true)
        end
            expand_grouped_states_fast!(ϕ_expanded, ϕ_mat, nPerGroup, T_all)
            ϕ_sum = vec(sum(ϕ_expanded; dims=1))   # length T_all

       if !SAR_conditional         
              
       init_y, Z = build_AR_init_opt_orig!(
                    x0_buf,
                    int_exp,
                    m0_buf,
                    group_map,
                    ws_presample,
                    Y,
                    state,
                    ϕ_expanded,
                    activeLags_ar,
                    p1, s1,
                    p_max,
                    σₑ²,
                    σy;
                    INTERCEPT = INTERCEPT,
                    cond_sma  = nothing,
                    Z = Z,
                    l = nPerGroup,
                    presample_method = :posterior,
                    #presample_method = :recursive,
                    #rng=rng
                    )
        
            end

       #cond_mean = compute_conditional_mean(Z, ϕ_expanded, state; INTERCEPT = INTERCEPT)
       compute_conditional_mean!(
            cond_mean,
            Z,
            ϕ_expanded,
            state,
            group_map_T;
            INTERCEPT = INTERCEPT
        )
        compute_residuals!(residuals, Y, cond_mean)
    end

    # ==================================================
    # Recompute Cargs (FAST + THREADED)
    # ==================================================
    if SAR

        if !SAR_conditional
            @inbounds copyto!(Cargs[1][1], view(Z, 1, :))
        end
         
    else
 
        build_Cargs_fast_threaded!(Cargs_raw, Z, T_all)
        Cargs = group_vector(Cargs_raw, nPerGroup)    
 
    end

    # ==================================================
    # Update observation noise / SV
    # ==================================================
    if SV
        if nPerGroup > 1

             h̄, ϕ̄v, μ̄v, σ̄²ₙ = UpdateErrorVolatility_grouped!(
               residuals,           # ORIGINAL scale residuals (length T_all)
                h̄, ξ̄, ϕ̄v, μ̄v, σ̄²ₙ,
                ϕ̄₀, κ̄₀, m̄₀, σ̄₀, ν̄₀, ψ̄₀,
                mixLogχ²₁, m, v;
                g=nPerGroup,
               offsetSV = eps())

            σₑ_g = exp.(h̄ ./ 2)
            expand_sigma_grouped!(σₑ, σₑ_g, l, T_all)

        else

            h̄, ϕ̄v, μ̄v, σ̄²ₙ =
                UpdateErrorVolatility(
                residuals, h̄, ξ̄, ϕ̄v, μ̄v, σ̄²ₙ,
                ϕ̄₀, κ̄₀, log(σ0^2), σ̄₀, ν̄₀, ψ̄₀, ### OBS chnaged mo to log()
                mixLogχ²₁, m, v;
                offsetSV = eps()
             )

            #σₑ = vec(reshape(exp.(h̄ ./ 2), :, 1)[pre_length+1:end, :])
            σₑ = reshape(exp.(h̄ ./ 2), :, 1)
        end

    elseif SVDSP

        hstar, h̄, h̃, Ssv, ξ̄, ϕ̄v, μ̄v =
            UpdateErrorVolatility_DSP(
                residuals, h̄, hstar, h̃, ξ̄, ϕ̄v, μ̄v,
                ϕ̄₀, κ̄₀, m̄₀, σ̄₀, ν̄₀, ψ̄₀,
                mixLogχ²₁, postDistsv, m, v, Ssv, σ0^2;
                σ̄²ₙ = 1.0,
                offsetSV = eps()
                #offsetSV = 10^-4
            )

        if d_order ==1
            σₑ = vec(reshape(exp.(hstar[2:end] ./ 2), :, 1))
        else
            σₑ = vec(reshape(exp.(hstar[3:end] ./ 2), :, 1))
        end
    else
 
        σₑ = compute_noise_SARMA(alpha_sigma_hat, beta_sigma, residuals)
        σₑ = fill(σₑ, T_all, 1)

    end

    # ==================================================
    # Store draws
    # ==================================================

    # ==================================================
    # Store MA errors during adaptation
    # ==================================================
    if SMA && i <= freeze_iter
        errors_mx[:, :, i] .= errors_reg
    end

     if  i > nBurn && ((i - nBurn) % thin_factor == 0)
        
        thin_idx += 1
        #idx = i - nBurn

        θpost[:, :,  thin_idx] .= state[2:end, :]
        Hpost[:, :,  thin_idx] .= H
       
        if SVDSP || SV
        
            if nPerGroup > 1
                σₑpost[:,  thin_idx]   .= σₑ_g
            else
                σₑpost[:,  thin_idx]   .= σₑ
            end
    
        else
            σₑpost[thin_idx]   = σₑ[1]
        end

        ϕpost[:, thin_idx]   .= ϕ
        μpost[:, thin_idx]   .= μ 
        #scale_post[:, thin_idx] .= σ²ₙ
        #cond_mean_post[:, thin_idx] .= cond_mean

        if !DSP_label
            static_state_var[:, thin_idx] .= static_var
        end

        if SV
            μ̃post[:, thin_idx] .= μ̄v
            ϕ̃post[:, thin_idx] .= ϕ̄v
            σ̄²ₙpost[:, thin_idx] .= σ̄²ₙ    
        elseif SVDSP
            h̃post[:, thin_idx] .= h̄
            μ̃post[:, thin_idx] .= μ̄v
            ϕ̃post[:, thin_idx] .= ϕ̄v
            #σ̄²ₙpost[:,idx] .= 1 ./ξ̄
        end

        if SARMA
            errors_mx[:, :,  thin_idx] .= residuals
            y_mx[:, :,  thin_idx]      .= init_y
        elseif SAR
            if !SAR_conditional
            y_mx[:, :,  thin_idx]      .= init_y
            end
            #intercept_true[thin_idx,: ] = x_mean .*(1 .-ϕ_sum) + state[2:end, 1]
            #intercept_true[thin_idx,: ] = x_mean .*(1 .-ϕ_sum) + sd .*state[2:end, 1]

        #elseif SMA
          # errors_mx[:, :, thin_idx] = errors_reg
        end
    end
end # end Gibbs

if SARMA
    return θpost, Hpost, σₑpost, errors_mx, y_mx

elseif SMA
    return θpost, Hpost, σₑpost, ϕpost, μpost, errors_mx

elseif SAR
    if !SAR_conditional
        if SV || SVDSP
            return θpost, Hpost, σₑpost, ϕpost, μpost, μ̃post, ϕ̃post, h̃post, σ̄²ₙpost, y_mx, static_state_var, intercept_true
        else
            return θpost, Hpost, σₑpost, ϕpost, μpost, y_mx, static_state_var, cond_mean_post, intercept_true
        end
    else
        if SV || SVDSP
            return θpost, Hpost, σₑpost, ϕpost, μpost, μ̃post, ϕ̃post, h̃post, σ̄²ₙpost, static_state_var, intercept_true
        else
            return θpost, Hpost, σₑpost, ϕpost, μpost
        end

    end
else
    error("No valid model type selected: SARMA=$SARMA, SMA=$SMA, SAR=$SAR")
end
end


### FFBS
function FFBSx(
    U, y_g, A, B,  Cargs, Σₑ, Σₙ, μ₀, Σ₀, max_iterations,
    α_ukf, β_ukf, κ_ukf, ws;
    resid_check::Bool = true,
    mode::Symbol=:iekf,
    startcol =    startcol,
    INTERCEPT::Bool=true,
    negative_signs::Bool = true
)

   # unpack what you use inside
    #(; cache_ar, l, intercept_dynamics,  ztrans) = modelSettings
    #Σₑ=σₑ .^2
    #Σₑ=Σₑ_g 
    
    T = size(y_g, 1)
    n = length(μ₀)
    ny = length(y_g[1])
    q = size(U, 2)

    staticA      = ndims(A) != 3
    staticΣₙ     = !(ndims(Σₙ) == 3 || eltype(Σₙ) <: PDMat)
    staticCargs  = !(Cargs isa Vector)

    # UT weights (unchanged)
    use_sigma = mode in (:iukf, :iukfl, :iplf, :diplf)
    if use_sigma
        λ  = α_ukf^2 * (n + κ_ukf) - n
        ωₘ = [λ/(n + λ); ones(2*n)/(2*(n + λ))]
        ωₛ = [λ/(n + λ) + (1 - α_ukf^2 + β_ukf); ωₘ[2:end]]
        γ  = sqrt(n + λ)
    else
        ωₘ = ωₛ = nothing
        γ  = 0.0
    end

    μ = copy(μ₀)
    Σ = Matrix(Σ₀)

    μ̄_buf = similar(μ)
    Ω̄_buf = similar(Σ)
    tmp_n_n = similar(Σ)

    μ_upd = similar(μ)
    Σ_upd = similar(Σ)

    nreg = length(cache_ar.activeLags)
    #ws = IEKFWorkspace(n, ny, nreg; T=Float64)
    #ws = IEKFWorkspace(n, ny, nreg, pFit; T=Float64)

    #Σₑ_g  = group_vector(Σₑ, l)
    Σₙ_buf = similar(Σ)

    μ_filter = zeros(T, n)
    Σ_filter = zeros(n, n, T)
    μ_pred   = zeros(T, n)
    Σ_pred   = zeros(n, n, T)

    #Σₑt = Diagonal(zeros(ny))
    At   = A

    KG_buf   = zeros(n, n)
    tmp_vec  = zeros(n)
    tmp_mat  = zeros(n,n)
    tmp_mat2 = zeros(n, n)

    #  0.025405 seconds (1.86 k allocations: 1.209 MiB)
     @views @inbounds for t in 1:T #t=1

        #At      = staticA ? A : view(A, :, :, t)
        Cargs_t = staticCargs ? Cargs : Cargs[t]

        #Σₑt = Diagonal(Σₑ_g[t])
        #copyto!(Σₑt.diag, Σₑ_g[t])

        if intercept_dynamics === :ll && INTERCEPT
            Σₙ_base = staticΣₙ ? Σₙ : Σₙ[t]
            fill!(Σₙ_buf, 0.0)
            Σₙ_buf[2:end, 2:end] .= Σₙ_base
            Σₙt = Σₙ_buf
        else
            Σₙt = staticΣₙ ? Σₙ : Σₙ[t]
        end

        u = (q == 1) ? U[t] : view(U, t, :)
        y = y_g[t]

        # predict
        mul!(μ̄_buf, At, μ)
        μ̄_buf .+= B * u

        mul!(tmp_n_n, At, Σ)
        mul!(Ω̄_buf, tmp_n_n, At')
        #Ω̄_buf .+= Σₙt

        if scaled==true
            #σ2_global  = mean(getindex.(Σₑ, 1))
            Σₙt_scaled  = copy(Σₙt)
            #Σₙt_scaled[1,1] *= σ2_global
            Σₙt_scaled[diagind(Σₙt_scaled)] .*= Sdiag
            Ω̄_buf .+= Σₙt_scaled
            
            #length(Σₙt_scaled[diagind(Σₙt_scaled)])
            #length(Sdiag)
            #Σₙt_scaled *= S.^2
            #Σₙt_scaled = copy(Σₙt)
            #Σₙt_scaled[1,1] *= Σₑ[t][1]
            #Ω̄_buf .+= Σₙt_scaled
        else
            Ω̄_buf .+= Σₙt
        end
     
        # update (RESTORE THIS)

        if negative_signs
            cache_type = cache_ar
        else
            cache_type = cache_ma
        end

        kalmanfilter_update_IEKF_seq!(
            μ_upd, Σ_upd,
            μ̄_buf, Ω̄_buf,
            u, y, At, B,
            Cargs_t,
            Σₑ[t],
            #Σₑt, 
            γ, ωₘ, ωₛ,
            max_iterations,
            resid_check,
            Val(mode),
            ws;
            cache    = cache_type,
            ztrans   = ztrans,
            clipped_partials = clipped_partials,
            INTERCEPT   = INTERCEPT,
            p_threshold = p_threshold,
            startcol    = startcol,
            negative_signs = negative_signs
        )

        # swap
        μ, μ_upd = μ_upd, μ
        Σ, Σ_upd = Σ_upd, Σ

         #μ =  μ_filter[5, :]
         #Σ =   Σ_filter[:, :, 5] 
        
        μ_filter[t, :]    .= μ
        Σ_filter[:, :, t] .= Σ
        μ_pred[t, :]      .= μ̄_buf
        Σ_pred[:, :, t]   .= Ω̄_buf
    end

    # ---- backward sampling ----
    #KG_buf   = zeros(n, n)
    #tmp_vec  = zeros(n)
    #tmp_mat  = zeros(n, n)
    #tmp_mat2 = zeros(n, n)

    #X = zeros(T, n)
    #copyto!(view(X, T, :),rand(MvNormal(view(μ_filter, T, :),Hermitian(view(Σ_filter, :, :, T)))))

    X = zeros(T, n)

    @views begin
        
        copyto!(tmp_mat, Σ_filter[:, :, T])

        @inbounds for i in 1:n
            tmp_mat[i, i] += 1e-12
        end

        Htmp = Hermitian(tmp_mat)   # create once OUTSIDE
        FT = cholesky!(Htmp)

        randn!(tmp_vec)
        mul!(tmp_vec, FT.L, tmp_vec)

        copyto!(view(X, T, :), view(μ_filter, T, :))
        
        @inbounds for i in 1:n
            X[T, i] += tmp_vec[i]
        end
    end

    #At_next = ndims(A) == 3 ? @view(A[:, :, t+1]) : A
   @inbounds for t = (T-1):-1:1 #t=1

        xt      = @view X[t, :]
        xnext   = @view X[t+1, :]
        μfilt   = @view μ_filter[t, :]
        Σfilt   = @view Σ_filter[:, :, t]
        μpred   = @view μ_pred[t+1, :]
        Σpred   = @view Σ_pred[:, :, t+1]

        BackwardSim_ref!(
            xt,
            xnext,
            μfilt,
            Σfilt,
            μpred,
            Σpred,
            A,
            KG_buf, tmp_vec, tmp_mat, tmp_mat2
        )
    end
   
    x0 = zeros(n)
    BackwardSim_ref!(
        x0,
        view(X, 1, :),
        μ₀,
        Σ₀,
        view(μ_pred, 1, :),
        view(Σ_pred, :, :, 1),
        A, 
        KG_buf, tmp_vec, tmp_mat, tmp_mat2
    )

    return [x0'; X]
end

###


function kalmanfilter_update_IEKF_seq!(
    μ::V,
    Σ::M,
    μ̄::V,
    Ω̄::M,
    u,
    y::V,
    A,
    B,
    Cargs,
    Σₑ::AbstractVector,
    γ::Real,
    ωₘ::Union{Nothing,AbstractVector},
    ωₛ::Union{Nothing,AbstractVector},
    max_iterations::Int,
    resid_check::Bool,
    ::Val{MODE},
    ws::IEKFWorkspace;
    cache::SARMARegCache,
    #jcache, 
    ztrans::AbstractString="partials",
    clipped_partials::Bool=false,
    INTERCEPT::Bool=false,
    p_threshold::Float64=Inf,
    startcol::Int,
    negative_signs::Bool = true
) where {
    MODE,
    V<:StridedVector,
    M<:StridedMatrix
}

    n  = length(μ̄)
    ny = length(y)

    # workspace views
    C̄        = ws.C̄
    resid     = ws.resid
    K         = ws.K
    tmp_ny_n  = ws.tmp_ny_n
    tmp_n_ny  = ws.tmp_n_ny
    tmp_n_n   = ws.tmp_n_n
    tmp_n_n2  = ws.tmp_n_n2
    S         = ws.S
    Ubuf      = ws.U
    I_n       = ws.I_n
    reg_terms = ws.reg_terms
    Jg        = ws.Jg

  
    @views begin

        # --------------------------------------------------
        # compute regression terms ψ(θ)
        # --------------------------------------------------

       θ₀ = INTERCEPT ? μ̄[1] : zero(eltype(μ̄))
       θ_work = INTERCEPT ? μ̄[startcol:end] : μ̄

       MultiSARMAtoReg_cached!(
            reg_terms,
            θ_work,
            cache;
            ztrans=ztrans,
            negative_signs=negative_signs
        )

        # --------------------------------------------------
        # Jacobian
        # --------------------------------------------------

        jacobian_C_fast!(
            C̄,
            reg_terms,
            Jg,
            μ̄,
            Cargs,
            cache,
            INTERCEPT=INTERCEPT,
            ztrans=ztrans,
            startcol=startcol,
            negative_signs=negative_signs
        )


        # --------------------------------------------------
        # residuals: y − h(μ̄)
        # --------------------------------------------------

        compute_resid!(resid, y, reg_terms, Cargs, θ₀)

    # ==================================================
    # CASE 1: sequential EKF for grouped observations
    # ==================================================
    if ny > 1

        ekf_update_sequential_diagR2!(
            μ,
            Σ,
            μ̄,
            Ω̄,
            C̄,
            resid,
            Σₑ,
            ws;
            joseph = false
        )


    clamp_partials!(μ, startcol, p_threshold)

    return nothing
end

# ==================================================
# CASE 2: scalar fallback (ny == 1)
# ==================================================

tmp = similar(μ)
mul!(tmp, Ω̄, view(C̄,1,:))
s = dot(view(C̄,1,:), tmp) + Σₑ[1]

# K = Ω̄ * C̄' / s
@inbounds for i in 1:n
    K[i,1] = zero(eltype(K))
    for k in 1:n
        K[i,1] += Ω̄[i,k] * C̄[1,k]
    end
    K[i,1] /= s
end

# μ = μ̄ + K * resid
@inbounds for i in 1:n
    μ[i] = μ̄[i] + K[i,1] * resid[1]
end

clamp_partials!(μ, startcol, p_threshold)

# Joseph covariance update
@inbounds for i in 1:n, j in 1:n
    tmp_n_n[i,j] = I_n[i,j] - K[i,1] * C̄[1,j]
end

mul!(tmp_n_n2, tmp_n_n, Ω̄)
mul!(Σ, tmp_n_n2, tmp_n_n')

@inbounds for i in 1:n, j in 1:n
    Σ[i,j] += K[i,1] * Σₑ[1] * K[j,1]
end

@inbounds for j in 1:n, i in j+1:n
    Σ[i,j] = Σ[j,i]
end

return nothing
    
end
end



"""
Sequential EKF update for diagonal measurement noise R = Diagonal(r).

This is *exactly equivalent* (up to floating point order) to the batch EKF update
when:
- you use the same fixed Jacobian C̄ computed at μ̄
- measurement noises are independent (R diagonal)
- you update using the same linearized model

Arguments:
- μ, Σ     : updated mean/cov (written in-place)
- μ̄, Ω̄    : predicted mean/cov
- C̄        : ny × n Jacobian at μ̄
- resid     : ny residuals y - h(μ̄) (computed with same μ̄)
- Rdiag     : vector length ny with diagonal entries of R
- ws        : workspace with buffers (see below)
"""

function ekf_update_sequential_diagR2!(
    μ,
    Σ,
    μ̄,
    Ω̄,
    C̄,
    resid,     # resid[j] = y[j] - yhat[j] evaluated at μ̄ (fixed!)
    Rdiag,
    ws;
    joseph::Bool = true
)

    n  = length(μ̄)
    ny = length(resid)

    v        = ws.v          # length n
    δμ       = ws.δμ         # length n  (NEW buffer in workspace)
    tmp_n_n  = ws.tmp_n_n
    tmp_n_n2 = ws.tmp_n_n2
    I_n      = ws.I_n

    copyto!(μ, μ̄)
    copyto!(Σ, Ω̄)
    fill!(δμ, zero(eltype(μ)))


    #@show size(ws.C̄)
    #@show length(ws.reg_terms)
    #@show size(ws.Jg)
    #@show length(Cargs[1])
    #@show length(activeLags_ma)

    @inbounds for j in 1:ny #j=1

        rj = Rdiag[j]

        # --------------------------------------------------
        # innovation: ν_j = resid[j] - c_j'*(μ-μ̄)
        # --------------------------------------------------
        cjδ = zero(eltype(μ))
        @simd for k in 1:n
            cjδ += C̄[j,k] * δμ[k]
        end
        
        ν = resid[j] - cjδ

        # --------------------------------------------------
        # v = Σ * c_j
        # --------------------------------------------------
        for i in 1:n
            s = zero(eltype(Σ))
            @simd for k in 1:n
                s += Σ[i,k] * C̄[j,k]
            end
            v[i] = s
        end

        # s = rj + c_j' v
        s = rj
        @simd for k in 1:n
            s += C̄[j,k] * v[k]
        end
        invs = one(s) / s

        # K = v / s  (implicitly)
        α = ν * invs

        # mean update: μ += K*ν, and keep δμ updated too
        @simd for i in 1:n
            Δ = v[i] * α
            μ[i]  += Δ
            δμ[i] += Δ
        end

        # covariance update
        if joseph
            # tmp = I − K c_j'
            for i in 1:n
                ki = v[i] * invs
                @simd for k in 1:n
                    tmp_n_n[i,k] = I_n[i,k] - ki * C̄[j,k]
                end
            end

            mul!(tmp_n_n2, tmp_n_n, Σ)
            mul!(Σ, tmp_n_n2, tmp_n_n')

            # + K r K'  with K = v/s
            β = rj * invs * invs
            for a in 1:n
                va = v[a]
                @simd for b in a:n
                    Σab = Σ[a,b] + β * va * v[b]
                    Σ[a,b] = Σab
                    Σ[b,a] = Σab
                end
            end
        else
            # Σ -= (v v') / s
            β = invs
            for a in 1:n
                va = v[a]
                @simd for b in a:n
                    Σab = Σ[a,b] - β * va * v[b]
                    Σ[a,b] = Σab
                    Σ[b,a] = Σab
                end
            end
        end
    end

    return nothing
end


function BackwardSim_ref!(
    out_x,
    x_next,
    μ_filt, Σ_filt,
    μ_pred_next,
    Σ_pred_next,
    A,                     # non-identity transition
    J_buf, tmp_vec, tmp_mat, tmp_mat2
)
    # --------------------------------------------------
    # Cholesky of predicted covariance Σ_pred_next
    # --------------------------------------------------
    Σp = Matrix(Σ_pred_next)
    @inbounds for i in 1:size(Σp,1)
        Σp[i,i] += 1e-12
    end
    
    Fp = cholesky!(Hermitian(Σp))

    # --------------------------------------------------
    # RTS / simulation smoother gain:
    # J = Σ_filt * A' * inv(Σ_pred_next)
    # --------------------------------------------------
    mul!(tmp_mat2, Σ_filt, A')      # tmp_mat2 = Σ_filt * A'
    ldiv!(tmp_mat, Fp, tmp_mat2')   # tmp_mat  = Σ_pred_next \ (Σ_filt*A')'
    J_buf .= tmp_mat'               # J = Σ_filt * A' / Σ_pred_next

    # --------------------------------------------------
    # Backward mean:
    # μ_back = μ_filt + J*(x_next - μ_pred_next)
    # --------------------------------------------------
    @. tmp_vec = x_next - μ_pred_next
    mul!(out_x, J_buf, tmp_vec)
    @. out_x += μ_filt

    # --------------------------------------------------
    # Backward covariance:
    # Σ_back = Σ_filt - J*Σ_pred_next*J'
    # --------------------------------------------------
    mul!(tmp_mat2, J_buf, Σ_pred_next)
    mul!(tmp_mat, tmp_mat2, J_buf')
    @. tmp_mat = Σ_filt - tmp_mat

    #@inbounds for i in 1:size(tmp_mat,1)
        #tmp_mat[i,i] += 1e-12
    #end

    # --------------------------------------------------
    # Draw sample
    # --------------------------------------------------
    # V1
    #Fback = cholesky!(Hermitian(tmp_mat))
    #randn!(tmp_vec)
    #mul!(tmp_vec, Fback.L, tmp_vec)
    #@. out_x += tmp_vec
    #V2
    try
        Fback = cholesky!(Hermitian(tmp_mat))
        randn!(tmp_vec)
        mul!(tmp_vec, Fback.L, tmp_vec)
        @. out_x += tmp_vec
    catch err
        if err isa LinearAlgebra.PosDefException
        copyto!(out_x, x_next)
    else
        rethrow()
    end
end

    return out_x
end



function FFBSx_sarma(
    U, y_g, A, B, Cargs, Σₑ, Σₙ, μ₀, Σ₀, max_iterations,
    α_ukf, β_ukf, κ_ukf, ws_ar, ws_ma;
    resid_check::Bool = true,
    mode::Symbol = :iekf,
    startcol::Int,
    INTERCEPT::Bool = true,
    ar_cols::UnitRange{Int},
    ma_cols::UnitRange{Int},
    nar_inter::Int,
    cache_ar::SARMARegCache,
    cache_ma::SARMARegCache,
    ztrans::AbstractString,
    clipped_partials::Bool,
    p_threshold::Float64,
    intercept_dynamics::Union{Symbol,Nothing}
)
   # unpack what you use inside
    #(; cache_ar, l, intercept_dynamics,  ztrans) = modelSettings
    #Σₑ=σₑ .^2
    #Σₑ=Σₑ_g 
    
    T = size(y_g, 1)
    n = length(μ₀)
    ny = length(y_g[1])
    q = size(U, 2)

    #@show typeof(Cargs)
    #@show length(Cargs)
    #@show typeof(Cargs[1])

    staticA      = ndims(A) != 3
    staticΣₙ     = !(ndims(Σₙ) == 3 || eltype(Σₙ) <: PDMat)
    staticCargs  = !(Cargs isa Vector)

    # UT weights (unchanged)
    use_sigma = mode in (:iukf, :iukfl, :iplf, :diplf)
    if use_sigma
        λ  = α_ukf^2 * (n + κ_ukf) - n
        ωₘ = [λ/(n + λ); ones(2*n)/(2*(n + λ))]
        ωₛ = [λ/(n + λ) + (1 - α_ukf^2 + β_ukf); ωₘ[2:end]]
        γ  = sqrt(n + λ)
    else
        ωₘ = ωₛ = nothing
        γ  = 0.0
    end

    μ = copy(μ₀)
    Σ = Matrix(Σ₀)

    μ̄_buf = similar(μ)
    Ω̄_buf = similar(Σ)
    tmp_n_n = similar(Σ)

    μ_upd = similar(μ)
    Σ_upd = similar(Σ)

    #Σₑ_g  = group_vector(Σₑ, l)
    Σₙ_buf = similar(Σ)

    μ_filter = zeros(T, n)
    Σ_filter = zeros(n, n, T)
    μ_pred   = zeros(T, n)
    Σ_pred   = zeros(n, n, T)

    #Σₑt = Diagonal(zeros(ny))
    At   = A

    KG_buf   = zeros(n, n)
    tmp_vec  = zeros(n)
    tmp_mat  = zeros(n,n)
    tmp_mat2 = zeros(n, n)

    #  0.025405 seconds (1.86 k allocations: 1.209 MiB)
     @views @inbounds for t in 1:T #t=1

        #At      = staticA ? A : view(A, :, :, t)
        Cargs_t = staticCargs ? Cargs : Cargs[t]

        #Σₑt = Diagonal(Σₑ_g[t])
        #copyto!(Σₑt.diag, Σₑ_g[t])

        if intercept_dynamics === :ll && INTERCEPT
            Σₙ_base = staticΣₙ ? Σₙ : Σₙ[t]
            fill!(Σₙ_buf, 0.0)
            Σₙ_buf[2:end, 2:end] .= Σₙ_base
            Σₙt = Σₙ_buf
        else
            Σₙt = staticΣₙ ? Σₙ : Σₙ[t]
        end

        u = (q == 1) ? U[t] : view(U, t, :)
        y = y_g[t]

        # predict
        mul!(μ̄_buf, At, μ)
        μ̄_buf .+= B * u

        mul!(tmp_n_n, At, Σ)
        mul!(Ω̄_buf, tmp_n_n, At')
        Ω̄_buf .+= Σₙt

        # update (RESTORE THIS)

        kalmanfilter_update_sarma!(
            μ_upd, Σ_upd,
            μ̄_buf, Ω̄_buf,
            u, y, At, B,
            Cargs_t,
            Σₑ[t],
            #Σₑt, 
            γ, ωₘ, ωₛ,
            max_iterations,
            resid_check,
            Val(mode),
            ws_ar, ws_ma;
            cache_ar    = cache_ar,
            cache_ma    = cache_ma,
            ztrans   = ztrans,
            clipped_partials = clipped_partials,
            INTERCEPT   = INTERCEPT,
            p_threshold = p_threshold,
            startcol    = startcol,
            ar_cols =  ar_cols,
            ma_cols =  ma_cols,
            nar_inter = nar_inter,
            kl_ar = kl_ar,
            kl_ma = kl_ma
        )

        # swap
        μ, μ_upd = μ_upd, μ
        Σ, Σ_upd = Σ_upd, Σ

         #μ =  μ_filter[5, :]
         #Σ =   Σ_filter[:, :, 5] 
        
        μ_filter[t, :]    .= μ
        Σ_filter[:, :, t] .= Σ
        μ_pred[t, :]      .= μ̄_buf
        Σ_pred[:, :, t]   .= Ω̄_buf
    end

    # ---- backward sampling ----
    #KG_buf   = zeros(n, n)
    #tmp_vec  = zeros(n)
    #tmp_mat  = zeros(n, n)
    #tmp_mat2 = zeros(n, n)

    #X = zeros(T, n)
    #copyto!(view(X, T, :),rand(MvNormal(view(μ_filter, T, :),Hermitian(view(Σ_filter, :, :, T)))))

    X = zeros(T, n)

    @views begin
        
        copyto!(tmp_mat, Σ_filter[:, :, T])

        @inbounds for i in 1:n
            tmp_mat[i, i] += 1e-12
        end

        Htmp = Hermitian(tmp_mat)   # create once OUTSIDE
        FT = cholesky!(Htmp)

        randn!(tmp_vec)
        mul!(tmp_vec, FT.L, tmp_vec)

        copyto!(view(X, T, :), view(μ_filter, T, :))
        
        @inbounds for i in 1:n
            X[T, i] += tmp_vec[i]
        end
    end

    #At_next = ndims(A) == 3 ? @view(A[:, :, t+1]) : A
   @inbounds for t = (T-1):-1:1 #t=1

        xt      = @view X[t, :]
        xnext   = @view X[t+1, :]
        μfilt   = @view μ_filter[t, :]
        Σfilt   = @view Σ_filter[:, :, t]
        μpred   = @view μ_pred[t+1, :]
        Σpred   = @view Σ_pred[:, :, t+1]

        BackwardSim_ref!(
            xt,
            xnext,
            μfilt,
            Σfilt,
            μpred,
            Σpred,
            A,
            KG_buf, tmp_vec, tmp_mat, tmp_mat2
        )
    end
   
    x0 = zeros(n)
    BackwardSim_ref!(
        x0,
        view(X, 1, :),
        μ₀,
        Σ₀,
        view(μ_pred, 1, :),
        view(Σ_pred, :, :, 1),
        A, 
        KG_buf, tmp_vec, tmp_mat, tmp_mat2
    )

    return [x0'; X]
end

#μ=μ_upd
#Σ= Σ_upd
#μ̄ =μ̄_buf
#Ω̄=Ω̄_buf
#u, y, At, B,
# Cargs=Cargs_t
#Σₑ=Σₑ_g[t], γ, ωₘ, ωₛ,
#max_iterations=1,resid_check=false,
#Val(mode),        # ← mode here
#ws;
#clipped_partials = clipped_partials,
#INTERCEPT = INTERCEPT,
#p_threshold = p_threshold

function kalmanfilter_update_sarma!(
    μ::V,
    Σ::M,
    μ̄::V,
    Ω̄::M,
    u,
    y::V,
    A,
    B,
    Cargs,
    Σₑ::AbstractVector,
    γ::Real,
    ωₘ::Union{Nothing,AbstractVector},
    ωₛ::Union{Nothing,AbstractVector},
    max_iterations::Int,
    resid_check::Bool,
    ::Val{MODE},
    ws_ar::IEKFWorkspace,
    ws_ma::IEKFWorkspace;
    cache_ar::SARMARegCache,
    cache_ma::SARMARegCache,
    ztrans::AbstractString="partials",
    clipped_partials::Bool=false,
    INTERCEPT::Bool=false,
    p_threshold::Float64=Inf,
    startcol::Int,
    ar_cols,
    ma_cols,
    nar_inter,
    kl_ar::Int,
    kl_ma::Int
) where {MODE,V<:StridedVector,M<:StridedMatrix}

    n  = length(μ̄)
    ny = length(y)

    C̄ = zeros(ny, n)

    # -------- AR workspace --------
    C_ar   = ws_ar.C̄
    reg_ar = ws_ar.reg_terms
    Jg_ar  = ws_ar.Jg

    # -------- MA workspace --------
    C_ma   = ws_ma.C̄
    reg_ma = ws_ma.reg_terms
    Jg_ma  = ws_ma.Jg

    # -------- combined buffers --------
    # -------- combined buffers --------
        resid = ws_ar.resid
        K     = zeros(eltype(μ), n, ny)

        tmp_ny_n = Matrix{Float64}(undef, ny, n)
        tmp_n_ny = Matrix{Float64}(undef, n, ny)
        tmp_n_n  = Matrix{Float64}(undef, n, n)
        tmp_n_n2 = Matrix{Float64}(undef, n, n)
        S        = Matrix{Float64}(undef, ny, ny)
        Ubuf     = Matrix{Float64}(undef, ny, ny)
        I_n      = Matrix{Float64}(LinearAlgebra.I, n, n)

    @views begin

        # --------------------------------------------------
        # compute regression terms ψ(θ)
        # --------------------------------------------------

        θ₀ = INTERCEPT ? μ̄[1] : zero(eltype(μ̄))

        θ_ar = view(μ̄, ar_cols)
        θ_ma = view(μ̄, ma_cols)

        MultiSARMAtoReg_cached!(
            reg_ar,
            θ_ar,
            cache_ar;
            ztrans = ztrans,
            negative_signs = true
        )

        MultiSARMAtoReg_cached!(
            reg_ma,
            θ_ma,
            cache_ma;
            ztrans = ztrans,
            negative_signs = false
        )

        # --------------------------------------------------
        # Jacobian
        # --------------------------------------------------

        θ_ar_input = INTERCEPT ? [θ₀; θ_ar] : θ_ar

        jacobian_C_fast!(
            C_ar,
            reg_ar,
            Jg_ar,
            θ_ar_input,
            [view(cargsj, 1:kl_ar) for cargsj in Cargs],
            cache_ar,
            INTERCEPT = INTERCEPT,
            ztrans = ztrans,
            startcol=  startcol,
            negative_signs = true
        )

        jacobian_C_fast!(
            C_ma,
            reg_ma,
            Jg_ma,
            θ_ma,
            [view(cargsj, kl_ar+1:kl_ar+kl_ma) for cargsj in Cargs],
            cache_ma,
            INTERCEPT = false,
            ztrans = ztrans,
            startcol=  startcol,
            negative_signs = false
        )

        fill!(C̄, 0.0)
        @views C̄[:, 1:nar_inter] .= C_ar
        @views C̄[:, ma_cols] .= C_ma

        # --------------------------------------------------
        # residuals: y − h(μ̄)
        # --------------------------------------------------

        compute_resid_sarma!(resid, y, reg_ar, reg_ma, Cargs, θ₀)

    # ==================================================
    # CASE 1: sequential EKF for grouped observations
    # ==================================================
    if ny > 1

        ekf_update_sequential_diagR2!(
            μ,
            Σ,
            μ̄,
            Ω̄,
            C̄,
            resid,
            Σₑ,
            ws;
            joseph = false
        )


    clamp_partials!(μ, startcol, p_threshold)

    return nothing
end

# ==================================================
# CASE 2: scalar fallback (ny == 1)
# ==================================================

tmp = similar(μ)
mul!(tmp, Ω̄, view(C̄,1,:))
s = dot(view(C̄,1,:), tmp) + Σₑ[1]

# K = Ω̄ * C̄' / s
@inbounds for i in 1:n
    K[i,1] = zero(eltype(K))
    for k in 1:n
        K[i,1] += Ω̄[i,k] * C̄[1,k]
    end
    K[i,1] /= s
end

# μ = μ̄ + K * resid
@inbounds for i in 1:n
    μ[i] = μ̄[i] + K[i,1] * resid[1]
end

clamp_partials!(μ, startcol, p_threshold)

# Joseph covariance update
@inbounds for i in 1:n, j in 1:n
    tmp_n_n[i,j] = I_n[i,j] - K[i,1] * C̄[1,j]
end

mul!(tmp_n_n2, tmp_n_n, Ω̄)
mul!(Σ, tmp_n_n2, tmp_n_n')

@inbounds for i in 1:n, j in 1:n
    Σ[i,j] += K[i,1] * Σₑ[1] * K[j,1]
end

@inbounds for j in 1:n, i in j+1:n
    Σ[i,j] = Σ[j,i]
end

return nothing
    
end
end

function observation_sarma_scalar(carg, θ₀, reg_ar, reg_ma)
    # pseudo-structure, adapt to your Cargs layout
    # e.g. carg may contain lagged y-part and lagged e-part
    return θ₀ + dot(carg[1:kl_ar], reg_ar) + dot(carg[kl_ar+1:kl_ar+kl_ma], reg_ma)
end

function compute_resid_sarma!(resid, y, reg_ar, reg_ma, Cargs, θ₀)
    @inbounds for j in eachindex(y)
        μhat = observation_sarma_scalar(Cargs[j], θ₀, reg_ar, reg_ma)
        resid[j] = y[j] - μhat
    end
    return nothing
end