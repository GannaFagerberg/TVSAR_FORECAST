module TVSAR_FORECAST

using Polynomials
using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Dates
using Serialization
using BandedMatrices

using CSV
using DataFrames

using Distributions
using SpecialFunctions
using StatsBase
using LogExpFunctions

using Optim
using ForwardDiff
using Roots
using ProgressMeter

using Polynomials
using KernelFunctions
using PDMats
using BandedMatrices
using PolyaGammaSamplers
using SpecTools

using Colors
using Measures
using LaTeXStrings
using Plots
using RCall
using Base.Threads

include("ARMAReparam.jl")
export FindActiveLagsMultiSAR,
       SetupARReg_active,
       SetupARReg,
       SARMARegCache,
       build_sarma_cache,
       arma_reparam!,
       arma_reparam,
       MultiSARMAtoReg_cached!,
       MultiSARMAtoReg_cached

include("build_AR_presample.jl")
       build_presample_workspace,
       sample_AR_presample!,
       sample_AR_presample_recursive!,
       build_AR_init_opt_orig!

include("build_MA_errors.jl")
export SMAWorkspace,
       build_SMA_workspace,
       build_MA_errors_banded,
       sample_SMA_presample_u_info!,
       sample_SMA_presample_u_simple!,
       update_beta!,
       update_beta_tvvar!,
       update_beta_hd_cd_tvvar!,
       update_beta_hd_cd_tvvar_no_cov!,
       update_beta_hd_cd_mean,
       init_beta_hd_cd

include("DSPSampler.jl")
export cg_sum,
       grouped_Z_scale,
       build_z!,
       loglik_phi_grouped,
       atanh_safe,
       Updateϕ_grouped_MH_transformed!,
       Updateϕ_grouped_MH!,
       update_dsp!,
       update_dsp_grouped!,
       Updateσ²ₙ_grouped,
       grouped_normal_variance,
       LogChi2Mix10,
       prepare_logchi2_mix10,
       UpdateMixAlloc,
       SetUpLogChi2Mixture,
       Update_h,
       Updateξ1,
       Updateμ,
       Updateϕ,
       LogVol2Covs,
       Vol2Covs,
       UpdateErrorVolatility,
       Updateσ²ₙ,
       ScaledInverseChiSq,
       UpdateErrorVolatility_grouped!,
       expand_sigma_grouped!

include("Gibbs.jl")
export IEKFWorkspace,
       GibbsSamplerTVSARMA_full,
       FFBSx,
       kalmanfilter_update_IEKF_seq!,
       ekf_update_sequential_diagR2!,
       BackwardSim!,
       FFBSx_sarma,
       kalmanfilter_update_sarma!,
       observation_sarma_scalar,
       compute_resid_sarma!

include("Jacobian_fast.jl")
    export JacobianWorkspace,
       build_jacobian_workspace,
       jacobian_reg_terms!,
       jacobian_C_fast!

include("TVSARUtils.jl")
    export build_group_map,
       group_vector,
       group_vector_view,
       expand_grouped_intercept!,
       compute_resid!,
       clamp_partials!,
       expand_grouped_states_fast!,
       compute_conditional_mean!,
       compute_residuals!,
       compute_noise_SARMA,
       Arima,
       build_Cargs_fast_threaded!,
       expand_grouped_states,
       select_period,
       apply_fisher_scaling


include("ForecastFunction.jl")
export true_f,
       PredLocalMultiSAR_SV_gr

include("PlottingFunctions.jl")
export summarize_and_plot_t,
       hdi,
       transform_theta,
       plot_state,
       summarize_and_plot


include("FisherFunctions.jl")
export  FisherInfo_full_global_gaussian,  FisherInfo_full_local_gaussian
       
end # module TVSAR_FORECAST
