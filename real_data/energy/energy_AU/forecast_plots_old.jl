H_freeze    = false
STATE_fixed = false
use_fourier = false

g_new  = 1
T      = length(y_g)
season = s1

forecastHorizons = collect(1:length(y_test))

nPredPerIter      = 20
statTrans         = ztrans

# ----------------------------------------------------------
# Initial lag vector for forecasting
# ----------------------------------------------------------

T_all = length(x)
zₜall_orig = reverse!(x[T_all  - p_max[1] + 1:T_all])
activeLags = activeLags_ar

# ----------------------------------------------------------
# Extract posterior states at time T
# ----------------------------------------------------------

θₜpost = SAR_res[1][T,:,:]
#Hₜpost = SAR44_res[10]

#histogram(SAR_res[5][1,:])
#plot(SAR_res[2][:,1,2])

if algoSettings.state_var_type == :DSP

    Hₜpost = SAR_res[2][end, :, :] .-
             log(nPerGroup_fit) .+
             log(g_new)

    μpost = SAR_res[5] .-
            log(nPerGroup_fit) .+
            log(g_new)

    ϕpost = SAR_res[4]

else
    # Hₜpost is VARIANCE here
    Hₜpost = (SAR_res[7] ./ nPerGroup_fit) .* g_new
    ϕpost = nothing
    μpost = nothing
end

#σₑₜpost = expand_grouped_states(SAR44_res[3], l, T_all)[T_all, 1:end]
σₑₜpost = SAR_res[3]

if use_fourier

    fourier_coeffs = SAR44_res[11][:,25,:] # static
    #fourier_coeffs = SAR44_res[12]
    #fourier_coeffs = SAR44_res[8]
    
    F_future = build_future_fourier_matrix(
        forecastHorizons;
        period = 24 * 365,
        K      = div(size(fourier_coeffs, 1), 2),
        t_last = T_all
    )

    S_future = F_future * fourier_coeffs
    four_s = reshape(S_future , size(S_future ,1), 1, size(S_future ,2))

    
plot_state(
    four_s;
    prefix="Errors",
    ylim=(-0.9, 0.9),
    xlim=(0,336),
    true_phi=nothing
)

end


# DSP parameters
# Initialize optional objects
σ²ₙpost   = nothing
ϕ̃post    = nothing
μ̃post    = nothing
h̃post    = nothing
σ̄²ₙpost   = nothing

# ----------------------------------------------------------
# Handle SV / SVDSP cases
# ----------------------------------------------------------

if algoSettings.obs_var_type == :SV
    μ̃post  = SAR44_res[6][1,:]
    ϕ̃post  = SAR44_res[7][1,:]
    σ̄²ₙpost = SAR44_res[9][1,:]

elseif algoSettings.obs_var_type == :SVDSP

    μ̃post = SAR44_res[6][1,:]
    ϕ̃post = SAR44_res[7][1,:]
    #h̃post = expand_grouped_states(SAR44_res[8], l, T_all)[T_all, 1:end]
    h̃post = SAR44_res[8][T_all, 1:end]

    #plot(h̃post)
end

# ----------------------------------------------------------
# Run Forecast Simulation
# ----------------------------------------------------------
t_st = time()

res_forecast, LPS_est, AE = PredLocalMultiSAR_SV_gr(
    nPredPerIter,
    y_test,
    zₜall_orig,
    p1,
    season,
    statTrans,
    θₜpost,
    Hₜpost,
    ϕpost,
    σ²ₙpost,
    μpost,
    σₑₜpost,
    ϕ̃post,
    μ̃post,
    h̃post,
    σ̄²ₙpost,
    forecastHorizons;
    state_var_type = state_var_type,
    obs_var_type   = obs_var_type,
    g_fc = g_new,
    INTERCEPT = INTERCEPT,
    H_freeze = false,
    STATE_fixed = false,
    use_fourier = false,
    intercept_dynamics = intercept_dynamics,
    startcol = startcol,
    p_threshold = p_threshold
)

t_end = time()

println("Elapsed: ", (t_end - t_st) / 60, " mins")

plot(
    forecastHorizons,
    LPS_est,
    xlabel = "Forecast horizon",
    ylabel = "LPS",
    legend = false
)

plot(
    forecastHorizons,
    AE,
    xlabel = "Forecast horizon",
    ylabel = "Absolute error",
    legend = false
)


# ----------------------------------------------------------
# Extract Forecast Results
# ----------------------------------------------------------

y_res = res_forecast

h_length=maximum(forecastHorizons)

y_h50 = reshape(
    y_res,
    length(y_test[1:h_length]),
    1,
    size(y_res, 2)
)

#res_transf = exp.(y_h50.+ x_mean).*scale_factor

log_transform = false
if log_transform
    res_transf =  exp.((y_h50 .+ train_mean))
else
    res_transf =  (y_h50 .+ train_mean)
end

y_mat = reshape(y_test, :, 1)
#truth = exp.(y_mat[1:forecastHorizons,:])
truth = y_mat[1:h_length,:] .+ train_mean

plot_state(
    #res_transf[1:forecastHorizons,:,:].+future_seasonal[1:forecastHorizons] ;
    res_transf[1:h_length,:,:] ;
    prefix = "TV-SAR(1,1,1)s=24,168",
    ylim = (5,10),
    xlim = (0,forecastHorizons),
    true_phi =truth,
    alpha = 0.05,
    use_hdi = true
)

#cd("C:/Users/Anna Fagerberg/Desktop/PROJECTS IN JULIA/PAPERS_Julia/SARMA/plots_sarma/presentation")
#savefig("TVSAR222_12_168_energy_forecasts.pdf")
#savefig("TVSAR22_24_energy_forecasts.pdf")

#cd("C:/Users/Anna Fagerberg/Desktop/PROJECTS IN JULIA/PAPERS_Julia/SARMA/REAL DATA/Electricity_Australia/res_four_block")
#savefig("UK_TVSAR222_24_168_gr_m_week_fr_joint_f_no_int.pdf")
#

#cd("C:/Users/Anna Fagerberg/Desktop/PROJECTS IN JULIA/PAPERS_Julia/SARMA/REAL DATA/Electricity_Australia/res_four_block")

#R_forecasts = CSV.read("sarima_forecast.csv", DataFrame)

#using Plots
#using Statistics

function compute_forecast_summary(res_transf; alpha=0.05)

    T, ncoeff, niter = size(res_transf)

    median_fc = zeros(T, ncoeff)
    lower_fc  = zeros(T, ncoeff)
    upper_fc  = zeros(T, ncoeff)

    for j in 1:ncoeff
        for t in 1:T
            samples = view(res_transf, t, j, :)

            median_fc[t,j] = median(samples)
            lower_fc[t,j]  = quantile(samples, alpha/2)
            upper_fc[t,j]  = quantile(samples, 1 - alpha/2)
        end
    end

    return median_fc, lower_fc, upper_fc
end


h_length = length(y_test[1:168])

using Plots
# --- your palette ---
col_true   = colors[2]
col_tv_med = colors[3]
col_tv_band= colors[3]

# --- R palette ---
col_r_med  = colors[1]
col_r_band = colors[1]

h_length = length(y_test[1:12])

median_fc, lower_fc, upper_fc = compute_forecast_summary(
    res_transf[1:h_length,:,:];
    alpha = 0.05
)





# extract (1 coeff)
fc_mean_julia1  = median_fc[:,1]
fc_lower_julia1 = lower_fc[:,1]
fc_upper_julia1 = upper_fc[:,1]


fc_mean_julia2  = median_fc2[:,1]
fc_lower_julia2 = lower_fc2[:,1]
fc_upper_julia2 = upper_fc2[:,1]

h = 1:length(R_forecasts.mean)

plot(h, fc_mean_julia,
     lw=3,
     color=col_tv_med,
     label="TV-SAR",
     background_color=:white,
     framestyle=:axes,
     grid=false,
     title="TV-SAR vs static SARIMA", linestyle=:dot)

# --- TV bands (solid, thinner) ---
plot!(h, fc_lower_julia, lw=2, color=col_tv_band, label="")
plot!(h, fc_upper_julia, lw=2, color=col_tv_band, label="")

# --- R forecasts ---
plot!(h, R_forecasts.mean,
      lw=3,
      color=col_r_med,
      label="Static SARIMA", linestyle=:dot)

plot!(h, R_forecasts.lower, lw=2, color=col_r_band, label="")
plot!(h, R_forecasts.upper, lw=2, color=col_r_band, label="")

# --- truth (ONLY dashed) ---
plot!(h, y_test[1:h_length],
      lw=3,
      color=col_true,
      linestyle=:dot,
      label="Truth")

##### SHADED

#using Plots

# --- your palette ---
col_true   = colors[2]
col_tv_med = colors[3]

# --- derived light shades (important) ---
col_tv_fill = colors[3]   # same color, but we use transparency
col_r_med   = colors[1]
col_r_fill  = colors[1]

h_length = length(y_test[1:12])

median_fc, lower_fc, upper_fc = compute_forecast_summary(
    res_transf[1:h_length,:,:];
    alpha = 0.05
)

# extract
fc_mean_julia  = median_fc[:,1]
fc_lower_julia = lower_fc[:,1]
fc_upper_julia = upper_fc[:,1]

h = 1:length(R_forecasts.mean)

# --- TV-SAR (with shaded band) ---
plot(h, fc_mean_julia,
     lw=2.5,
     color=col_tv_med,
     ribbon=(fc_mean_julia - fc_lower_julia,
             fc_upper_julia - fc_mean_julia),
     fillalpha=0.2,                  # ← key for light shading
     label="TV-SAR",
     background_color=:white,
     framestyle=:axes,
     grid=false,
     title="TV-SAR vs static SARIMA")

# --- Static SARIMA (shaded band) ---
plot!(h, R_forecasts.mean,
      lw=2.5,
      color=col_r_med,
      ribbon=(R_forecasts.mean .- R_forecasts.lower,
              R_forecasts.upper .- R_forecasts.mean),
      fillalpha=0.15,
      label="Static SARIMA")

# --- truth ---
plot!(h, y_test[1:h_length],
      lw=2.5,
      color=col_true,
      linestyle=:dot,
      label="Truth")

#savefig("Sim_TVSAR11_forecasts_comparison.pdf")



##### SHADED ENERG

using Plots

# --- your palette ---
col_true   = colors[2]
col_tv_med = colors[3]

# --- derived light shades (important) ---
col_tv_fill = colors[3]   # same color, but we use transparency
col_r_med   = colors[1]
col_r_fill  = colors[1]

h_length = length(y_test[1:168])

h = 1:h_length 

# --- TV-SAR (with shaded band) ---
plot(h, fc_mean_julia1,
     lw=2.5,
     color=col_tv_med,
     ribbon=(fc_mean_julia1 - fc_lower_julia1,
             fc_upper_julia1 - fc_mean_julia1),
     fillalpha=0.2,                  # ← key for light shading
     label="TVSAR (daily and weekly)",
     background_color=:white,
     framestyle=:axes,
     grid=false,
     title="TVSAR(2,2,2) (daily,weekly) vs TVSAR(2,2) (daily)")

# --- Static SARIMA (shaded band) ---
plot!(h, fc_mean_julia2,
      lw=2.5,
      color=col_r_med,
      ribbon=(fc_mean_julia2 .- fc_lower_julia2,
              fc_upper_julia2 .- fc_mean_julia2),
      fillalpha=0.15,
      label="TVSAR (daily)")

# --- truth ---
plot!(h, y_test[1:h_length],
      lw=2.5,
      color=col_true,
      linestyle=:dot,
      label="Truth")

xticks!(xticks_pos, xticks_lab)
#

# pick one point per day (midnight)
idx = findall(t -> hour(t) == 12, times)

xticks_pos = idx
xticks_lab = first.(dayname.(times[idx]), 3)