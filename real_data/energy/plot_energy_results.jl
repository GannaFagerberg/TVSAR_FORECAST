using Statistics
using StatsBase
using Plots
using Dates


default(
    background_color = :white,
    background_color_subplot = :white,
    framestyle = :axes,   # ✅ keeps x & y axes only
    grid = false
)

# Number of estimated/grouped states
T_group = size(SAR44_res[1], 1)

# Map grouped states to dates across the training sample
idx_group = round.(Int, range(1, length(timestamp_train), length=T_group))
time_ind  = timestamp_train[idx_group]

# Continuous year representation for x-axis
time_group = year.(time_ind) .+ (month.(time_ind) .- 1) ./ 12

# One tick per year
years = unique(year.(time_ind))

tick_pos = [
    time_group[findfirst(==(yr), year.(time_ind))]
    for yr in years
]

xticks_custom = (
    tick_pos,
    string.(years)
)


summarize_and_plot_t(
    SAR_res[1][:, 1:1, :] .+x_med;
    ylim = (5100, 5200),
    prefix = "Intercept",
    true_phi = nothing,
    xindex = time_group,
    xticks = xticks_custom
)

# ==========================================================
# 2. REGULAR AR(1)
# ==========================================================


# Number of regular AR coefficients
p_reg = p1[1]

# Transform REGULAR AR block only
trans_AR_reg = transform_theta(
    SAR_res[1][:, startcol:(startcol + p_reg - 1), :],
    ztrans = ztrans,
    negative_signs = !SMA
)

for j in 1:p_reg

    summarize_and_plot_t(
        trans_AR_reg[:, j:j, :];
        ylim = (-1, 1),
        prefix = latexstring("\\phi_{$j,t}"),
        true_phi = nothing,
        xindex = time_group,
        xticks = xticks_custom
    )

end

# ==========================================================
# Seasonal AR coefficients only
# ==========================================================

nseasonal = length(season) - 1

# First column after the regular AR block
seasonal_startcol = startcol + p1[1]

for k in 2:length(season) #k=3

    p_seas = p1[k]          # AR order for this seasonal period
    s      = season[k]      # seasonal period, e.g. 24 or 168

    # Columns belonging to this seasonal AR block
    cols = seasonal_startcol:(seasonal_startcol + p_seas - 1)

    # Transform THIS seasonal block only
    trans_AR_seas = transform_theta(
        SAR_res[1][:, cols, :],
        ztrans = ztrans,
        negative_signs = !SMA
    )

    # Plot coefficients within this seasonal block
    for j in 1:p_seas

        summarize_and_plot_t(
            trans_AR_seas[:, j:j, :];
            ylim = (-1, 1),
            prefix = latexstring("\\Phi_{$j,t}^{($s)}"),
            true_phi = nothing,
            xindex = time_group,
            xticks = xticks_custom
        )

    end

    # Move to start of next seasonal block
    seasonal_startcol += p_seas
end

# ==========================================================
# MEASUREMENT VOLATILITY σ_{e,t}
# ==========================================================

histogram( SAR44_res[3])


