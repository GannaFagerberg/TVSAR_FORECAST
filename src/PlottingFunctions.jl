
colors = Base.parse.(Colorant,[
    "#6C8EBF", "#c0a34d", "#780000", "#007878",     
    "#b5c6df","#eadaaa","#AE6666", "#4CA0A0","#bf9d6c", "#3A6B35", 
    "#9d6a6d","#d9c6c7", "#98bbb9", "#bf8d6c", 
    "#CBD18F"])


function summarize_and_plot_t(
    trans_theta;
    prefix = L"",
    ylim = (-1,1),
    xlim = nothing,
    true_phi = nothing,
    alpha = 0.05,
    use_hdi = true,
    xindex = nothing,
    xticks = nothing
)

    col_true   = "#c0a34d"
    col_median = "#780000"
    col_band   = "#6C8EBF"

    T, ncoeff, niter = size(trans_theta)

    # choose x-axis
    if xindex === nothing
    xvals = 1:T
    else
        xvals = xindex
    end

    for j in 1:ncoeff

        coeff = trans_theta[:, j, :]

        lower_bound = zeros(T)
        upper_bound = zeros(T)
        median_vals = zeros(T)

        for t in 1:T
            samples = view(coeff, t, :)

            if use_hdi
                hdi_low, hdi_high = hdi(samples, 1 - alpha)
                lower_bound[t] = hdi_low
                upper_bound[t] = hdi_high
            else
                lower_bound[t] = quantile(samples, alpha/2)
                upper_bound[t] = quantile(samples, 1 - alpha/2)
            end

            median_vals[t] = median(samples)
        end

        plt = plot(
            xvals,
            median_vals,
            label = L"\text{median}",
            color = col_median,
            lw = 2,
            ylim = ylim,
            title = prefix,
            titlefont = font(18),
            xticks = xticks
        )

        plot!(xvals, lower_bound, label="", color=col_band, lw=1.5, legend=false)
        plot!(xvals, upper_bound, label="", color=col_band, lw=1.5, legend=false)

        if true_phi !== nothing
            plot!(
                xvals,
                true_phi[:, j],
                label = L"\text{true }\phi",
                color = col_true,
                lw = 2,
                linestyle = :dot
            )
        end

        if xlim !== nothing
            xlims!(xlim)
        end

        display(plt)
    end
end


### HDI function
function hdi(samples::AbstractVector{<:Real}, cred_mass::Float64=0.95)
    sorted_samples = sort(samples)
    n_samples = length(samples)
    cred_idx = floor(Int, cred_mass * n_samples)
    n_intervals = n_samples - cred_idx
    min_width = Inf
    hdi_low = NaN
    hdi_high = NaN

    for i in 1:n_intervals
        low = sorted_samples[i]
        high = sorted_samples[i + cred_idx]
        width = high - low
        if width < min_width
            min_width = width
            hdi_low = low
            hdi_high = high
        end
    end
    return hdi_low, hdi_high
end


function transform_theta(theta; ztrans="partials", negative_signs=true)
    
    nt, nv, niter = size(theta)
    out = similar(theta)  # allocate output with same size/type

    for t in 1:nt
        for it in 1:niter
            x = @view theta[t, :, it]      # vector of length 3
            ϕ, _ = arma_reparam(x; ztrans=ztrans, threshold = nothing, negative_signs=negative_signs )        # apply your function
            out[t, :, it] .= ϕ            # store transformed vector
        end
    end
    return out
end

function plot_state(x; prefix, ylim, xlim, true_phi=nothing,  alpha=0.05, use_hdi=true)
    summarize_and_plot(x; prefix=prefix, ylim=ylim, xlim=xlim,true_phi=true_phi, alpha=alpha, use_hdi=use_hdi)
end



### For forecasts
function summarize_and_plot(
    trans_theta;
    prefix = L"",
    ylim = (-1,1),
    xlim = (0,500),
    true_phi = nothing,
    alpha = 0.05,
    use_hdi = true
)

    col_true   = "#c0a34d"
    col_median = "#780000"
    col_band   = "#6C8EBF"

    T, ncoeff, niter = size(trans_theta)

    for j in 1:ncoeff

        coeff = trans_theta[:, j, :]

        lower_bound = zeros(T)
        upper_bound = zeros(T)
        median_vals = zeros(T)

        for t in 1:T
            samples = view(coeff, t, :)

            if use_hdi
                hdi_low, hdi_high = hdi(samples, 1 - alpha)
                lower_bound[t] = hdi_low
                upper_bound[t] = hdi_high
            else
                lower_bound[t] = quantile(samples, alpha/2)
                upper_bound[t] = quantile(samples, 1 - alpha/2)
            end

            median_vals[t] = Statistics.median(samples)
        end

        plt = plot(
            1:T,
            median_vals,
            label = L"\text{median}",
            color = col_median,
            lw = 2,
            ylim = ylim,
            xlim = xlim,
            title = prefix,
            titlefont = font(18)
        )

        plot!(1:T, lower_bound, label="", color=col_band, lw=1.5)
        plot!(1:T, upper_bound, label="", color=col_band, lw=1.5)

        if true_phi !== nothing
            plot!(
                1:T,
                true_phi[:, j],
                label = L"\text{true }\phi",
                color = col_true,
                lw = 2,
                linestyle = :dot
            )
        end

        display(plt)
    end
end
