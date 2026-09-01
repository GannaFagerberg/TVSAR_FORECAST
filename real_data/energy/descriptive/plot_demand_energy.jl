using Plots

### Compare two years
function plot_years(df, start_year, end_year)
    idx = (year.(df.DATE) .>= start_year) .&
          (year.(df.DATE) .<= end_year)

    plot(
        df.DATE[idx],
        df.energy[idx],
        xlabel = "Date",
        ylabel = "Energy",
        label = false
    )
end

### Overlay years

using Plots, Dates

function overlay_years(df, years)

    p = plot(
        xlabel = "Date",
        ylabel = "Energy",
        legend = :topright
    )

    for yr in years
        idx = year.(df.DATE) .== yr

        dates_yr = df.DATE[idx]
        y_yr     = df.energy[idx]

        # Map all years onto the same calendar year
        common_dates = Date.(2000, month.(dates_yr), day.(dates_yr))

        plot!(
            p,
            common_dates,
            y_yr,
            label = string(yr),
            lw = 1.5
        )
    end

    return p
end

### Inspect the data

function plot_weeks(df, start_date; nweeks=1)

    end_date = start_date + Day(7 * nweeks - 1)

    idx = (df.DATE .>= start_date) .&
          (df.DATE .<= end_date)

    dates_plot = df.DATE[idx]
    y_plot     = df.energy[idx]

    plot(
        dates_plot,
        y_plot,
        xticks = (dates_plot, dayname.(dates_plot)),
        xrotation = 45,
        xlabel = "",
        ylabel = "Energy",
        label = false,
        marker = :circle,
        lw = 2,
        size = (900, 450)
    )
end


### Plot several weeks ahead 
plot_weeks(df_train, Date(2018, 7, 1), nweeks=5)

### Plot two years
plot_years(df_train, 2018, 2020)
plot_years(df_train, 2021, 2022)

### Plot single year
idx = year.(df_train.DATE) .== 2021
plot(
    df_train.DATE[idx],
    df_train.energy[idx],
    xlabel = "Date",
    ylabel = "Energy",
    label = false
)

### Overlay years
p = overlay_years(df_train, 2020:2021)
display(p)

### Plot some dates_yr
plot(df_train.energy[1:21], xlabel = "Date", ylabel = "Energy", label = false)