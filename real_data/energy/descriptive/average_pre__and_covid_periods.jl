using DataFrames
using Dates
using Statistics
using Plots
using CSV

default(
    background_color = :white,
    background_color_subplot = :white,
    framestyle = :axes,
    grid = false
)

# ============================================================
# SETTINGS
# ============================================================

# Pre-COVID period = your estimation sample
pre_start =
    minimum(df_train.datetime)

pre_end =
    maximum(df_train.datetime)

# Start immediately after training sample to avoid overlap
covid_start =
    pre_end + Hour(1)

# End of COVID period used in your analysis
covid_end =
    DateTime(2021, 7, 19, 23)


col_pre =
    "#007878"

col_covid =
    "#780000"


season_specs = [
    (
        name   = "Summer",
        months = [12, 1, 2]
    ),
    (
        name   = "Autumn",
        months = [3, 4, 5]
    ),
    (
        name   = "Winter",
        months = [6, 7, 8]
    ),
    (
        name   = "Spring",
        months = [9, 10, 11]
    )
]


println("Pre-COVID:")
println(pre_start, " -- ", pre_end)

println("\nCOVID:")
println(covid_start, " -- ", covid_end)


# ============================================================
# FUNCTION
# Average demand by hour of week
#
# Monday 00:00 = 0
# ...
# Sunday 23:00 = 167
# ============================================================

function average_week_profile(
    df,
    period_start,
    period_end,
    months_selected
)

    # --------------------------------------------------------
    # Restrict to period and season
    # --------------------------------------------------------

    mask =
        (df.datetime .>= period_start) .&
        (df.datetime .<= period_end) .&
        in.(
            month.(df.datetime),
            Ref(months_selected)
        )

    d =
        df[mask, :]


    # --------------------------------------------------------
    # Hour of week
    #
    # Monday 00:00 -> 0
    # Monday 23:00 -> 23
    # Tuesday 00:00 -> 24
    # ...
    # Sunday 23:00 -> 167
    # --------------------------------------------------------

    hour_of_week =
        (dayofweek.(d.datetime) .- 1) .* 24 .+
        hour.(d.datetime)


    tmp =
        DataFrame(
            hour_of_week = hour_of_week,
            demand       = Float64.(d.demand_VIC)
        )


    # --------------------------------------------------------
    # Average all observations corresponding to each
    # hour of the week
    # --------------------------------------------------------

    profile =
        combine(
            groupby(
                tmp,
                :hour_of_week
            ),
            :demand => mean => :mean_demand,
            :demand => length => :n
        )

    sort!(
        profile,
        :hour_of_week
    )

    @assert nrow(profile) == 168

    return profile
end


# ============================================================
# COMPUTE PROFILES
# ============================================================

season_results =
    Dict()

for s in season_specs

    pre =
        average_week_profile(
            df,
            pre_start,
            pre_end,
            s.months
        )

    covid =
        average_week_profile(
            df,
            covid_start,
            covid_end,
            s.months
        )

    season_results[s.name] = (
        pre = pre,
        covid = covid
    )


    println()
    println(s.name)

    println(
        "  Pre-COVID observations: ",
        sum(pre.n)
    )

    println(
        "  COVID observations:     ",
        sum(covid.n)
    )

    println(
        "  Mean pre-COVID demand:   ",
        round(mean(pre.mean_demand), digits = 1)
    )

    println(
        "  Mean COVID demand:       ",
        round(mean(covid.mean_demand), digits = 1)
    )
end


# ============================================================
# AXIS
# ============================================================

day_positions =
    collect(
        12:24:156
    )

day_labels =
    [
        "Mon",
        "Tue",
        "Wed",
        "Thu",
        "Fri",
        "Sat",
        "Sun"
    ]

day_boundaries =
    collect(
        24:24:144
    )


# ============================================================
# FOUR PANELS
# ============================================================

season_plots =
    Plots.Plot[]

for (i, s) in enumerate(season_specs)

    res =
        season_results[s.name]


    # --------------------------------------------------------
    # Pre-COVID
    # --------------------------------------------------------

    p = plot(
        res.pre.hour_of_week,
        res.pre.mean_demand;
        color = col_pre,
        lw = 2,
        label =
            i == 1 ?
            "Pre-COVID" :
            "",
        ylabel = "Average demand",
        title = s.name,
        titlefont = font(10),
        legend =
            i == 1 ?
            :topright :
            false,
        legendfontsize = 8,
        xlims = (0, 167),
        xticks =
            i == 4 ?
            (day_positions, day_labels) :
            false,
        left_margin = 10Plots.mm
    )


    # --------------------------------------------------------
    # COVID
    # --------------------------------------------------------

    plot!(
        p,
        res.covid.hour_of_week,
        res.covid.mean_demand;
        color = col_covid,
        lw = 2,
        label =
            i == 1 ?
            "COVID-19 period" :
            ""
    )


    # --------------------------------------------------------
    # Day boundaries
    # --------------------------------------------------------

    vline!(
        p,
        day_boundaries;
        color = :gray,
        linestyle = :dash,
        alpha = 0.20,
        lw = 0.7,
        label = false
    )


    push!(
        season_plots,
        p
    )
end


# ============================================================
# COMBINE
# ============================================================

p_covid_seasons = plot(
    season_plots...;
    layout = (4, 1),
    link = :x,
    size = (1100, 1000),
    margin = 3Plots.mm
)

display(p_covid_seasons)


# ============================================================
# SAVE FIGURE
# ============================================================

save_dir =
    raw"C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST\cluster\results"

savefig(
    p_covid_seasons,
    joinpath(
        save_dir,
        "victoria_average_week_covid_vs_precovid_by_season.pdf"
    )
)


# ============================================================
# OPTIONAL: SAVE NUMERICAL PROFILES
# ============================================================

profile_out =
    DataFrame()

for s in season_specs

    res =
        season_results[s.name]

    tmp =
        DataFrame(
            Season =
                fill(s.name, 168),

            Hour_of_week =
                res.pre.hour_of_week,

            Pre_COVID =
                res.pre.mean_demand,

            COVID =
                res.covid.mean_demand,

            Difference =
                res.covid.mean_demand .-
                res.pre.mean_demand
        )

    append!(
        profile_out,
        tmp
    )
end


CSV.write(
    joinpath(
        save_dir,
        "victoria_covid_vs_precovid_weekly_profiles.csv"
    ),
    profile_out
)