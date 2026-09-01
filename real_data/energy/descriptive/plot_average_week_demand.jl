using DataFrames
using Dates
using Statistics
using Plots

default(
    background_color = :white,
    background_color_subplot = :white,
    framestyle = :axes,
    grid = false
)

# ============================================================
# SETTINGS
# ============================================================

n_weeks = 3      # change to 3 for three weeks
H = 24 * 7 * n_weeks

season_specs = [
    (
        name   = "Summer",
        months = [12, 1, 2],
        color  = "#3A6B35"
    ),
    (
        name   = "Autumn",
        months = [3, 4, 5],
        color  = "#bf8d6c"
    ),
    (
        name   = "Winter",
        months = [6, 7, 8],
        color  = "#780000"
    ),
    (
        name   = "Spring",
        months = [9, 10, 11],
        color  = "#007878"
    )
]


# ============================================================
# Average consecutive n-week profile for a season
# ============================================================

function seasonal_multiweek_profile(
    df_train,
    months_selected;
    n_weeks = 2
)

    H = 24 * 7 * n_weeks

    timestamps =
        df_train.datetime

    # --------------------------------------------------------
    # Candidate windows start Monday at 00:00
    # --------------------------------------------------------

    candidate_idx = findall(
        (dayofweek.(timestamps) .== 1) .&
        (hour.(timestamps) .== 0)
    )

    profiles =
        Vector{Vector{Float64}}()

    starts_used =
        DateTime[]


    for idx in candidate_idx

        last_idx =
            idx + H - 1

        # Window must fit inside training sample
        if last_idx > nrow(df_train)
            continue
        end

        window =
            df_train[idx:last_idx, :]

        # ----------------------------------------------------
        # Check that it really is H consecutive hourly obs.
        # ----------------------------------------------------

        expected_end =
            timestamps[idx] + Hour(H - 1)

        if window.datetime[end] != expected_end
            continue
        end

        # ----------------------------------------------------
        # Entire window must belong to selected season
        # ----------------------------------------------------

        if !all(
            m -> m in months_selected,
            month.(window.datetime)
        )
            continue
        end

        push!(
            profiles,
            Float64.(window.demand_VIC)
        )

        push!(
            starts_used,
            window.datetime[1]
        )
    end


    @assert !isempty(profiles) """
    No complete $(n_weeks)-week windows found
    for months $months_selected.
    """


    # --------------------------------------------------------
    # H × number of seasonal windows
    # --------------------------------------------------------

    profile_matrix =
        hcat(profiles...)

    mean_profile =
        vec(
            mean(
                profile_matrix,
                dims = 2
            )
        )


    return (
        mean_profile = mean_profile,
        n_windows    = length(profiles),
        starts_used  = starts_used
    )
end


# ============================================================
# CALCULATE SEASONAL PROFILES
# ============================================================

season_profiles =
    Dict()

for s in season_specs

    result =
        seasonal_multiweek_profile(
            df_train,
            s.months;
            n_weeks = n_weeks
        )

    season_profiles[s.name] =
        result

    println(
        s.name,
        ": ",
        result.n_windows,
        " complete ",
        n_weeks,
        "-week windows"
    )
end


# ============================================================
# PLOT
# ============================================================

p_season_weeks = plot(
    xlabel = "",
    ylabel = "Average electricity demand",
    legend = :topright,
    legendfontsize = 8,
    size = (1200, 500),
    left_margin = 8Plots.mm
)

x =collect(0:(H - 1))


for s in season_specs

    prof =
        season_profiles[s.name]

    plot!(
        p_season_weeks,
        x,
        prof.mean_profile;
        color = s.color,
        lw = 2,
        label = s.name
    )
end


# ============================================================
# X-AXIS
# Mon Tue ... Sun | Mon Tue ... Sun
# ============================================================

weekdays =
    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

day_labels =
    repeat(
        weekdays,
        n_weeks
    )

day_positions =
    collect(
        12:24:(H - 12)
    )

plot!(
    p_season_weeks;
    xticks = (
        day_positions,
        day_labels
    ),
    xlims = (0, H - 1)
)


# ============================================================
# DAY BOUNDARIES
# ============================================================

vline!(
    p_season_weeks,
    collect(24:24:(H - 24));
    color = :gray,
    linestyle = :dash,
    alpha = 0.20,
    lw = 0.7,
    label = false
)


# ============================================================
# STRONGER WEEK BOUNDARY
# ============================================================

week_boundaries =
    collect(
        168:168:(H - 1)
    )

if !isempty(week_boundaries)

    vline!(
        p_season_weeks,
        week_boundaries;
        color = :gray,
        linestyle = :solid,
        alpha = 0.6,
        lw = 1.3,
        label = false
    )

end


display(p_season_weeks)


# ============================================================
# SAVE
# ============================================================

save_dir =
    raw"C:\Users\Anna Fagerberg\JuliaPackages\TVSAR_FORECAST\cluster\results"

savefig(
    p_season_weeks,
    joinpath(
        save_dir,
        "victoria_average_$(n_weeks)_week_profiles_by_season.pdf"
    )
)