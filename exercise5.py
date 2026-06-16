import heapq
import math
import random
import statistics


SEED = 20260614
Z_975 = 1.96


def mean(values):
    return sum(values) / len(values)


def sample_var(values):
    if len(values) < 2:
        return 0.0
    return statistics.variance(values)


def sample_sd(values):
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def covariance(xs, ys):
    xbar = mean(xs)
    ybar = mean(ys)
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / (len(xs) - 1)


def correlation(xs, ys):
    sx = sample_sd(xs)
    sy = sample_sd(ys)
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return covariance(xs, ys) / (sx * sy)


def ci(mean_value, sd_value, n):
    half_width = Z_975 * sd_value / math.sqrt(n)
    return mean_value - half_width, mean_value + half_width


def summarize_replications(estimates, exact=None):
    m = mean(estimates)
    sd = sample_sd(estimates)
    low, high = ci(m, sd, len(estimates))
    out = {
        "estimate": m,
        "sd": sd,
        "low": low,
        "high": high,
    }
    if exact is not None:
        out["abs_error"] = abs(m - exact)
    return out


def fmt(x, digits=6):
    if isinstance(x, str):
        return x
    if x is None:
        return ""
    if math.isinf(x):
        return "inf"
    if abs(x) >= 10000 or (0 < abs(x) < 0.0001):
        return f"{x:.3e}"
    return f"{x:.{digits}f}"


def print_table(title, headers, rows):
    widths = []
    for i, header in enumerate(headers):
        widths.append(max(len(header), max(len(str(row[i])) for row in rows)))

    print(title)
    print("-" * len(title))
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))))
    print()


# parts 1-4: integral of exp(x) from 0 to 1


def crude_integral_replication(rng, n):
    return mean([math.exp(rng.random()) for _ in range(n)])


def antithetic_integral_replication(rng, n_function_evals):
    pairs = n_function_evals // 2
    values = []
    for _ in range(pairs):
        u = rng.random()
        values.append((math.exp(u) + math.exp(1.0 - u)) / 2.0)
    return mean(values)


def estimate_control_c(rng, n_pilot):
    us = [rng.random() for _ in range(n_pilot)]
    xs = [math.exp(u) for u in us]
    return -covariance(xs, us) / sample_var(us)


def control_variate_integral_replication(rng, n, c):
    values = []
    for _ in range(n):
        u = rng.random()
        values.append(math.exp(u) + c * (u - 0.5))
    return mean(values)


def stratified_integral_replication(rng, n, strata):
    per_stratum = n // strata
    values = []
    for h in range(strata):
        for _ in range(per_stratum):
            u = (h + rng.random()) / strata
            values.append(math.exp(u))
    return mean(values)


def part1_to_4_integral():
    exact = math.e - 1.0
    R = 1000
    n = 100
    strata = 10

    rng_crude = random.Random(SEED + 1)
    crude = [crude_integral_replication(rng_crude, n) for _ in range(R)]
    crude_summary = summarize_replications(crude, exact)
    crude_variance = sample_var(crude)

    rng_anti = random.Random(SEED + 2)
    anti = [antithetic_integral_replication(rng_anti, n) for _ in range(R)]
    anti_summary = summarize_replications(anti, exact)

    rng_pilot = random.Random(SEED + 30)
    c = estimate_control_c(rng_pilot, 20_000)
    rng_control = random.Random(SEED + 3)
    control = [control_variate_integral_replication(rng_control, n, c) for _ in range(R)]
    control_summary = summarize_replications(control, exact)

    rng_strata = random.Random(SEED + 4)
    stratified = [stratified_integral_replication(rng_strata, n, strata) for _ in range(R)]
    stratified_summary = summarize_replications(stratified, exact)

    rows = []
    methods = [
        ("Crude MC", crude_summary, 1.0, "100 uniforms"),
        (
            "Antithetic",
            anti_summary,
            crude_variance / sample_var(anti),
            "50 uniforms plus 50 antithetic values",
        ),
        (
            "Control variate",
            control_summary,
            crude_variance / sample_var(control),
            f"c = {c:.4f}; theoretical c is about -1.6903",
        ),
        (
            "Stratified",
            stratified_summary,
            crude_variance / sample_var(stratified),
            f"{strata} strata, 10 samples per stratum",
        ),
    ]

    for name, summary, vr, note in methods:
        rows.append(
            [
                name,
                fmt(summary["estimate"]),
                fmt(summary["sd"]),
                f"[{fmt(summary['low'])}, {fmt(summary['high'])}]",
                fmt(summary["abs_error"]),
                fmt(vr),
                note,
            ]
        )

    print(f"Exact integral value: e - 1 = {exact:.9f}")
    print()
    print_table(
        "Parts 1-4: integral estimates",
        [
            "Method",
            "Mean estimate",
            "SD of rep estimates",
            "95% CI",
            "Abs. error",
            "VR vs crude",
            "Notes",
        ],
        rows,
    )

    return {
        "exact": exact,
        "crude_variance": crude_variance,
        "control_c": c,
    }


# parts 5-6: blocking system from exercise 4


def exp_from_uniform(u, rate):
    return -math.log(1.0 - u) / rate


def simulate_blocking_from_streams(interarrivals, service_times, m=10):
    clock = 0.0
    busy = 0
    blocked = 0
    departures = []

    for interarrival, service_time in zip(interarrivals, service_times):
        clock += interarrival

        while departures and departures[0] <= clock:
            heapq.heappop(departures)
            busy -= 1

        if busy < m:
            busy += 1
            heapq.heappush(departures, clock + service_time)
        else:
            blocked += 1

    return blocked / len(interarrivals)


def simulate_blocking_with_controls(rng, n_arrivals=10_000, m=10):
    clock = 0.0
    busy = 0
    blocked = 0
    departures = []
    interarrival_sum = 0.0
    service_sum = 0.0

    for _ in range(n_arrivals):
        interarrival = rng.expovariate(1.0)
        service_time = rng.expovariate(1.0 / 8.0)
        interarrival_sum += interarrival
        service_sum += service_time

        clock += interarrival

        while departures and departures[0] <= clock:
            heapq.heappop(departures)
            busy -= 1

        if busy < m:
            busy += 1
            heapq.heappush(departures, clock + service_time)
        else:
            blocked += 1

    return blocked / n_arrivals, interarrival_sum / n_arrivals, service_sum / n_arrivals


def apply_control_variate(xs, zs, expected_z):
    c = -covariance(xs, zs) / sample_var(zs)
    ys = [x + c * (z - expected_z) for x, z in zip(xs, zs)]
    return c, ys


def part5_blocking_control_variate():
    R = 200
    rng = random.Random(SEED + 5)
    records = [simulate_blocking_with_controls(rng) for _ in range(R)]
    xs = [r[0] for r in records]
    avg_interarrivals = [r[1] for r in records]
    avg_services = [r[2] for r in records]

    crude = summarize_replications(xs)
    crude_variance = sample_var(xs)

    controls = [
        ("Average interarrival time", avg_interarrivals, 1.0),
        ("Average offered service time", avg_services, 8.0),
    ]

    rows = [
        [
            "Crude",
            "",
            fmt(crude["estimate"]),
            f"[{fmt(crude['low'])}, {fmt(crude['high'])}]",
            fmt(crude["sd"]),
            "",
            "",
            "1.000000",
        ]
    ]

    for name, zs, expected_z in controls:
        c, ys = apply_control_variate(xs, zs, expected_z)
        summary = summarize_replications(ys)
        vr = crude_variance / sample_var(ys)
        rows.append(
            [
                "Control variate",
                name,
                fmt(summary["estimate"]),
                f"[{fmt(summary['low'])}, {fmt(summary['high'])}]",
                fmt(summary["sd"]),
                fmt(c),
                fmt(correlation(xs, zs)),
                fmt(vr),
            ]
        )

    print_table(
        "Part 5: control variates for blocking simulation",
        [
            "Estimator",
            "Control Z",
            "Mean",
            "95% CI",
            "SD of reps",
            "c",
            "corr(X,Z)",
            "VR",
        ],
        rows,
    )


def poisson_interarrivals(rng, n):
    return [rng.expovariate(1.0) for _ in range(n)]


def hyperexp_interarrivals(rng, n):
    values = []
    for _ in range(n):
        if rng.random() < 0.8:
            values.append(rng.expovariate(0.8333))
        else:
            values.append(rng.expovariate(5.0))
    return values


def exponential_services(rng, n):
    return [rng.expovariate(1.0 / 8.0) for _ in range(n)]


def paired_poisson_hyper_streams(rng, n):
    poisson = []
    hyper = []
    services = []
    for _ in range(n):
        u_time = rng.random()
        u_mix = rng.random()
        u_service = rng.random()

        poisson.append(exp_from_uniform(u_time, 1.0))
        if u_mix < 0.8:
            hyper.append(exp_from_uniform(u_time, 0.8333))
        else:
            hyper.append(exp_from_uniform(u_time, 5.0))
        services.append(exp_from_uniform(u_service, 1.0 / 8.0))

    return poisson, hyper, services


def part6_common_random_numbers():
    R = 200
    n = 10_000

    rng_ind = random.Random(SEED + 6)
    independent_diffs = []
    for _ in range(R):
        p_inter = poisson_interarrivals(rng_ind, n)
        p_service = exponential_services(rng_ind, n)
        h_inter = hyperexp_interarrivals(rng_ind, n)
        h_service = exponential_services(rng_ind, n)

        p_blocked = simulate_blocking_from_streams(p_inter, p_service)
        h_blocked = simulate_blocking_from_streams(h_inter, h_service)
        independent_diffs.append(h_blocked - p_blocked)

    rng_crn = random.Random(SEED + 7)
    crn_diffs = []
    for _ in range(R):
        p_inter, h_inter, services = paired_poisson_hyper_streams(rng_crn, n)
        p_blocked = simulate_blocking_from_streams(p_inter, services)
        h_blocked = simulate_blocking_from_streams(h_inter, services)
        crn_diffs.append(h_blocked - p_blocked)

    ind = summarize_replications(independent_diffs)
    crn = summarize_replications(crn_diffs)
    vr = sample_var(independent_diffs) / sample_var(crn_diffs)

    rows = [
        [
            "Independent random numbers",
            fmt(ind["estimate"]),
            fmt(ind["sd"]),
            f"[{fmt(ind['low'])}, {fmt(ind['high'])}]",
            "1.000000",
        ],
        [
            "Common random numbers",
            fmt(crn["estimate"]),
            fmt(crn["sd"]),
            f"[{fmt(crn['low'])}, {fmt(crn['high'])}]",
            fmt(vr),
        ],
    ]

    print_table(
        "Part 6: CRN comparison, hyperexponential minus Poisson",
        ["Experiment", "Mean D", "SD of D", "95% CI", "VR"],
        rows,
    )


# part 7: importance sampling for normal tail probs


def normal_sf(a):
    return 0.5 * math.erfc(a / math.sqrt(2.0))


def crude_normal_tail(rng, a, n):
    values = [1.0 if rng.gauss(0.0, 1.0) > a else 0.0 for _ in range(n)]
    estimate = mean(values)
    sd = sample_sd(values)
    low, high = ci(estimate, sd, n)
    return estimate, sd, low, high, sample_var(values)


def is_normal_tail(rng, a, n, sigma2):
    sigma = math.sqrt(sigma2)
    values = []
    for _ in range(n):
        y = rng.gauss(a, sigma)
        likelihood_ratio = sigma * math.exp(
            -0.5 * y * y + ((y - a) ** 2) / (2.0 * sigma2)
        )
        values.append((1.0 if y > a else 0.0) * likelihood_ratio)

    estimate = mean(values)
    sd = sample_sd(values)
    low, high = ci(estimate, sd, n)
    return estimate, sd, low, high, sample_var(values)


def part7_normal_importance_sampling():
    rng = random.Random(SEED + 8)
    rows = []
    sample_sizes = [1_000, 10_000, 100_000]
    sigma2_values = [0.5, 1.0, 2.0]

    for a in [2.0, 4.0]:
        exact = normal_sf(a)
        crude_by_n = {}

        for n in sample_sizes:
            est, sd, low, high, obs_var = crude_normal_tail(rng, a, n)
            crude_by_n[n] = obs_var
            rows.append(
                [
                    fmt(a, 1),
                    str(n),
                    "Crude MC",
                    "",
                    fmt(exact),
                    fmt(est),
                    f"[{fmt(low)}, {fmt(high)}]",
                    fmt(sd),
                    "1.000000",
                ]
            )

            exact_crude_var = exact * (1.0 - exact)
            for sigma2 in sigma2_values:
                est, sd, low, high, obs_var = is_normal_tail(rng, a, n, sigma2)
                vr = exact_crude_var / obs_var if obs_var > 0 else math.inf
                rows.append(
                    [
                        fmt(a, 1),
                        str(n),
                        "Importance sampling",
                        fmt(sigma2, 1),
                        fmt(exact),
                        fmt(est),
                        f"[{fmt(low)}, {fmt(high)}]",
                        fmt(sd),
                        fmt(vr),
                    ]
                )

    print_table(
        "Part 7: normal tail probability",
        ["a", "n", "Method", "sigma^2", "Exact", "Estimate", "95% CI", "Obs. SD", "VR"],
        rows,
    )


# part 8: importance sampling for the truncated-exp integral


def truncated_exp_sample(rng, lam):
    if abs(lam) < 1e-12:
        return rng.random()
    u = rng.random()
    return -math.log(1.0 - u * (1.0 - math.exp(-lam))) / lam


def truncated_exp_density(x, lam):
    if abs(lam) < 1e-12:
        return 1.0
    return lam * math.exp(-lam * x) / (1.0 - math.exp(-lam))


def is_integral_truncated_exp(rng, lam, n):
    values = []
    for _ in range(n):
        y = truncated_exp_sample(rng, lam)
        g = truncated_exp_density(y, lam)
        values.append(math.exp(y) / g)

    estimate = mean(values)
    sd = sample_sd(values)
    low, high = ci(estimate, sd, n)
    return estimate, sd, low, high, sample_var(values)


def part8_integral_importance_sampling():
    rng = random.Random(SEED + 9)
    exact = math.e - 1.0
    n = 100_000

    crude_obs_var = (math.e**2 - 1.0) / 2.0 - exact**2
    lambdas = [-2.0, -1.0, -0.5, 0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    rows = []
    best = None

    for lam in lambdas:
        est, sd, low, high, obs_var = is_integral_truncated_exp(rng, lam, n)
        vr = crude_obs_var / obs_var if obs_var > 1e-30 else math.inf
        if best is None or obs_var < best["obs_var"]:
            best = {"lambda": lam, "obs_var": obs_var, "estimate": est}

        rows.append(
            [
                fmt(lam, 1),
                fmt(est),
                f"[{fmt(low)}, {fmt(high)}]",
                fmt(sd),
                fmt(obs_var),
                fmt(vr),
                fmt(abs(est - exact)),
            ]
        )

    print_table(
        "Part 8: importance sampling for the integral",
        [
            "lambda",
            "Estimate",
            "95% CI",
            "Obs. SD",
            "Var(weight)",
            "VR vs crude",
            "Abs. error",
        ],
        rows,
    )
    print(
        f"Best lambda in this run: {best['lambda']:.1f}. "
        "The value lambda = -1 makes the truncated exponential density "
        "proportional to exp(x), so the weighted observations are almost constant."
    )
    print()


def main():
    print("Exercise 5 - variance reduction")
    print("================================")
    print(f"Random seed: {SEED}")
    print()

    part1_to_4_integral()
    part5_blocking_control_variate()
    part6_common_random_numbers()
    part7_normal_importance_sampling()
    part8_integral_importance_sampling()




if __name__ == "__main__":
    main()
