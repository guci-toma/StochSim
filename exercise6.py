import math
import os
import random
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 20260614
PICS = "pics"


def mean(xs):
    return sum(xs) / len(xs)


def variance(xs):
    return statistics.variance(xs)


def quantile(xs, p):
    xs = sorted(xs)
    pos = p * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


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
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))))
    print()


# chi-square p-values, done locally in case scipy is missing


def gammq(a, x):
    if x < 0.0 or a <= 0.0:
        raise ValueError("bad arguments for incomplete gamma")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - gammp_series(a, x)
    return gammq_cf(a, x)


def gammp_series(a, x):
    gln = math.lgamma(a)
    ap = a
    summ = 1.0 / a
    delta = summ
    for _ in range(1000):
        ap += 1.0
        delta *= x / ap
        summ += delta
        if abs(delta) < abs(summ) * 1e-14:
            break
    return summ * math.exp(-x + a * math.log(x) - gln)


def gammq_cf(a, x):
    gln = math.lgamma(a)
    b = x + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi_square_pvalue(statistic, df):
    return gammq(df / 2.0, statistic / 2.0)


def chi_square_test(counts, exact_probs):
    n = sum(counts)
    stat = 0.0
    for obs, p in zip(counts, exact_probs):
        expected = n * p
        stat += (obs - expected) ** 2 / expected
    df = len(counts) - 1
    return stat, df, chi_square_pvalue(stat, df)


def total_variation(empirical, exact):
    return 0.5 * sum(abs(a - b) for a, b in zip(empirical, exact))


# plotting


def tidy_axes(ax, title, xlabel, ylabel, grid_axis="both"):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis=grid_axis, alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def downsample_pair(xs, ys, max_points):
    if len(xs) <= max_points:
        return xs, ys
    step = math.ceil(len(xs) / max_points)
    return xs[::step], ys[::step]


def save_grouped_bar_plot(labels, series_a, series_b, path, title):
    x_positions = list(range(len(labels)))
    bar_width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        [x - bar_width / 2 for x in x_positions],
        series_a,
        width=bar_width,
        label="Empirical",
        color="#3b6ea8",
    )
    ax.bar(
        [x + bar_width / 2 for x in x_positions],
        series_b,
        width=bar_width,
        label="Exact",
        color="#d76f45",
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(bottom=0.0)
    tidy_axes(ax, title, "State", "Probability", grid_axis="y")
    ax.legend(frameon=False)
    save_figure(fig, path)


def save_heatmap(states, probs, path, m=10, title="State probabilities"):
    grid = [[float("nan") for _ in range(m + 1)] for _ in range(m + 1)]
    for (i, j), prob in zip(states, probs):
        grid[j][i] = prob

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#eeeeee")

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    image = ax.imshow(
        grid,
        origin="lower",
        extent=(-0.5, m + 0.5, -0.5, m + 0.5),
        cmap=cmap,
        vmin=0.0,
        aspect="equal",
    )
    ax.set_xticks(range(m + 1))
    ax.set_yticks(range(m + 1))
    tidy_axes(ax, title, "i", "j", grid_axis="both")
    fig.colorbar(image, ax=ax, label="Probability")
    save_figure(fig, path)


def save_line_plot(values, path, title, ylabel, max_points=5000):
    xs = list(range(len(values)))
    xs, ys = downsample_pair(xs, values, max_points)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(xs, ys, color="#3b6ea8", linewidth=0.9)
    tidy_axes(ax, title, "MCMC iteration after burn-in", ylabel)
    save_figure(fig, path)


def save_histogram(values, path, title, xlabel, bins=60):
    xmin = quantile(values, 0.005)
    xmax = quantile(values, 0.995)
    if xmax <= xmin:
        xmax = xmin + 1.0

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(
        values,
        bins=bins,
        range=(xmin, xmax),
        color="#8fbdd8",
        edgecolor="white",
        linewidth=0.6,
    )
    ax.axvline(mean(values), color="#a23b3b", linewidth=1.5, label="Mean")
    tidy_axes(ax, title, xlabel, "Frequency", grid_axis="y")
    ax.legend(frameon=False)
    save_figure(fig, path)


def save_scatter(xs, ys, path, title, xlabel, ylabel, max_points=4000):
    xs, ys = downsample_pair(xs, ys, max_points)
    xmin, xmax = quantile(xs, 0.005), quantile(xs, 0.995)
    ymin, ymax = quantile(ys, 0.005), quantile(ys, 0.995)
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.scatter(xs, ys, s=9, alpha=0.35, color="#277c9f", edgecolors="none")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    tidy_axes(ax, title, xlabel, ylabel)
    save_figure(fig, path)


def save_comparison_plot(labels, values, path, title, ylabel):
    x_positions = list(range(len(values)))

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.bar(x_positions, values, width=0.62, color="#619b71")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(bottom=0.0)
    tidy_axes(ax, title, "Method", ylabel, grid_axis="y")
    save_figure(fig, path)


# parts 1 and 2: finite-state mcmc


def log_weight_truncated_poisson(i, A):
    return i * math.log(A) - math.lgamma(i + 1)


def exact_truncated_poisson(A, m):
    logs = [log_weight_truncated_poisson(i, A) for i in range(m + 1)]
    max_log = max(logs)
    weights = [math.exp(x - max_log) for x in logs]
    total = sum(weights)
    return [w / total for w in weights]


def run_part1_mh(A=8.0, m=10, total=100_000, burn=10_000):
    rng = random.Random(SEED + 1)
    current = 0
    counts = [0] * (m + 1)
    accepted = 0

    for step in range(total):
        proposal = current + (1 if rng.random() < 0.5 else -1)
        if 0 <= proposal <= m:
            log_ratio = log_weight_truncated_poisson(proposal, A) - log_weight_truncated_poisson(current, A)
            if math.log(rng.random()) < min(0.0, log_ratio):
                current = proposal
                accepted += 1
        if step >= burn:
            counts[current] += 1

    exact = exact_truncated_poisson(A, m)
    n_kept = total - burn
    empirical = [c / n_kept for c in counts]
    chi2, df, p_value = chi_square_test(counts, exact)

    rows = []
    for i in range(m + 1):
        rows.append([str(i), fmt(empirical[i]), fmt(exact[i]), fmt(n_kept * exact[i], 2), str(counts[i])])

    print_table(
        "Part 1: truncated Poisson, A = 8 and m = 10",
        ["i", "Empirical prob", "Exact prob", "Expected count", "Observed count"],
        rows,
    )
    print(f"Part 1 chi-square statistic = {chi2:.6f}")
    print(f"Part 1 degrees of freedom = {df}")
    print(f"Part 1 p-value = {p_value:.6f}")
    print(f"Part 1 acceptance rate = {accepted / total:.6f}")
    print()

    save_grouped_bar_plot(
        [str(i) for i in range(m + 1)],
        empirical,
        exact,
        os.path.join(PICS, "ex6_part1_truncated_poisson.png"),
        "Truncated Poisson probabilities",
    )

    return {
        "empirical": empirical,
        "exact": exact,
        "counts": counts,
        "chi2": chi2,
        "df": df,
        "p": p_value,
        "acceptance": accepted / total,
    }


def valid_states(m):
    return [(i, j) for i in range(m + 1) for j in range(m + 1 - i)]


def log_weight_joint(state, A1, A2):
    i, j = state
    return i * math.log(A1) - math.lgamma(i + 1) + j * math.log(A2) - math.lgamma(j + 1)


def exact_joint_probs(states, A1, A2):
    logs = [log_weight_joint(s, A1, A2) for s in states]
    max_log = max(logs)
    weights = [math.exp(x - max_log) for x in logs]
    total = sum(weights)
    return [w / total for w in weights]


def finite_sample(rng, weights):
    u = rng.random() * sum(weights)
    total = 0.0
    for idx, w in enumerate(weights):
        total += w
        if u <= total:
            return idx
    return len(weights) - 1


def run_direct_joint_mh(A1, A2, m, total, burn):
    rng = random.Random(SEED + 2)
    current = (0, 0)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    counts = {s: 0 for s in valid_states(m)}
    accepted = 0

    for step in range(total):
        di, dj = directions[rng.randrange(4)]
        proposal = (current[0] + di, current[1] + dj)
        if proposal in counts:
            log_ratio = log_weight_joint(proposal, A1, A2) - log_weight_joint(current, A1, A2)
            if math.log(rng.random()) < min(0.0, log_ratio):
                current = proposal
                accepted += 1
        if step >= burn:
            counts[current] += 1

    return counts, accepted / total


def run_coordinate_joint_mh(A1, A2, m, total, burn):
    rng = random.Random(SEED + 3)
    current = (0, 0)
    counts = {s: 0 for s in valid_states(m)}
    accepted_i = 0
    accepted_j = 0

    for step in range(total):
        i, j = current
        proposal = (i + (1 if rng.random() < 0.5 else -1), j)
        if proposal in counts:
            log_ratio = log_weight_joint(proposal, A1, A2) - log_weight_joint(current, A1, A2)
            if math.log(rng.random()) < min(0.0, log_ratio):
                current = proposal
                accepted_i += 1

        i, j = current
        proposal = (i, j + (1 if rng.random() < 0.5 else -1))
        if proposal in counts:
            log_ratio = log_weight_joint(proposal, A1, A2) - log_weight_joint(current, A1, A2)
            if math.log(rng.random()) < min(0.0, log_ratio):
                current = proposal
                accepted_j += 1

        if step >= burn:
            counts[current] += 1

    return counts, accepted_i / total, accepted_j / total


def conditional_weights(A, max_value):
    return [math.exp(i * math.log(A) - math.lgamma(i + 1)) for i in range(max_value + 1)]


def run_gibbs_joint(A1, A2, m, total, burn):
    rng = random.Random(SEED + 4)
    i, j = 0, 0
    counts = {s: 0 for s in valid_states(m)}
    i_weights = [conditional_weights(A1, max_i) for max_i in range(m + 1)]
    j_weights = [conditional_weights(A2, max_j) for max_j in range(m + 1)]

    for step in range(total):
        max_i = m - j
        i = finite_sample(rng, i_weights[max_i])
        max_j = m - i
        j = finite_sample(rng, j_weights[max_j])
        if step >= burn:
            counts[(i, j)] += 1

    return counts


def summarize_joint_method(name, states, exact, counts, n_kept, acceptance_text):
    observed = [counts[s] for s in states]
    empirical = [c / n_kept for c in observed]
    chi2, df, p_value = chi_square_test(observed, exact)
    tv = total_variation(empirical, exact)

    rows = []
    for s, emp, ex, obs in zip(states, empirical, exact, observed):
        rows.append([f"({s[0]},{s[1]})", fmt(emp), fmt(ex), fmt(n_kept * ex, 2), str(obs)])

    print_table(
        f"Part 2 state probabilities: {name}",
        ["state", "Empirical prob", "Exact prob", "Expected count", "Observed count"],
        rows,
    )

    return {
        "name": name,
        "counts": counts,
        "observed": observed,
        "empirical": empirical,
        "chi2": chi2,
        "df": df,
        "p": p_value,
        "tv": tv,
        "acceptance": acceptance_text,
    }


def run_part2_methods(A1=4.0, A2=4.0, m=10, total=200_000, burn=20_000):
    states = valid_states(m)
    exact = exact_joint_probs(states, A1, A2)
    n_kept = total - burn

    direct_counts, direct_acc = run_direct_joint_mh(A1, A2, m, total, burn)
    direct = summarize_joint_method(
        "Direct MH",
        states,
        exact,
        direct_counts,
        n_kept,
        f"{direct_acc:.6f}",
    )
    save_heatmap(
        states,
        direct["empirical"],
        os.path.join(PICS, "ex6_part2a_direct_mh_heatmap.png"),
        m,
        "Direct MH empirical probabilities",
    )

    coord_counts, acc_i, acc_j = run_coordinate_joint_mh(A1, A2, m, total, burn)
    coord = summarize_joint_method(
        "Coordinatewise MH",
        states,
        exact,
        coord_counts,
        n_kept,
        f"i={acc_i:.6f}, j={acc_j:.6f}",
    )
    save_heatmap(
        states,
        coord["empirical"],
        os.path.join(PICS, "ex6_part2b_coordinate_mh_heatmap.png"),
        m,
        "Coordinatewise MH empirical probabilities",
    )

    gibbs_counts = run_gibbs_joint(A1, A2, m, total, burn)
    gibbs = summarize_joint_method(
        "Gibbs sampling",
        states,
        exact,
        gibbs_counts,
        n_kept,
        "always accepts",
    )
    save_heatmap(
        states,
        gibbs["empirical"],
        os.path.join(PICS, "ex6_part2c_gibbs_heatmap.png"),
        m,
        "Gibbs sampler empirical probabilities",
    )

    rows = []
    for result in [direct, coord, gibbs]:
        rows.append(
            [
                result["name"],
                fmt(result["chi2"]),
                str(result["df"]),
                fmt(result["p"]),
                result["acceptance"],
                fmt(result["tv"]),
            ]
        )

    print_table(
        "Part 2 method comparison",
        ["Method", "Chi-square", "df", "p-value", "Acceptance rate", "TV distance"],
        rows,
    )
    save_comparison_plot(
        ["Direct", "Coord", "Gibbs"],
        [direct["tv"], coord["tv"], gibbs["tv"]],
        os.path.join(PICS, "ex6_part2_method_comparison.png"),
        "Part 2 total variation distance",
        "TV distance",
    )

    return [direct, coord, gibbs]


# part 3: bayesian posterior sampling


def generate_prior_sample(rng, rho=0.5):
    z1 = rng.gauss(0.0, 1.0)
    z2 = rng.gauss(0.0, 1.0)
    xi = z1
    gamma = rho * z1 + math.sqrt(1.0 - rho * rho) * z2
    theta = math.exp(xi)
    psi = math.exp(gamma)
    return xi, gamma, theta, psi


def simulate_normal_data(rng, n, theta, psi):
    sd = math.sqrt(psi)
    return [rng.gauss(theta, sd) for _ in range(n)]


def log_prior_xi_gamma(xi, gamma, rho=0.5):
    denom = 2.0 * (1.0 - rho * rho)
    return -(xi * xi - 2.0 * rho * xi * gamma + gamma * gamma) / denom


def make_log_posterior(data, rho=0.5):
    n = len(data)
    sx = sum(data)
    sx2 = sum(x * x for x in data)

    def logpost(xi, gamma):
        theta = math.exp(xi)
        psi = math.exp(gamma)
        sse = sx2 - 2.0 * theta * sx + n * theta * theta
        log_likelihood = -0.5 * n * gamma - sse / (2.0 * psi)
        return log_prior_xi_gamma(xi, gamma, rho) + log_likelihood

    return logpost


def mh_short_run(rng, logpost, start, scale, steps):
    x, y = start
    lp = logpost(x, y)
    accepted = 0
    for _ in range(steps):
        xp = x + rng.gauss(0.0, scale)
        yp = y + rng.gauss(0.0, scale)
        lpp = logpost(xp, yp)
        if math.log(rng.random()) < min(0.0, lpp - lp):
            x, y, lp = xp, yp, lpp
            accepted += 1
    return accepted / steps


def tune_scale(rng, logpost, start):
    scale = 0.35
    for _ in range(10):
        acc = mh_short_run(rng, logpost, start, scale, 2500)
        if 0.20 <= acc <= 0.50:
            return scale, acc
        if acc < 0.20:
            scale *= 0.65
        else:
            scale *= 1.35
    return scale, acc


def run_posterior_mh(rng, logpost, start, total=100_000, burn=20_000):
    scale, tune_acc = tune_scale(rng, logpost, start)
    x, y = start
    lp = logpost(x, y)
    accepted = 0
    xi_samples = []
    gamma_samples = []

    for step in range(total):
        xp = x + rng.gauss(0.0, scale)
        yp = y + rng.gauss(0.0, scale)
        lpp = logpost(xp, yp)
        if math.log(rng.random()) < min(0.0, lpp - lp):
            x, y, lp = xp, yp, lpp
            accepted += 1
        if step >= burn:
            xi_samples.append(x)
            gamma_samples.append(y)

    theta_samples = [math.exp(x) for x in xi_samples]
    psi_samples = [math.exp(y) for y in gamma_samples]
    return {
        "scale": scale,
        "tune_acceptance": tune_acc,
        "acceptance": accepted / total,
        "theta": theta_samples,
        "psi": psi_samples,
    }


def summarize_posterior(samples):
    return {
        "mean": mean(samples),
        "median": quantile(samples, 0.5),
        "low": quantile(samples, 0.025),
        "high": quantile(samples, 0.975),
    }


def run_part3_bayes():
    rng_prior = random.Random(SEED + 5)
    xi_true, gamma_true, theta_true, psi_true = generate_prior_sample(rng_prior)

    print("Part 3(a): generated prior values")
    print("---------------------------------")
    print(f"xi = {xi_true:.6f}")
    print(f"gamma = {gamma_true:.6f}")
    print(f"theta = exp(xi) = {theta_true:.6f}")
    print(f"psi = exp(gamma) = {psi_true:.6f}")
    print()

    comparison_rows = []
    data_rows = []

    for n in [10, 100, 1000]:
        rng_data = random.Random(SEED + 100 + n)
        data = simulate_normal_data(rng_data, n, theta_true, psi_true)
        sample_mean = mean(data)
        sample_var = variance(data)
        data_rows.append([str(n), fmt(sample_mean), fmt(sample_var), fmt(theta_true), fmt(psi_true)])

        logpost = make_log_posterior(data)
        start_theta = max(sample_mean, 1e-3)
        start_psi = max(sample_var, 1e-3)
        start = (math.log(start_theta), math.log(start_psi))
        rng_mcmc = random.Random(SEED + 200 + n)
        out = run_posterior_mh(rng_mcmc, logpost, start)

        theta_summary = summarize_posterior(out["theta"])
        psi_summary = summarize_posterior(out["psi"])

        print_table(
            f"Part 3 posterior summary, n = {n}",
            ["Quantity", "Mean", "Median", "2.5%", "97.5%", "True value"],
            [
                [
                    "theta",
                    fmt(theta_summary["mean"]),
                    fmt(theta_summary["median"]),
                    fmt(theta_summary["low"]),
                    fmt(theta_summary["high"]),
                    fmt(theta_true),
                ],
                [
                    "psi",
                    fmt(psi_summary["mean"]),
                    fmt(psi_summary["median"]),
                    fmt(psi_summary["low"]),
                    fmt(psi_summary["high"]),
                    fmt(psi_true),
                ],
            ],
        )
        print(f"n = {n} sample mean = {sample_mean:.6f}")
        print(f"n = {n} sample variance = {sample_var:.6f}")
        print(f"n = {n} tuned proposal scale = {out['scale']:.6f}")
        print(f"n = {n} tuning acceptance check = {out['tune_acceptance']:.6f}")
        print(f"n = {n} final acceptance rate = {out['acceptance']:.6f}")
        print()

        save_line_plot(
            out["theta"],
            os.path.join(PICS, f"ex6_part3_n{n}_theta_trace.png"),
            f"Theta trace, n = {n}",
            "theta",
        )
        save_line_plot(
            out["psi"],
            os.path.join(PICS, f"ex6_part3_n{n}_psi_trace.png"),
            f"Psi trace, n = {n}",
            "psi",
        )
        save_histogram(
            out["theta"],
            os.path.join(PICS, f"ex6_part3_n{n}_theta_hist.png"),
            f"Theta posterior, n = {n}",
            "theta",
        )
        save_histogram(
            out["psi"],
            os.path.join(PICS, f"ex6_part3_n{n}_psi_hist.png"),
            f"Psi posterior, n = {n}",
            "psi",
        )
        save_scatter(
            out["theta"],
            out["psi"],
            os.path.join(PICS, f"ex6_part3_n{n}_joint.png"),
            f"Joint posterior samples, n = {n}",
            "theta",
            "psi",
        )

        comparison_rows.append(
            [
                str(n),
                fmt(sample_mean),
                fmt(sample_var),
                fmt(theta_summary["mean"]),
                f"[{fmt(theta_summary['low'])}, {fmt(theta_summary['high'])}]",
                fmt(psi_summary["mean"]),
                f"[{fmt(psi_summary['low'])}, {fmt(psi_summary['high'])}]",
                fmt(out["acceptance"]),
            ]
        )

    print_table(
        "Part 3(b): simulated data summaries",
        ["n", "Sample mean", "Sample variance", "True theta", "True psi"],
        data_rows,
    )

    print_table(
        "Part 3 final comparison",
        [
            "n",
            "Sample mean",
            "Sample variance",
            "Post mean theta",
            "95% CI theta",
            "Post mean psi",
            "95% CI psi",
            "Acceptance",
        ],
        comparison_rows,
    )


def print_package_note():
    print("Plots are made with matplotlib and saved as PNG files.")
    print()


def main():
    os.makedirs(PICS, exist_ok=True)
    print("Exercise 6 - MCMC")
    print("=================")
    print(f"Random seed: {SEED}")
    print(f"Plots folder: {PICS}")
    print()
    print_package_note()

    run_part1_mh()
    run_part2_methods()
    run_part3_bayes()


    print("Saved plots:")
    for filename in sorted(os.listdir(PICS)):
        if filename.endswith(".png"):
            print(f"  {os.path.join(PICS, filename)}")


if __name__ == "__main__":
    main()
