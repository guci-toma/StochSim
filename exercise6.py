import importlib.util
import math
import os
import random
import statistics
import struct
import zlib


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


# small png plot helpers


class Canvas:
    def __init__(self, width=900, height=600, bg=(255, 255, 255)):
        self.width = width
        self.height = height
        self.pixels = bytearray(bg * (width * height))

    def set_pixel(self, x, y, color):
        x = int(x)
        y = int(y)
        if 0 <= x < self.width and 0 <= y < self.height:
            i = 3 * (y * self.width + x)
            self.pixels[i : i + 3] = bytes(color)

    def line(self, x0, y0, x1, y1, color, width=1):
        x0 = int(round(x0))
        y0 = int(round(y0))
        x1 = int(round(x1))
        y1 = int(round(y1))
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            for ox in range(-(width // 2), width // 2 + 1):
                for oy in range(-(width // 2), width // 2 + 1):
                    self.set_pixel(x0 + ox, y0 + oy, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def rect(self, x0, y0, x1, y1, color):
        x0, x1 = sorted((int(round(x0)), int(round(x1))))
        y0, y1 = sorted((int(round(y0)), int(round(y1))))
        x0 = max(0, min(self.width - 1, x0))
        x1 = max(0, min(self.width - 1, x1))
        y0 = max(0, min(self.height - 1, y0))
        y1 = max(0, min(self.height - 1, y1))
        for y in range(y0, y1 + 1):
            start = 3 * (y * self.width + x0)
            end = 3 * (y * self.width + x1)
            self.pixels[start : end + 3] = bytes(color) * (x1 - x0 + 1)

    def save(self, path):
        rows = []
        stride = self.width * 3
        for y in range(self.height):
            start = y * stride
            rows.append(b"\x00" + bytes(self.pixels[start : start + stride]))
        raw = b"".join(rows)

        def chunk(kind, data):
            return (
                struct.pack(">I", len(data))
                + kind
                + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
            )

        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0),
        )
        png += chunk(b"IDAT", zlib.compress(raw, 9))
        png += chunk(b"IEND", b"")
        with open(path, "wb") as f:
            f.write(png)


def mapper(xmin, xmax, ymin, ymax, width=900, height=600):
    left, right, top, bottom = 60, 30, 35, 55
    pw = width - left - right
    ph = height - top - bottom

    def f(x, y):
        px = left + (x - xmin) / (xmax - xmin) * pw
        py = height - bottom - (y - ymin) / (ymax - ymin) * ph
        return px, py

    return f, left, right, top, bottom


def save_grouped_bar_plot(series_a, series_b, path):
    canvas = Canvas()
    n = len(series_a)
    ymax = max(max(series_a), max(series_b)) * 1.15
    mp, left, right, top, bottom = mapper(-0.5, n - 0.5, 0.0, ymax)
    canvas.line(left, canvas.height - bottom, canvas.width - right, canvas.height - bottom, (80, 80, 80), 2)
    canvas.line(left, top, left, canvas.height - bottom, (80, 80, 80), 2)
    for i, (a, b) in enumerate(zip(series_a, series_b)):
        x0, y0 = mp(i - 0.35, 0.0)
        x1, y1 = mp(i - 0.05, a)
        canvas.rect(x0, y1, x1, y0, (45, 110, 190))
        x0, y0 = mp(i + 0.05, 0.0)
        x1, y1 = mp(i + 0.35, b)
        canvas.rect(x0, y1, x1, y0, (220, 90, 60))
    canvas.save(path)


def save_heatmap(states, probs, path, m=10):
    canvas = Canvas(700, 700)
    left, top = 80, 60
    cell = 52
    max_p = max(probs)
    prob_by_state = dict(zip(states, probs))
    for i in range(m + 1):
        for j in range(m + 1):
            x0 = left + i * cell
            y0 = top + (m - j) * cell
            if (i, j) in prob_by_state:
                level = int(255 * prob_by_state[(i, j)] / max_p)
                color = (255 - level // 2, 255 - level, 255)
            else:
                color = (235, 235, 235)
            canvas.rect(x0, y0, x0 + cell - 2, y0 + cell - 2, color)
            canvas.line(x0, y0, x0 + cell - 2, y0, (180, 180, 180), 1)
            canvas.line(x0, y0, x0, y0 + cell - 2, (180, 180, 180), 1)
    canvas.save(path)


def save_line_plot(values, path):
    canvas = Canvas()
    if len(values) > 5000:
        step = len(values) // 5000
        values = values[::step]
    ymin = min(values)
    ymax = max(values)
    pad = 0.08 * (ymax - ymin) if ymax > ymin else 1.0
    mp, left, right, top, bottom = mapper(0, len(values) - 1, ymin - pad, ymax + pad)
    canvas.line(left, canvas.height - bottom, canvas.width - right, canvas.height - bottom, (80, 80, 80), 2)
    canvas.line(left, top, left, canvas.height - bottom, (80, 80, 80), 2)
    prev = None
    for i, y in enumerate(values):
        px, py = mp(i, y)
        if prev:
            canvas.line(prev[0], prev[1], px, py, (45, 110, 190), 1)
        prev = (px, py)
    canvas.save(path)


def save_histogram(values, path, bins=60):
    canvas = Canvas()
    xmin = quantile(values, 0.005)
    xmax = quantile(values, 0.995)
    if xmax <= xmin:
        xmax = xmin + 1.0
    width = (xmax - xmin) / bins
    counts = [0] * bins
    for x in values:
        if xmin <= x <= xmax:
            idx = min(bins - 1, int((x - xmin) / width))
            counts[idx] += 1
    ymax = max(counts) * 1.15
    mp, left, right, top, bottom = mapper(xmin, xmax, 0.0, ymax)
    canvas.line(left, canvas.height - bottom, canvas.width - right, canvas.height - bottom, (80, 80, 80), 2)
    canvas.line(left, top, left, canvas.height - bottom, (80, 80, 80), 2)
    for i, count in enumerate(counts):
        x0 = xmin + i * width
        x1 = x0 + width
        px0, py0 = mp(x0, 0.0)
        px1, py1 = mp(x1, count)
        canvas.rect(px0 + 1, py1, px1 - 1, py0, (170, 205, 235))
    canvas.save(path)


def save_scatter(xs, ys, path, max_points=4000):
    canvas = Canvas()
    if len(xs) > max_points:
        step = len(xs) // max_points
        xs = xs[::step]
        ys = ys[::step]
    xmin, xmax = quantile(xs, 0.005), quantile(xs, 0.995)
    ymin, ymax = quantile(ys, 0.005), quantile(ys, 0.995)
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    mp, left, right, top, bottom = mapper(xmin, xmax, ymin, ymax)
    canvas.line(left, canvas.height - bottom, canvas.width - right, canvas.height - bottom, (80, 80, 80), 2)
    canvas.line(left, top, left, canvas.height - bottom, (80, 80, 80), 2)
    for x, y in zip(xs, ys):
        if xmin <= x <= xmax and ymin <= y <= ymax:
            px, py = mp(x, y)
            canvas.rect(px - 1, py - 1, px + 1, py + 1, (30, 120, 160))
    canvas.save(path)


def save_comparison_plot(labels, values, path):
    canvas = Canvas()
    n = len(values)
    ymax = max(values) * 1.2 if max(values) > 0 else 1.0
    mp, left, right, top, bottom = mapper(-0.5, n - 0.5, 0.0, ymax)
    canvas.line(left, canvas.height - bottom, canvas.width - right, canvas.height - bottom, (80, 80, 80), 2)
    canvas.line(left, top, left, canvas.height - bottom, (80, 80, 80), 2)
    for i, value in enumerate(values):
        x0, y0 = mp(i - 0.3, 0.0)
        x1, y1 = mp(i + 0.3, value)
        canvas.rect(x0, y1, x1, y0, (80, 150, 95))
    canvas.save(path)


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
        empirical,
        exact,
        os.path.join(PICS, "ex6_part1_truncated_poisson.png"),
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
    save_heatmap(states, direct["empirical"], os.path.join(PICS, "ex6_part2a_direct_mh_heatmap.png"), m)

    coord_counts, acc_i, acc_j = run_coordinate_joint_mh(A1, A2, m, total, burn)
    coord = summarize_joint_method(
        "Coordinatewise MH",
        states,
        exact,
        coord_counts,
        n_kept,
        f"i={acc_i:.6f}, j={acc_j:.6f}",
    )
    save_heatmap(states, coord["empirical"], os.path.join(PICS, "ex6_part2b_coordinate_mh_heatmap.png"), m)

    gibbs_counts = run_gibbs_joint(A1, A2, m, total, burn)
    gibbs = summarize_joint_method(
        "Gibbs sampling",
        states,
        exact,
        gibbs_counts,
        n_kept,
        "always accepts",
    )
    save_heatmap(states, gibbs["empirical"], os.path.join(PICS, "ex6_part2c_gibbs_heatmap.png"), m)

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

        save_line_plot(out["theta"], os.path.join(PICS, f"ex6_part3_n{n}_theta_trace.png"))
        save_line_plot(out["psi"], os.path.join(PICS, f"ex6_part3_n{n}_psi_trace.png"))
        save_histogram(out["theta"], os.path.join(PICS, f"ex6_part3_n{n}_theta_hist.png"))
        save_histogram(out["psi"], os.path.join(PICS, f"ex6_part3_n{n}_psi_hist.png"))
        save_scatter(out["theta"], out["psi"], os.path.join(PICS, f"ex6_part3_n{n}_joint.png"))

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
    packages = ["numpy", "scipy", "matplotlib"]
    missing = [p for p in packages if importlib.util.find_spec(p) is None]
    if missing:
        print("Missing optional packages:", ", ".join(missing))
        print("using the local mcmc / chi-square / png helper code")
    else:
        print("Optional scientific packages are available, but the file runs on its own anyway.")
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
