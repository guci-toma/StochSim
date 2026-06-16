import importlib.util
import math
import os
import random
import statistics
import struct
import zlib


SEED = 20260614
PLOT_DIR = "exercise3_plots"


# for the n = 10 CI bit
T_975_DF9 = 2.262157
CHI2_025_DF9 = 2.700389
CHI2_975_DF9 = 19.022768


def mean(xs):
    return sum(xs) / len(xs)


def variance(xs):
    return statistics.variance(xs)


def rel_error(sample_value, true_value):
    return abs(sample_value - true_value) / abs(true_value)


def fmt(x, digits=6):
    if isinstance(x, str):
        return x
    if x is None:
        return ""
    if math.isnan(x):
        return "nan"
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


def nonzero_uniform(rng):
    u = rng.random()
    while u <= 0.0:
        u = rng.random()
    return u


# sampling methods


def sample_exponential_inverse(rng, n, rate):
    # just inverse transform
    return [-math.log(nonzero_uniform(rng)) / rate for _ in range(n)]


def sample_normal_box_muller(rng, n):
    # box-muller, 2 normals each time
    out = []
    while len(out) < n:
        u1 = nonzero_uniform(rng)
        u2 = rng.random()
        r = math.sqrt(-2.0 * math.log(u1))
        angle = 2.0 * math.pi * u2
        out.append(r * math.cos(angle))
        if len(out) < n:
            out.append(r * math.sin(angle))
    return out


def sample_pareto_inverse(rng, n, k, beta):
    # plain type I pareto
    return [beta * (nonzero_uniform(rng) ** (-1.0 / k)) for _ in range(n)]


def sample_lomax_inverse(rng, n, beta, k=1.0):
    # lomax / pareto II
    return [beta * (nonzero_uniform(rng) ** (-1.0 / k) - 1.0) for _ in range(n)]


def sample_lomax_composition(rng, n, beta):
    # composition version from the exercise
    out = []
    for _ in range(n):
        y = -math.log(nonzero_uniform(rng)) / beta
        x = -math.log(nonzero_uniform(rng)) / y
        out.append(x)
    return out


# cdfs and densities


def exp_cdf(x, rate):
    if x < 0.0:
        return 0.0
    return 1.0 - math.exp(-rate * x)


def exp_density(x, rate):
    if x < 0.0:
        return 0.0
    return rate * math.exp(-rate * x)


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_density(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def pareto_cdf(x, k, beta):
    if x < beta:
        return 0.0
    return 1.0 - (beta / x) ** k


def pareto_density(x, k, beta):
    if x < beta:
        return 0.0
    return k * (beta**k) / (x ** (k + 1.0))


def pareto_quantile(p, k, beta):
    return beta / ((1.0 - p) ** (1.0 / k))


def pareto_plot_label(k):
    if float(k).is_integer():
        return str(int(k))
    return str(k).replace(".", "_")


def lomax_cdf(x, beta, k=1.0):
    if x < 0.0:
        return 0.0
    return 1.0 - (1.0 + x / beta) ** (-k)


def lomax_density(x, beta, k=1.0):
    if x < 0.0:
        return 0.0
    return (k / beta) * (1.0 + x / beta) ** (-(k + 1.0))


def lomax_quantile(p, beta, k=1.0):
    return beta * ((1.0 - p) ** (-1.0 / k) - 1.0)


# ks checks


def ks_probability(z):
    if z <= 0.0:
        return 1.0
    total = 0.0
    for j in range(1, 200):
        term = (-1) ** (j - 1) * math.exp(-2.0 * j * j * z * z)
        total += term
        if abs(term) < 1e-12:
            break
    return max(0.0, min(1.0, 2.0 * total))


def ks_1sample(samples, cdf):
    xs = sorted(samples)
    n = len(xs)
    d = 0.0
    for i, x in enumerate(xs, start=1):
        fx = cdf(x)
        d_plus = i / n - fx
        d_minus = fx - (i - 1) / n
        d = max(d, d_plus, d_minus)
    z = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    return d, ks_probability(z)


def ks_2sample(xs, ys):
    xs = sorted(xs)
    ys = sorted(ys)
    n = len(xs)
    m = len(ys)
    i = 0
    j = 0
    d = 0.0

    while i < n or j < m:
        if j >= m or (i < n and xs[i] <= ys[j]):
            value = xs[i]
        else:
            value = ys[j]

        while i < n and xs[i] <= value:
            i += 1
        while j < m and ys[j] <= value:
            j += 1

        d = max(d, abs(i / n - j / m))

    en = math.sqrt(n * m / (n + m))
    z = (en + 0.12 + 0.11 / en) * d
    return d, ks_probability(z)


# small png helper so this still works without matplotlib


class Canvas:
    def __init__(self, width=900, height=600, bg=(255, 255, 255)):
        self.width = width
        self.height = height
        self.pixels = bytearray(bg * (width * height))

    def set_pixel(self, x, y, color):
        x = int(x)
        y = int(y)
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = 3 * (y * self.width + x)
            self.pixels[idx : idx + 3] = bytes(color)

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
        x0 = max(0, x0)
        x1 = min(self.width - 1, x1)
        y0 = max(0, y0)
        y1 = min(self.height - 1, y1)
        for y in range(y0, y1 + 1):
            start = 3 * (y * self.width + x0)
            end = 3 * (y * self.width + x1)
            row = bytes(color) * (x1 - x0 + 1)
            self.pixels[start : end + 3] = row

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


def make_mapper(xmin, xmax, ymin, ymax, width=900, height=600):
    left, right, top, bottom = 70, 25, 35, 55
    plot_w = width - left - right
    plot_h = height - top - bottom

    def map_xy(x, y):
        px = left + (x - xmin) / (xmax - xmin) * plot_w
        py = height - bottom - (y - ymin) / (ymax - ymin) * plot_h
        return px, py

    return map_xy, left, right, top, bottom


def save_histogram(samples, density, xmin, xmax, path, bins=60):
    canvas = Canvas()
    width = canvas.width
    height = canvas.height
    bin_width = (xmax - xmin) / bins
    counts = [0 for _ in range(bins)]
    for x in samples:
        if xmin <= x <= xmax:
            idx = min(bins - 1, int((x - xmin) / bin_width))
            counts[idx] += 1

    hist_density = [c / (len(samples) * bin_width) for c in counts]
    curve = []
    for i in range(400):
        x = xmin + (xmax - xmin) * i / 399
        curve.append((x, density(x)))

    ymax = max(max(hist_density), max(y for _, y in curve)) * 1.1
    mapper, left, right, top, bottom = make_mapper(xmin, xmax, 0.0, ymax, width, height)

    canvas.line(left, height - bottom, width - right, height - bottom, (80, 80, 80), 2)
    canvas.line(left, top, left, height - bottom, (80, 80, 80), 2)

    for i, h in enumerate(hist_density):
        x0 = xmin + i * bin_width
        x1 = x0 + bin_width
        px0, py0 = mapper(x0, 0.0)
        px1, py1 = mapper(x1, h)
        canvas.rect(px0 + 1, py1, px1 - 1, py0, (180, 205, 235))

    prev = None
    for x, y in curve:
        px, py = mapper(x, y)
        if prev is not None:
            canvas.line(prev[0], prev[1], px, py, (190, 50, 45), 2)
        prev = (px, py)

    canvas.save(path)


def save_two_histogram(samples_a, samples_b, density, xmin, xmax, path, bins=80):
    canvas = Canvas()
    width = canvas.width
    height = canvas.height
    bin_width = (xmax - xmin) / bins

    def hist_line(samples):
        counts = [0 for _ in range(bins)]
        for x in samples:
            if xmin <= x <= xmax:
                idx = min(bins - 1, int((x - xmin) / bin_width))
                counts[idx] += 1
        return [
            (xmin + (i + 0.5) * bin_width, counts[i] / (len(samples) * bin_width))
            for i in range(bins)
        ]

    line_a = hist_line(samples_a)
    line_b = hist_line(samples_b)
    curve = [(xmin + (xmax - xmin) * i / 399, 0.0) for i in range(400)]
    curve = [(x, density(x)) for x, _ in curve]
    ymax = max(
        max(y for _, y in line_a),
        max(y for _, y in line_b),
        max(y for _, y in curve),
    ) * 1.1
    mapper, left, right, top, bottom = make_mapper(xmin, xmax, 0.0, ymax, width, height)

    canvas.line(left, height - bottom, width - right, height - bottom, (80, 80, 80), 2)
    canvas.line(left, top, left, height - bottom, (80, 80, 80), 2)

    for points, color in [(line_a, (30, 105, 180)), (line_b, (40, 150, 95)), (curve, (190, 50, 45))]:
        prev = None
        for x, y in points:
            px, py = mapper(x, y)
            if prev is not None:
                canvas.line(prev[0], prev[1], px, py, color, 2)
            prev = (px, py)

    canvas.save(path)


def save_ci_plot(intervals, true_value, path):
    canvas = Canvas()
    width = canvas.width
    height = canvas.height
    lows = [lo for lo, _, _ in intervals]
    highs = [hi for _, hi, _ in intervals]
    xmin = min(min(lows), true_value)
    xmax = max(max(highs), true_value)
    pad = 0.08 * (xmax - xmin)
    xmin -= pad
    xmax += pad
    mapper, left, right, top, bottom = make_mapper(xmin, xmax, 0.0, len(intervals) + 1, width, height)

    canvas.line(left, height - bottom, width - right, height - bottom, (80, 80, 80), 2)
    canvas.line(left, top, left, height - bottom, (80, 80, 80), 2)
    x_true, _ = mapper(true_value, 0.0)
    canvas.line(x_true, top, x_true, height - bottom, (0, 0, 0), 2)

    for idx, (lo, hi, contains) in enumerate(intervals, start=1):
        px0, py = mapper(lo, idx)
        px1, _ = mapper(hi, idx)
        color = (35, 140, 85) if contains else (200, 45, 45)
        canvas.line(px0, py, px1, py, color, 2)

    canvas.save(path)


# actual exercise parts


def part1_and_part2():
    rng = random.Random(SEED + 1)
    n_main = 10_000
    n_pareto = 100_000
    beta = 1.0
    rate = 2.0

    exp_samples = sample_exponential_inverse(rng, n_main, rate)
    exp_mean = mean(exp_samples)
    exp_var = variance(exp_samples)
    exp_ks, exp_p = ks_1sample(exp_samples, lambda x: exp_cdf(x, rate))
    save_histogram(
        exp_samples,
        lambda x: exp_density(x, rate),
        0.0,
        -math.log(0.005) / rate,
        os.path.join(PLOT_DIR, "exponential_hist.png"),
    )

    normal_samples = sample_normal_box_muller(rng, n_main)
    normal_mean = mean(normal_samples)
    normal_var = variance(normal_samples)
    normal_ks, normal_p = ks_1sample(normal_samples, normal_cdf)
    save_histogram(
        normal_samples,
        normal_density,
        -4.0,
        4.0,
        os.path.join(PLOT_DIR, "normal_box_muller_hist.png"),
    )

    check_rows = [
        [
            "Exponential inverse",
            str(n_main),
            fmt(exp_mean),
            fmt(1.0 / rate),
            fmt(exp_var),
            fmt(1.0 / (rate * rate)),
            fmt(exp_ks),
            fmt(exp_p),
        ],
        [
            "Normal Box-Muller",
            str(n_main),
            fmt(normal_mean),
            fmt(0.0),
            fmt(normal_var),
            fmt(1.0),
            fmt(normal_ks),
            fmt(normal_p),
        ],
    ]

    pareto_rows = []
    for k in [2.05, 2.5, 3.0, 4.0]:
        samples = sample_pareto_inverse(rng, n_pareto, k, beta)
        sample_mean = mean(samples)
        sample_var = variance(samples)
        true_mean = beta * k / (k - 1.0)
        true_var = beta * beta * k / ((k - 1.0) ** 2 * (k - 2.0))
        ks, p_value = ks_1sample(samples, lambda x, kk=k: pareto_cdf(x, kk, beta))

        name = pareto_plot_label(k)
        save_histogram(
            samples,
            lambda x, kk=k: pareto_density(x, kk, beta),
            beta,
            pareto_quantile(0.995, k, beta),
            os.path.join(PLOT_DIR, f"pareto_k_{name}_hist.png"),
        )

        check_rows.append(
            [
                f"Pareto inverse k={k:g}",
                str(n_pareto),
                fmt(sample_mean),
                fmt(true_mean),
                fmt(sample_var),
                fmt(true_var),
                fmt(ks),
                fmt(p_value),
            ]
        )
        pareto_rows.append(
            [
                fmt(k, 2),
                fmt(sample_mean),
                fmt(true_mean),
                fmt(rel_error(sample_mean, true_mean)),
                fmt(sample_var),
                fmt(true_var),
                fmt(rel_error(sample_var, true_var)),
                fmt(ks),
                fmt(p_value),
            ]
        )

    print_table(
        "Part 1: distribution checks",
        [
            "Distribution",
            "n",
            "Sample mean",
            "Theory mean",
            "Sample var",
            "Theory var",
            "KS D",
            "KS p",
        ],
        check_rows,
    )

    print_table(
        "Part 2: Pareto moment comparison, beta = 1",
        [
            "k",
            "Sample mean",
            "Theory mean",
            "Mean rel. err.",
            "Sample var",
            "Theory var",
            "Var rel. err.",
            "KS D",
            "KS p",
        ],
        pareto_rows,
    )


def part3_ci_coverage():
    rng = random.Random(SEED + 2)
    n = 10
    true_mean = 0.0
    true_var = 1.0
    mean_intervals = []
    var_intervals = []

    for _ in range(100):
        sample = sample_normal_box_muller(rng, n)
        xbar = mean(sample)
        s2 = variance(sample)
        s = math.sqrt(s2)

        mean_low = xbar - T_975_DF9 * s / math.sqrt(n)
        mean_high = xbar + T_975_DF9 * s / math.sqrt(n)
        mean_contains = mean_low <= true_mean <= mean_high
        mean_intervals.append((mean_low, mean_high, mean_contains))

        var_low = (n - 1) * s2 / CHI2_975_DF9
        var_high = (n - 1) * s2 / CHI2_025_DF9
        var_contains = var_low <= true_var <= var_high
        var_intervals.append((var_low, var_high, var_contains))

    save_ci_plot(
        mean_intervals,
        true_mean,
        os.path.join(PLOT_DIR, "normal_mean_ci_coverage.png"),
    )
    save_ci_plot(
        var_intervals,
        true_var,
        os.path.join(PLOT_DIR, "normal_variance_ci_coverage.png"),
    )

    mean_hits = sum(1 for _, _, hit in mean_intervals if hit)
    var_hits = sum(1 for _, _, hit in var_intervals if hit)

    rows = [
        ["Mean", f"{mean_hits}/100", fmt(mean_hits / 100.0), "t interval"],
        ["Variance", f"{var_hits}/100", fmt(var_hits / 100.0), "chi-square interval"],
    ]
    print_table(
        "Part 3: normal confidence interval coverage",
        ["Parameter", "Contained true value", "Observed coverage", "Method"],
        rows,
    )


def quantile_from_sorted(xs, p):
    if not xs:
        return float("nan")
    pos = p * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    weight = pos - lo
    return xs[lo] * (1.0 - weight) + xs[hi] * weight


def part4_composition():
    rng = random.Random(SEED + 3)
    n = 100_000
    beta = 1.0

    direct = sample_lomax_inverse(rng, n, beta, k=1.0)
    composition = sample_lomax_composition(rng, n, beta)

    direct_sorted = sorted(direct)
    comp_sorted = sorted(composition)

    direct_ks, direct_p = ks_1sample(direct, lambda x: lomax_cdf(x, beta, 1.0))
    comp_ks, comp_p = ks_1sample(composition, lambda x: lomax_cdf(x, beta, 1.0))
    two_ks, two_p = ks_2sample(direct, composition)

    save_two_histogram(
        direct,
        composition,
        lambda x: lomax_density(x, beta, 1.0),
        0.0,
        lomax_quantile(0.99, beta, 1.0),
        os.path.join(PLOT_DIR, "pareto_composition_vs_inversion.png"),
    )

    rows = [
        [
            "Direct inversion",
            fmt(mean(direct)),
            fmt(quantile_from_sorted(direct_sorted, 0.50)),
            fmt(quantile_from_sorted(direct_sorted, 0.90)),
            fmt(quantile_from_sorted(direct_sorted, 0.95)),
            fmt(quantile_from_sorted(direct_sorted, 0.99)),
            fmt(direct_ks),
            fmt(direct_p),
        ],
        [
            "Composition",
            fmt(mean(composition)),
            fmt(quantile_from_sorted(comp_sorted, 0.50)),
            fmt(quantile_from_sorted(comp_sorted, 0.90)),
            fmt(quantile_from_sorted(comp_sorted, 0.95)),
            fmt(quantile_from_sorted(comp_sorted, 0.99)),
            fmt(comp_ks),
            fmt(comp_p),
        ],
        [
            "Theory",
            "infinite",
            fmt(lomax_quantile(0.50, beta, 1.0)),
            fmt(lomax_quantile(0.90, beta, 1.0)),
            fmt(lomax_quantile(0.95, beta, 1.0)),
            fmt(lomax_quantile(0.99, beta, 1.0)),
            "",
            "",
        ],
    ]

    print_table(
        "Part 4: Pareto Type II / Lomax comparison, beta = 1 and k = 1",
        ["Method", "Sample mean", "q50", "q90", "q95", "q99", "KS D", "KS p"],
        rows,
    )
    print(f"Two-sample KS, direct vs composition: D = {two_ks:.6f}, p = {two_p:.6f}")
    print()


def print_package_note():
    packages = ["numpy", "scipy", "pandas", "matplotlib"]
    available = []
    missing = []
    for package in packages:
        if importlib.util.find_spec(package) is None:
            missing.append(package)
        else:
            available.append(package)

    if available:
        print("Available optional packages:", ", ".join(available))
    if missing:
        print("Missing optional packages:", ", ".join(missing))
        print("using the fallback code for the checks and png plots")
    print()


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("Exercise 3 - sampling from continuous distributions")
    print("===================================================")
    print(f"Random seed: {SEED}")
    print(f"Plots folder: {PLOT_DIR}")
    print()

    print_package_note()
    part1_and_part2()
    part3_ci_coverage()
    part4_composition()


    print("Saved plots:")
    for filename in sorted(os.listdir(PLOT_DIR)):
        if filename.endswith(".png"):
            print(f"  {os.path.join(PLOT_DIR, filename)}")


if __name__ == "__main__":
    main()
