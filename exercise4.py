import heapq
import math
import os
import random
import statistics

try:
    import numpy as np
except ImportError:
    np = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


PLOT_DIR = "pics"
SEED = 12345
SERVER_COUNT = 10
ARRIVALS_PER_REPLICATION = 10_000
REPLICATIONS = 10
MEAN_INTERARRIVAL = 1.0
MEAN_SERVICE = 8.0
T_VALUE_95_10_REPS = 2.262


def erlang_b(m, A):
    top = A**m / math.factorial(m)
    bottom = sum(A**i / math.factorial(i) for i in range(m + 1))
    return top / bottom


def ci_95(values):
    if np is not None:
        mean_value = float(np.mean(values))
        sd_value = float(np.std(values, ddof=1))
    else:
        mean_value = statistics.mean(values)
        sd_value = statistics.stdev(values)
    half_width = T_VALUE_95_10_REPS * sd_value / math.sqrt(len(values))
    return mean_value, mean_value - half_width, mean_value + half_width


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def exp_mean(mean):
    return random.expovariate(1.0 / mean)


def erlang_arrival(mean=1.0, k=2):
    # erlang arrivals with the right mean
    rate = k / mean
    total = 0.0
    for _ in range(k):
        total += random.expovariate(rate)
    return total


def hyperexp_arrival():
    # mixture from the exercise
    if random.random() < 0.8:
        return random.expovariate(0.8333)
    return random.expovariate(5.0)


def pareto_service(mean=8.0, k=2.05):
    xm = mean * (k - 1.0) / k
    u = random.random()
    return xm / (u ** (1.0 / k))


def simulate(m, n_arrivals, next_interarrival, next_service):
    clock = 0.0
    busy = 0
    blocked = 0
    departures = []

    for _ in range(n_arrivals):
        clock += next_interarrival()

        while departures and departures[0] <= clock:
            heapq.heappop(departures)
            busy -= 1

        if busy < m:
            busy += 1
            heapq.heappush(departures, clock + next_service())
        else:
            blocked += 1

    return blocked / n_arrivals


def run_case(case):
    reps = []
    for _ in range(REPLICATIONS):
        reps.append(
            simulate(
                m=SERVER_COUNT,
                n_arrivals=ARRIVALS_PER_REPLICATION,
                next_interarrival=case["arrival_fn"],
                next_service=case["service_fn"],
            )
        )

    mean, low, high = ci_95(reps)
    return {
        "case": case["case"],
        "arrival": case["arrival"],
        "service": case["service"],
        "notes": case["notes"],
        "reps": reps,
        "mean": mean,
        "low": low,
        "high": high,
    }


def print_case(result):
    print(result["case"])
    print("-" * len(result["case"]))
    print("replications:", " ".join(f"{x:.5f}" for x in result["reps"]))
    print(f"mean blocked fraction: {result['mean']:.6f}")
    print(f"95% CI: [{result['low']:.6f}, {result['high']:.6f}]")
    print()


def print_table(results):
    headers = [
        "Case",
        "Arrival distribution",
        "Service distribution",
        "Mean blocked fraction",
        "95% confidence interval",
        "Notes",
    ]

    rows = []
    for r in results:
        rows.append(
            [
                r["case"],
                r["arrival"],
                r["service"],
                f"{r['mean']:.6f}",
                f"[{r['low']:.6f}, {r['high']:.6f}]",
                r["notes"],
            ]
        )

    widths = []
    for i, header in enumerate(headers):
        widths.append(max(len(header), max(len(row[i]) for row in rows)))

    print("Final comparison table")
    print("----------------------")
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(row))))


def save_blocking_plot(results, exact_reference=None):
    if plt is None:
        print("matplotlib not available, skipping plot.")
        return

    ensure_dir(PLOT_DIR)

    labels = [r["case"] for r in results]
    means = [r["mean"] for r in results]
    lower_errors = [r["mean"] - r["low"] for r in results]
    upper_errors = [r["high"] - r["mean"] for r in results]
    x_positions = list(range(len(results)))

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(x_positions, means, color="#3b6ea8", alpha=0.9)
    ax.errorbar(
        x_positions,
        means,
        yerr=[lower_errors, upper_errors],
        fmt="none",
        ecolor="#222222",
        capsize=4,
        linewidth=1.2,
    )
    if exact_reference is not None:
        ax.axhline(
            exact_reference,
            color="#a23b3b",
            linestyle="--",
            linewidth=1.3,
            label="Erlang-B reference",
        )
        ax.legend(frameon=False)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("Case")
    ax.set_ylabel("Blocked fraction")
    ax.set_title("Exercise 4: blocking probability by case")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, "ex4_blocking_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main():
    random.seed(SEED)

    m = SERVER_COUNT
    mean_interarrival = MEAN_INTERARRIVAL
    mean_service = MEAN_SERVICE
    A = mean_service / mean_interarrival
    exact = erlang_b(m, A)

    print("Exact Erlang-B check")
    print("--------------------")
    print(f"m = {m}")
    print(f"A = {A:.1f}")
    print(f"Erlang-B blocking probability = {exact:.6f}")
    print()

    cases = [
        {
            "case": "Part 1",
            "arrival": "Poisson / exponential interarrival, mean 1",
            "service": "Exponential, mean 8",
            "notes": f"baseline; exact Erlang-B = {exact:.6f}",
            "arrival_fn": lambda: exp_mean(MEAN_INTERARRIVAL),
            "service_fn": lambda: exp_mean(MEAN_SERVICE),
        },
        {
            "case": "Part 2a",
            "arrival": "Erlang interarrival, k = 2, mean 1",
            "service": "Exponential, mean 8",
            "notes": "k = 2 was used as a simple less-variable renewal process",
            "arrival_fn": lambda: erlang_arrival(mean=MEAN_INTERARRIVAL, k=2),
            "service_fn": lambda: exp_mean(MEAN_SERVICE),
        },
        {
            "case": "Part 2b",
            "arrival": "Hyperexponential, p = 0.8/0.2",
            "service": "Exponential, mean 8",
            "notes": "more variable arrivals than the Poisson case",
            "arrival_fn": hyperexp_arrival,
            "service_fn": lambda: exp_mean(MEAN_SERVICE),
        },
        {
            "case": "Part 3a",
            "arrival": "Poisson / exponential interarrival, mean 1",
            "service": "Constant, S = 8",
            "notes": "same arrival rate and same mean service time",
            "arrival_fn": lambda: exp_mean(MEAN_INTERARRIVAL),
            "service_fn": lambda: MEAN_SERVICE,
        },
        {
            "case": "Part 3b",
            "arrival": "Poisson / exponential interarrival, mean 1",
            "service": "Pareto Type I, k = 1.05, mean 8",
            "notes": "very heavy tail; finite runs converge slowly",
            "arrival_fn": lambda: exp_mean(MEAN_INTERARRIVAL),
            "service_fn": lambda: pareto_service(mean=MEAN_SERVICE, k=1.05),
        },
        {
            "case": "Part 3c",
            "arrival": "Poisson / exponential interarrival, mean 1",
            "service": "Pareto Type I, k = 2.05, mean 8",
            "notes": "heavier-tailed than exponential, but less extreme than k = 1.05",
            "arrival_fn": lambda: exp_mean(MEAN_INTERARRIVAL),
            "service_fn": lambda: pareto_service(mean=MEAN_SERVICE, k=2.05),
        },
        {
            "case": "Part 3d",
            "arrival": "Poisson / exponential interarrival, mean 1",
            "service": "Uniform(0, 16)",
            "notes": "extra service distribution with mean 8",
            "arrival_fn": lambda: exp_mean(MEAN_INTERARRIVAL),
            "service_fn": lambda: random.uniform(0.0, 2.0 * MEAN_SERVICE),
        },
        {
            "case": "Part 3e",
            "arrival": "Poisson / exponential interarrival, mean 1",
            "service": "Gamma, shape 2, scale 4",
            "notes": "extra service distribution with mean 8",
            "arrival_fn": lambda: exp_mean(MEAN_INTERARRIVAL),
            "service_fn": lambda: random.gammavariate(2.0, MEAN_SERVICE / 2.0),
        },
    ]

    results = []
    for case in cases:
        result = run_case(case)
        results.append(result)
        print_case(result)

    print_table(results)
    save_blocking_plot(results, exact)



if __name__ == "__main__":
    main()
