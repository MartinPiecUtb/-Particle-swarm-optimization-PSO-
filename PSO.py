from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


DIM = 10
NP = 40
MAXFES = 10000
RUNS = 10
SEED = 2026
OUT = Path(__file__).resolve().parent / "PSO_results_corrected"


def sphere(x):
    return np.sum(x**2, axis=-1)


def rastrigin(x):
    return 10 * DIM + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=-1)


FUNCTIONS = {
    "Sphere": (sphere, -100.0, 100.0),
    "Rastrigin": (rastrigin, -5.12, 5.12),
}

VARIANTS = {
    "Linear inertia, star": ("linear", "star", 2.0, 2.0),
    "Constant inertia, star": ("constant", "star", 1.49618, 1.49618),
    "Constant inertia, ring 3": ("constant", "ring", 1.49618, 1.49618),
}


def pso(function, lower, upper, variant, seed):
    inertia, topology, c1, c2 = VARIANTS[variant]
    rng = np.random.default_rng(seed)
    vmax = 0.2 * (upper - lower)

    x = rng.uniform(lower, upper, (NP, DIM))
    v = np.zeros((NP, DIM))
    pbest_x = x.copy()
    pbest_f = function(x)
    fes = NP

    best_index = np.argmin(pbest_f)
    gbest_x = pbest_x[best_index].copy()
    gbest_f = float(pbest_f[best_index])
    history_fes = [fes]
    history_best = [gbest_f]
    no_evaluation_steps = 0

    while fes < MAXFES:
        if topology == "star":
            leaders = gbest_x
        else:
            # Three neighbors on either side, plus the particle itself.
            offsets = np.arange(-3, 4)
            neighbors = np.column_stack([np.roll(pbest_f, -offset) for offset in offsets])
            best_offsets = offsets[np.argmin(neighbors, axis=1)]
            best_neighbors = (np.arange(NP) + best_offsets) % NP
            leaders = pbest_x[best_neighbors]

        if inertia == "linear":
            w = 0.8 - 0.5 * (fes - NP) / (MAXFES - NP)
        else:
            w = 0.7

        r1 = rng.random((NP, DIM))
        r2 = rng.random((NP, DIM))
        v = w * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (leaders - x)
        v = np.clip(v, -vmax, vmax)
        x = x + v

        inside = np.all((x >= lower) & (x <= upper), axis=1)
        indices = np.flatnonzero(inside)[:MAXFES - fes]

        if indices.size == 0:
            no_evaluation_steps += 1
            if no_evaluation_steps >= 5000:
                raise RuntimeError("Swarm did not return to the search area.")
            continue

        no_evaluation_steps = 0
        values = function(x[indices])
        fes += len(indices)

        improved = values < pbest_f[indices]
        updated_indices = indices[improved]
        pbest_x[updated_indices] = x[updated_indices]
        pbest_f[updated_indices] = values[improved]

        best_index = np.argmin(pbest_f)
        if pbest_f[best_index] < gbest_f:
            gbest_f = float(pbest_f[best_index])
            gbest_x = pbest_x[best_index].copy()

        history_fes.append(fes)
        history_best.append(gbest_f)

    return gbest_f, np.array(history_fes), np.array(history_best)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    grid = np.unique(np.linspace(NP, MAXFES, 250, dtype=int))

    for function_index, (name, (function, lower, upper)) in enumerate(FUNCTIONS.items()):
        plt.figure(figsize=(9, 5.5))
        for variant in VARIANTS:
            results = []
            curves = []
            for run in range(RUNS):
                seed = SEED + function_index * 1000 + run
                result, fes, best = pso(function, lower, upper, variant, seed)
                results.append(result)
                step_indices = np.maximum(np.searchsorted(fes, grid, side="right") - 1, 0)
                curves.append(best[step_indices])

            results = np.asarray(results)
            average_curve = np.mean(curves, axis=0)
            plt.semilogy(grid, np.maximum(average_curve, 1e-15), label=variant)
            row = {
                "function": name,
                "variant": variant,
                "mean": float(np.mean(results)),
                "std": float(np.std(results, ddof=1)),
                "best": float(np.min(results)),
                "worst": float(np.max(results)),
            }
            rows.append(row)
            print(f"{name:10} | {variant:25} | mean: {row['mean']:.8g} | "
                  f"std: {row['std']:.8g} | best: {row['best']:.8g}")

        plt.title(f"PSO convergence - {name} ({DIM}D, {RUNS} runs)")
        plt.xlabel("Function evaluations (FES)")
        plt.ylabel("Mean best objective value (log scale)")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / f"{name.lower()}_convergence.png", dpi=180)
        plt.close()

    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["function", "variant", "mean", "std", "best", "worst"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
