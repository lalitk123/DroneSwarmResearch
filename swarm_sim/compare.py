"""Head-to-head comparison of the three coordination strategies under comms failure.

Run: python -m swarm_sim.compare
Outputs to results/: comparison.csv, comparison_summary.json, comparison_<scenario>.png
"""

from __future__ import annotations

import json
import os

import numpy as np

from .config import SimConfig
from .strategies import CentralizedStrategy
from .decentralized import DecentralizedStrategy
from .stigmergy import StigmergyStrategy
from .experiments import sweep, aggregate, write_csv
from .metrics import resilience_auc

STRATEGIES = {
    "centralized": CentralizedStrategy,
    "decentralized": DecentralizedStrategy,
    "stigmergy": StigmergyStrategy,
}

# scenario -> severity levels (0 = no failure, higher = worse)
SCENARIOS = {
    "S1_packet_loss": [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
    "S2_range": [0.0, 0.3, 0.6, 0.8, 0.95],
}


def run(out_dir="results", n_seeds=8):
    base_cfg = SimConfig(n_drones=10, n_victims=15, max_steps=600, seed=100)
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []
    summary = {}
    for scenario, levels in SCENARIOS.items():
        summary[scenario] = {}
        for sname, factory in STRATEGIES.items():
            rows = sweep(factory, scenario, levels, n_seeds, base_cfg)
            for r in rows:
                r["strategy"] = sname
            all_rows.extend(rows)
            lv, cov_mean, _ = aggregate(rows, "coverage")
            _, det_mean, _ = aggregate(rows, "detection_rate")
            baseline = cov_mean[0] if cov_mean else 1.0
            summary[scenario][sname] = {
                "levels": lv,
                "coverage_mean": cov_mean,
                "detection_mean": det_mean,
                "coverage_resilience_auc": resilience_auc(lv, cov_mean, baseline=baseline),
            }

    write_csv(all_rows, os.path.join(out_dir, "comparison.csv"))
    with open(os.path.join(out_dir, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    _print_table(summary)
    _plot(summary, out_dir)
    return summary


def _print_table(summary):
    for scenario, by_strat in summary.items():
        print(f"\n=== {scenario} : coverage by severity ===")
        levels = next(iter(by_strat.values()))["levels"]
        header = "severity ".ljust(14) + "".join(f"{s:>14}" for s in by_strat)
        print(header)
        for i, lv in enumerate(levels):
            row = f"{lv:<14.2f}" + "".join(
                f"{by_strat[s]['coverage_mean'][i]:>14.3f}" for s in by_strat)
            print(row)
        print("resilience AUC ".ljust(14) +
              "".join(f"{by_strat[s]['coverage_resilience_auc']:>14.3f}" for s in by_strat))


def _plot(summary, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plots skipped: {e})")
        return
    for scenario, by_strat in summary.items():
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for sname, d in by_strat.items():
            ax.plot(d["levels"], d["coverage_mean"], marker="o", label=sname)
        ax.set_xlabel("failure severity")
        ax.set_ylabel("coverage fraction")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Coverage degradation — {scenario}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        p = os.path.join(out_dir, f"comparison_{scenario}.png")
        fig.savefig(p, dpi=120)
        print(f"Saved plot: {p}")


if __name__ == "__main__":
    run()
