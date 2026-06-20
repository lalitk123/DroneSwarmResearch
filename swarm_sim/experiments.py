"""Monte-Carlo experiment runner: sweeps comms-failure severity over seeds and
produces degradation curves + a resilience AUC score."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from .config import SimConfig
from .comm import PerfectComm, PacketLoss, RangeComm
from .simulator import Simulator
from .strategies import CentralizedStrategy
from .metrics import resilience_auc


def make_comm(scenario: str, level: float, rng: np.random.Generator, cfg: SimConfig):
    """Build a CommModel for a scenario at a given severity level."""
    if scenario == "S0_perfect":
        return PerfectComm()
    if scenario == "S1_packet_loss":
        return PacketLoss(p=level, rng=rng)
    if scenario == "S2_range":
        # level interpreted as fraction of nominal comm_radius (smaller = worse)
        return RangeComm(radius=max(1.0, cfg.comm_radius * (1.0 - level)), rng=rng,
                         mode="pathloss")
    raise ValueError(scenario)


def run_trials(strategy_factory: Callable[[], object], scenario: str,
               level: float, n_seeds: int, base_cfg: SimConfig):
    rows = []
    for s in range(n_seeds):
        cfg = SimConfig(**{**base_cfg.__dict__, "seed": base_cfg.seed + s})
        rng = np.random.default_rng(cfg.seed + 9999)
        comm = make_comm(scenario, level, rng, cfg)
        sim = Simulator(cfg, strategy_factory(), comm=comm)
        m = sim.run().as_dict()
        m.update(scenario=scenario, level=level, seed=cfg.seed)
        rows.append(m)
    return rows


def sweep(strategy_factory, scenario, levels: List[float], n_seeds: int,
          base_cfg: SimConfig):
    all_rows = []
    for lv in levels:
        all_rows.extend(run_trials(strategy_factory, scenario, lv, n_seeds, base_cfg))
    return all_rows


def aggregate(rows, metric="coverage"):
    """Return (levels, mean, std) for a metric, grouped by level."""
    levels = sorted({r["level"] for r in rows})
    means, stds = [], []
    for lv in levels:
        vals = [r[metric] for r in rows if r["level"] == lv]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    return levels, means, stds


def write_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def demo(out_dir="results", n_seeds=8, make_plot=True):
    """v0 demo: centralized baseline under S0 (perfect) and S1 (packet loss sweep)."""
    base_cfg = SimConfig(n_drones=10, n_victims=15, max_steps=600, seed=100)
    os.makedirs(out_dir, exist_ok=True)

    # S0 baseline (single point)
    s0 = run_trials(CentralizedStrategy, "S0_perfect", 0.0, n_seeds, base_cfg)
    # S1 packet-loss sweep
    levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    s1 = sweep(CentralizedStrategy, "S1_packet_loss", levels, n_seeds, base_cfg)

    rows = s0 + s1
    write_csv(rows, os.path.join(out_dir, "centralized_demo.csv"))

    lv, cov_mean, cov_std = aggregate(s1, "coverage")
    _, det_mean, _ = aggregate(s1, "detection_rate")
    baseline_cov = float(np.mean([r["coverage"] for r in s0]))
    auc = resilience_auc(lv, cov_mean, baseline=baseline_cov)

    summary = {
        "baseline_coverage_S0": baseline_cov,
        "packet_loss_levels": lv,
        "coverage_mean": cov_mean,
        "coverage_std": cov_std,
        "detection_rate_mean": det_mean,
        "coverage_resilience_auc": auc,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("=== Centralized baseline demo ===")
    print(f"S0 perfect-comms coverage: {baseline_cov:.3f}")
    for l, c, d in zip(lv, cov_mean, det_mean):
        print(f"  packet_loss p={l:.1f}  coverage={c:.3f}  detection={d:.3f}")
    print(f"Coverage resilience AUC (1.0=flat/ideal): {auc:.3f}")

    if make_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.errorbar(lv, cov_mean, yerr=cov_std, marker="o", label="coverage")
            ax.plot(lv, det_mean, marker="s", label="detection rate")
            ax.set_xlabel("packet-loss probability p")
            ax.set_ylabel("performance")
            ax.set_title("Centralized baseline — degradation under packet loss")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            p = os.path.join(out_dir, "degradation_curve.png")
            fig.savefig(p, dpi=120)
            print(f"Saved plot: {p}")
        except Exception as e:  # pragma: no cover
            print(f"(plot skipped: {e})")

    return summary


if __name__ == "__main__":
    demo()
