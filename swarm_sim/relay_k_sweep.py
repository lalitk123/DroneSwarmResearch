"""Relay-count (K) sweep under range and dead-zone comms failure.

Question: how many relay drones are needed, and where do returns diminish? Each relay
is a searcher sacrificed, so there is a connectivity-vs-search-throughput tradeoff and
an interior optimum. Fleet size is fixed at 12; K relays => (12-K) searchers.

Run: python -m swarm_sim.relay_k_sweep
Outputs: results/relay_k_sweep.csv, results/relay_k_sweep.png
"""

from __future__ import annotations

import csv
import os

import numpy as np

from .config import SimConfig
from .simulator import Simulator
from .strategies import CentralizedStrategy
from .relay import RelayStrategy
from .comm import RangeComm, DeadZone

BASE = dict(width=40, height=40, n_obstacles=25, obstacle_clusters=6,
            n_drones=12, n_victims=12, max_steps=600, seed=300, comm_radius=18.0)
N_SEEDS = 10
K_VALUES = [0, 1, 2, 3, 4, 5, 6, 7]


def _dead_zone(area_fraction, cfg):
    side = int((area_fraction * cfg.width * cfg.height) ** 0.5)
    cy, cx = cfg.height // 2, cfg.width // 2
    h = side // 2
    return DeadZone([(cy - h, cx - h, cy + h, cx + h)])


def _make_comm(scenario, level, rng, cfg):
    if scenario == "range":
        return RangeComm(radius=level, rng=rng, mode="disk")  # level = disk radius
    if scenario == "dead_zone":
        return _dead_zone(level, cfg)                          # level = area fraction
    raise ValueError(scenario)


def _run(scenario, level, k):
    cov, det, conn = [], [], []
    for s in range(N_SEEDS):
        cfg = SimConfig(**{**BASE, "seed": BASE["seed"] + s})
        rng = np.random.default_rng(cfg.seed + 9999)
        comm = _make_comm(scenario, level, rng, cfg)
        strat = (CentralizedStrategy() if k == 0
                 else RelayStrategy(CentralizedStrategy(), n_relays=k))
        m = Simulator(cfg, strat, comm=comm, multihop=True).run()
        cov.append(m.coverage); det.append(m.detection_rate)
        conn.append(m.mean_connectivity)
    return float(np.mean(cov)), float(np.mean(det)), float(np.mean(conn))


# scenario conditions: (label, scenario, level)
CONDITIONS = [
    ("range r=12", "range", 12),
    ("range r=9", "range", 9),
    ("deadzone 20%", "dead_zone", 0.20),
    ("deadzone 30%", "dead_zone", 0.30),
]


def _knee(ks, covs, tol=0.02):
    """First K beyond which coverage improves by < tol vs previous (diminishing returns).
    Returns (best_k_by_coverage, knee_k)."""
    best_k = ks[int(np.argmax(covs))]
    knee = ks[-1]
    for i in range(1, len(ks)):
        if covs[i] - covs[i - 1] < tol:
            knee = ks[i - 1]
            break
    return best_k, knee


def main(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    results = {}
    for label, scenario, level in CONDITIONS:
        covs, conns = [], []
        for k in K_VALUES:
            cov, det, conn = _run(scenario, level, k)
            covs.append(cov); conns.append(conn)
            rows.append(dict(condition=label, scenario=scenario, level=level, K=k,
                             searchers=BASE["n_drones"] - k,
                             coverage=round(cov, 3), detection=round(det, 3),
                             connectivity=round(conn, 3)))
        results[label] = (covs, conns)

    with open(os.path.join(out_dir, "relay_k_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"Fleet=12, multihop ON. Coverage by relay count K (searchers = 12-K)\n")
    header = "K".ljust(4) + "".join(f"{c[0]:>16}" for c in CONDITIONS)
    print(header)
    for i, k in enumerate(K_VALUES):
        print(f"{k:<4}" + "".join(f"{results[c[0]][0][i]:>16.3f}" for c in CONDITIONS))
    print("\nOptimum / diminishing-returns knee per condition:")
    for label, _, _ in CONDITIONS:
        covs, _ = results[label]
        best_k, knee = _knee(K_VALUES, covs)
        print(f"  {label:<16} best K={best_k} (cov {max(covs):.3f}); "
              f"knee at K={knee} (cov {covs[K_VALUES.index(knee)]:.3f})")

    _plot(results, out_dir)
    return results


def _plot(results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for label, (covs, conns) in results.items():
        ax1.plot(K_VALUES, covs, "-o", label=label)
        ax2.plot(K_VALUES, conns, "-o", label=label)
    for ax, ttl, yl in ((ax1, "Coverage vs relay count K", "coverage"),
                        (ax2, "Connectivity vs relay count K", "largest-CC fraction")):
        ax.set_xlabel("K relays (searchers = 12 - K)")
        ax.set_ylabel(yl); ax.set_title(ttl); ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "relay_k_sweep.png")
    fig.savefig(p, dpi=120)
    print(f"Saved plot: {p}")


if __name__ == "__main__":
    main()
