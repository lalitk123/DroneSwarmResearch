"""Multi-branch vs single-chain relay routing under a line-of-sight dead-zone WALL.

A dead-zone wall (block_crossing=True) partially bisects the map between the base (top-
left) and the search region (bottom), leaving a gap at one end. With limited comm range
the swarm cannot self-bridge: any link crossing the wall is severed, and a single straight
relay chain to the farthest searcher crosses the wall and stays broken. The multi-branch
router clusters searchers and routes each branch AROUND the wall end.

Sweep = wall length (how much of the width it blocks -> longer detour). Multi-hop ON.

Run: python -m swarm_sim.relay_multibranch_experiment
Outputs: results/relay_multibranch.csv, results/relay_multibranch.png
"""

from __future__ import annotations

import csv
import os

import numpy as np

from .config import SimConfig
from .simulator import Simulator
from .strategies import CentralizedStrategy
from .relay import RelayStrategy, MultiBranchRelayStrategy
from .comm import RangeComm, DeadZone, Composite

BASE = dict(width=50, height=50, n_obstacles=30, obstacle_clusters=6,
            n_drones=12, n_victims=15, max_steps=600, seed=500, comm_radius=18.0)
N_SEEDS = 10
N_RELAYS = 6
WALL_LENGTHS = [30, 38, 44]   # cols 0..L blocked at mid-height; gap = width - L


def _zone_rects(cfg, wall_len):
    """Horizontal wall across the middle rows, cols 0..wall_len (gap on the right)."""
    cy = cfg.height // 2
    return [(cy - 3, 0, cy + 3, int(wall_len))]


def _run(make_strat, wall_len):
    cov, det, conn = [], [], []
    for s in range(N_SEEDS):
        cfg = SimConfig(**{**BASE, "seed": BASE["seed"] + s})
        zones = _zone_rects(cfg, wall_len)
        rng = np.random.default_rng(cfg.seed + 9999)
        comm = Composite([RangeComm(radius=cfg.comm_radius, rng=rng, mode="disk"),
                          DeadZone(zones, block_crossing=True)])
        m = Simulator(cfg, make_strat(zones), comm=comm, multihop=True).run()
        cov.append(m.coverage); det.append(m.detection_rate); conn.append(m.mean_connectivity)
    return float(np.mean(cov)), float(np.mean(det)), float(np.mean(conn))


VARIANTS = {
    "centralized": lambda zones: CentralizedStrategy(),
    "single-chain relay": lambda zones: RelayStrategy(CentralizedStrategy(), n_relays=N_RELAYS),
    "multi-branch relay": lambda zones: MultiBranchRelayStrategy(
        CentralizedStrategy(), n_relays=N_RELAYS, dead_zones=zones, n_branches=2),
}


def main(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    series = {k: {"cov": [], "conn": []} for k in VARIANTS}
    for wl in WALL_LENGTHS:
        for name, fac in VARIANTS.items():
            cov, det, conn = _run(fac, wl)
            series[name]["cov"].append(cov)
            series[name]["conn"].append(conn)
            rows.append(dict(variant=name, wall_len=wl,
                             coverage=round(cov, 3), detection=round(det, 3),
                             connectivity=round(conn, 3)))

    with open(os.path.join(out_dir, "relay_multibranch.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"50x50, 12 drones, {N_RELAYS} relays, range {BASE['comm_radius']:.0f}, "
          f"block-crossing wall, multihop ON")
    print("coverage(detection) | connectivity, by wall length (cols blocked of 50)\n")
    print("wall".ljust(8) + "".join(f"{k:>26}" for k in VARIANTS))
    for i, wl in enumerate(WALL_LENGTHS):
        line = f"{wl:<8}"
        for k in VARIANTS:
            line += f"{series[k]['cov'][i]:>10.3f}        {series[k]['conn'][i]:>6.3f}"
        print(line)

    _plot(series, out_dir)
    return series


def _plot(series, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")
        return
    x = WALL_LENGTHS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for name in VARIANTS:
        ax1.plot(x, series[name]["cov"], "-o", label=name)
        ax2.plot(x, series[name]["conn"], "-o", label=name)
    for ax, ttl, yl in ((ax1, "Coverage vs wall length", "coverage"),
                        (ax2, "Connectivity vs wall length", "largest-CC fraction")):
        ax.set_xlabel("wall length (cols blocked of 50)")
        ax.set_ylabel(yl); ax.set_title(ttl)
        ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "relay_multibranch.png")
    fig.savefig(p, dpi=120)
    print(f"Saved plot: {p}")


if __name__ == "__main__":
    main()
