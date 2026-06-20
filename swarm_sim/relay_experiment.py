"""Adaptive relay-drone extension experiment.

Question: do dedicated relay drones (repositioning to bridge base<->searchers) rescue
the connectivity-dependent centralized strategy under range-limited comms?

Setup: 40x40 map, 12 drones, multi-hop comm routing ON, disk-range comm swept from
generous to severe. Compare centralized alone vs centralized + K relays (K searchers
sacrificed). Reports coverage, detection, and true largest-connected-component fraction.

Run: python -m swarm_sim.relay_experiment
Outputs: results/relay_rescue.csv, results/relay_rescue.png
"""

from __future__ import annotations

import csv
import os

import numpy as np

from .config import SimConfig
from .simulator import Simulator
from .strategies import CentralizedStrategy
from .decentralized import DecentralizedStrategy
from .relay import RelayStrategy
from .comm import RangeComm

BASE = dict(width=40, height=40, n_obstacles=25, obstacle_clusters=6,
            n_drones=12, n_victims=12, max_steps=600, seed=300, comm_radius=18.0)
N_SEEDS = 10
RADII = [18, 12, 9, 6, 4]
N_RELAYS = 4


def _run(make_strat, radius):
    cov, det, conn = [], [], []
    for s in range(N_SEEDS):
        cfg = SimConfig(**{**BASE, "seed": BASE["seed"] + s})
        rng = np.random.default_rng(cfg.seed + 9999)
        comm = RangeComm(radius=radius, rng=rng, mode="disk")
        m = Simulator(cfg, make_strat(), comm=comm, multihop=True).run()
        cov.append(m.coverage); det.append(m.detection_rate)
        conn.append(m.mean_connectivity)
    return float(np.mean(cov)), float(np.mean(det)), float(np.mean(conn))


VARIANTS = {
    "centralized": lambda: CentralizedStrategy(),
    "centralized+relay": lambda: RelayStrategy(CentralizedStrategy(), n_relays=N_RELAYS),
    "decentralized": lambda: DecentralizedStrategy(),
    "decentralized+relay": lambda: RelayStrategy(DecentralizedStrategy(), n_relays=N_RELAYS),
}


def main(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    series = {k: {"cov": [], "conn": []} for k in VARIANTS}
    for rad in RADII:
        for name, fac in VARIANTS.items():
            cov, det, conn = _run(fac, rad)
            series[name]["cov"].append(cov)
            series[name]["conn"].append(conn)
            rows.append(dict(variant=name, disk_radius=rad,
                             coverage=round(cov, 3), detection=round(det, 3),
                             connectivity=round(conn, 3)))

    with open(os.path.join(out_dir, "relay_rescue.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # print
    print(f"40x40, 12 drones, {N_RELAYS} relays, multihop ON; coverage by disk radius")
    print("radius".ljust(8) + "".join(f"{k:>22}" for k in VARIANTS))
    for i, rad in enumerate(RADII):
        print(f"{rad:<8}" + "".join(f"{series[k]['cov'][i]:>22.3f}" for k in VARIANTS))

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    x = RADII
    for name in VARIANTS:
        style = "--o" if "relay" in name else "-s"
        ax1.plot(x, series[name]["cov"], style, label=name)
        ax2.plot(x, series[name]["conn"], style, label=name)
    for ax, ttl, yl in ((ax1, "Coverage vs comm range", "coverage"),
                        (ax2, "Connectivity vs comm range", "largest-CC fraction")):
        ax.set_xlabel("disk comm radius (smaller = worse)")
        ax.set_ylabel(yl); ax.set_title(ttl)
        ax.invert_xaxis(); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "relay_rescue.png")
    fig.savefig(p, dpi=120)
    print(f"Saved plot: {p}")


if __name__ == "__main__":
    main()
