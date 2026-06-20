"""Swarm-scaling study on the 100x100 map.

3 strategies x fleet sizes {10,20,30} x {perfect, range degradation, dead zone}.
Question: how do coverage, detection, mission time, and connectivity scale with swarm
size for each strategy — and does added density let the slow-but-comms-free stigmergy
strategy catch up on the large map?

Run: python -m swarm_sim.scaling_study
Outputs: results/scaling.csv, results/scaling_<metric>.png
"""

from __future__ import annotations

import csv
import os

import numpy as np

from .config import SimConfig
from .simulator import Simulator
from .strategies import CentralizedStrategy
from .decentralized import DecentralizedStrategy
from .stigmergy import StigmergyStrategy
from .comm import PerfectComm, RangeComm, DeadZone

BASE = dict(width=100, height=100, n_obstacles=160, obstacle_clusters=20,
            n_victims=30, sensing_radius=3.0, comm_radius=15.0,
            max_steps=1000, seed=400)
N_SEEDS = 5
FLEETS = [10, 20, 30]
STRATEGIES = {
    "centralized": CentralizedStrategy,
    "decentralized": DecentralizedStrategy,
    "stigmergy": StigmergyStrategy,
}


def _dead_zone(cfg, area_fraction=0.20):
    side = int((area_fraction * cfg.width * cfg.height) ** 0.5)
    cy, cx = cfg.height // 2, cfg.width // 2
    h = side // 2
    return DeadZone([(cy - h, cx - h, cy + h, cx + h)])


def _comm(scenario, rng, cfg):
    if scenario == "perfect":
        return PerfectComm()
    if scenario == "range":
        return RangeComm(radius=cfg.comm_radius * 0.4, rng=rng, mode="pathloss")
    if scenario == "dead_zone":
        return _dead_zone(cfg, 0.20)
    raise ValueError(scenario)


SCENARIOS = ["perfect", "range", "dead_zone"]


def _run(strategy_cls, scenario, n_drones):
    cov, det, mt, conn, capped = [], [], [], [], 0
    for s in range(N_SEEDS):
        cfg = SimConfig(**{**BASE, "n_drones": n_drones, "seed": BASE["seed"] + s})
        rng = np.random.default_rng(cfg.seed + 9999)
        m = Simulator(cfg, strategy_cls(), comm=_comm(scenario, rng, cfg)).run()
        cov.append(m.coverage); det.append(m.detection_rate)
        conn.append(m.mean_connectivity)
        if m.mission_time >= 0:
            mt.append(m.mission_time)
        else:
            capped += 1
    return {
        "coverage": float(np.mean(cov)),
        "detection": float(np.mean(det)),
        "mission_time": float(np.mean(mt)) if mt else float("nan"),
        "connectivity": float(np.mean(conn)),
        "pct_capped": 100.0 * capped / N_SEEDS,
    }


def main(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    data = {}  # (scenario, strategy) -> {metric: [by fleet]}
    for scenario in SCENARIOS:
        for sname, cls in STRATEGIES.items():
            series = {k: [] for k in ("coverage", "detection", "mission_time",
                                      "connectivity", "pct_capped")}
            for n in FLEETS:
                r = _run(cls, scenario, n)
                for k in series:
                    series[k].append(r[k])
                rows.append(dict(scenario=scenario, strategy=sname, n_drones=n,
                                 **{k: round(v, 3) for k, v in r.items()}))
            data[(scenario, sname)] = series

    with open(os.path.join(out_dir, "scaling.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    _print(data)
    _plot(data, out_dir)
    return data


def _print(data):
    for scenario in SCENARIOS:
        print(f"\n=== {scenario} : coverage(detection) | mission_t | %capped, by fleet ===")
        print("fleet".ljust(7) + "".join(f"{s:>26}" for s in STRATEGIES))
        for i, n in enumerate(FLEETS):
            cells = ""
            for s in STRATEGIES:
                d = data[(scenario, s)]
                cells += (f"{d['coverage'][i]:>6.3f}({d['detection'][i]:>4.2f})"
                          f"|{d['mission_time'][i]:>5.0f}|{d['pct_capped'][i]:>3.0f}%")
            print(f"{n:<7}{cells}")


def _plot(data, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plots skipped: {e})")
        return
    for metric, yl, ylim in (("coverage", "coverage", (0, 1.05)),
                             ("detection", "detection rate", (0, 1.05)),
                             ("mission_time", "mission time (steps)", None)):
        fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(15, 4.5), sharey=True)
        for ax, scenario in zip(axes, SCENARIOS):
            for s in STRATEGIES:
                ax.plot(FLEETS, data[(scenario, s)][metric], "-o", label=s)
            ax.set_title(f"{scenario}")
            ax.set_xlabel("swarm size")
            ax.set_xticks(FLEETS)
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel(yl)
        axes[-1].legend()
        fig.suptitle(f"Scaling — {yl} (100x100, 30 victims)")
        fig.tight_layout()
        p = os.path.join(out_dir, f"scaling_{metric}.png")
        fig.savefig(p, dpi=120)
        print(f"Saved plot: {p}")


if __name__ == "__main__":
    main()
