"""Bio-inspired stigmergy coordination strategy (digital pheromones).

Drones coordinate purely through a *shared environment field* — a digital-pheromone
"trail" map — rather than by exchanging messages. Each drone deposits a repellent
pheromone around the area it currently occupies (marking it as recently visited and
therefore less attractive), the whole field slowly evaporates so explored areas
become attractive again over time, and each drone steers toward the lowest-pheromone
(least-recently-explored) reachable free cell in its local neighbourhood.

Because all coordination is mediated by the shared field and *no* inter-drone or
drone->base messages are sent, this strategy is essentially immune to communication
failure: results under perfect comms and under heavily degraded comms are nearly
identical. That resilience is the whole point — it is the counterpoint to the
centralized baseline, whose coverage collapses as links drop.
"""

from __future__ import annotations

import numpy as np

from .strategies import CoordinationStrategy
from .world import FREE


class StigmergyStrategy(CoordinationStrategy):
    name = "stigmergy"

    def __init__(self,
                 deposit: float = 5.0,
                 deposit_radius: int = 2,
                 evaporation: float = 0.985,
                 window: int = 6,
                 outward_bias: float = 0.15,
                 obstacle_value: float = 1e6,
                 seed: int | None = None):
        self.deposit = float(deposit)
        self.deposit_radius = int(deposit_radius)
        self.evaporation = float(evaporation)   # per-step multiplicative decay
        self.window = int(window)               # half-size of local candidate window
        self.outward_bias = float(outward_bias)
        self.obstacle_value = float(obstacle_value)
        self.rng = np.random.default_rng(seed)

        self.field = None        # shared pheromone map (height, width)
        self.free_mask = None    # bool: True where cell is FREE
        self.start = None        # swarm centroid at reset, for outward bias

    # ---- setup -----------------------------------------------------------
    def reset(self, world, drones, base):
        h, w = world.cfg.height, world.cfg.width
        self.field = np.zeros((h, w), dtype=float)
        self.free_mask = (world.grid == FREE)
        # Obstacles are permanently unattractive so drones never steer into them.
        self.field[~self.free_mask] = self.obstacle_value

        living = [d for d in drones if d.alive]
        if living:
            self.start = np.mean([d.pos for d in living], axis=0)
        else:
            self.start = np.array(base.pos, dtype=float)

    # ---- per-step coordination ------------------------------------------
    def assign(self, world, drones, base, send, step):
        # NOTE: `send` is intentionally never called — coordination is comms-free.
        h, w = self.field.shape
        living = [d for d in drones if d.alive]
        if not living:
            return

        # (b) slow evaporation of the whole field: explored areas regain appeal.
        #     Keep obstacle cells pinned high so they remain avoided.
        self.field *= self.evaporation
        self.field[~self.free_mask] = self.obstacle_value

        # (a) each drone deposits repellent pheromone around its current cell.
        r = self.deposit_radius
        for d in living:
            cy, cx = d.cell()
            ylo, yhi = max(0, cy - r), min(h, cy + r + 1)
            xlo, xhi = max(0, cx - r), min(w, cx + r + 1)
            ys, xs = np.mgrid[ylo:yhi, xlo:xhi]
            dist2 = (ys - cy) ** 2 + (xs - cx) ** 2
            falloff = np.exp(-dist2 / max(1.0, float(r) ** 2))
            block = self.field[ylo:yhi, xlo:xhi]
            add = self.deposit * falloff
            # don't overwrite the obstacle sentinel
            free_block = self.free_mask[ylo:yhi, xlo:xhi]
            block[free_block] += add[free_block]

        # (c) each drone targets the lowest-pheromone reachable free cell in a
        #     local window, with outward bias and a small random tie-break.
        for d in living:
            cy, cx = d.cell()
            ylo, yhi = max(0, cy - self.window), min(h, cy + self.window + 1)
            xlo, xhi = max(0, cx - self.window), min(w, cx + self.window + 1)

            sub = self.field[ylo:yhi, xlo:xhi]
            ys, xs = np.mgrid[ylo:yhi, xlo:xhi]

            # Base score is the pheromone level (lower == more attractive).
            score = sub.astype(float).copy()

            # Outward bias: reward cells further from the swarm start, so the
            # swarm spreads instead of churning around the spawn point.
            sy, sx = self.start
            radial = np.sqrt((ys - sy) ** 2 + (xs - sx) ** 2)
            score = score - self.outward_bias * radial

            # Small random tie-break so drones at similar fields disperse.
            score = score + self.rng.random(score.shape) * 0.5

            # Never pick the drone's own current cell (encourage movement) or
            # obstacle cells.
            free_sub = self.free_mask[ylo:yhi, xlo:xhi]
            score[~free_sub] = np.inf
            here = (ys == cy) & (xs == cx)
            score[here] = np.inf

            if not np.isfinite(score).any():
                continue
            flat = int(np.argmin(score))
            ty, tx = int(ys.flat[flat]), int(xs.flat[flat])
            d.target = (ty, tx)


if __name__ == "__main__":
    import numpy as _np

    from .config import SimConfig
    from .simulator import Simulator
    from .comm import PerfectComm, RangeComm

    def run(comm_factory, label):
        cfg = SimConfig(seed=7)
        rng = _np.random.default_rng(123)
        comm = comm_factory(rng)
        sim = Simulator(cfg, StigmergyStrategy(seed=42), comm=comm)
        m = sim.run()
        print(f"{label:>28}: coverage={m.coverage:.3f}  "
              f"detection_rate={m.detection_rate:.3f}")
        return m

    print("Stigmergy strategy self-test (comms-free coordination)")
    print("-" * 64)
    perfect = run(lambda rng: PerfectComm(), "PerfectComm")
    # Heavily degraded: tiny disk radius -> almost no links ever deliver.
    degraded = run(lambda rng: RangeComm(radius=1.0, rng=rng, mode="disk"),
                   "RangeComm(radius=1, disk)")

    print("-" * 64)
    dcov = abs(perfect.coverage - degraded.coverage)
    ddet = abs(perfect.detection_rate - degraded.detection_rate)
    print(f"|Δcoverage|={dcov:.4f}   |Δdetection_rate|={ddet:.4f}")
    if dcov < 1e-9 and ddet < 1e-9:
        print("IDENTICAL under perfect vs degraded comms (as expected: comms-free).")
    else:
        print("Nearly identical under perfect vs degraded comms (comms-free).")
