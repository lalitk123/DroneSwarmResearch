"""Bio-inspired stigmergy adapter (digital pheromones).

Ported from swarm_sim/stigmergy.py::StigmergyStrategy. Drones coordinate purely
through a shared pheromone field: each deposits repellent around its current cell,
the field evaporates, and each steers toward the lowest-pheromone reachable free
cell in a local window (with outward bias + tie-break). NO inter-drone messages are
exchanged, so this is essentially immune to communication failure — the counterpoint
to the centralized baseline.

ROS2 note: the shared field is the one piece of genuinely *shared* state. In the
co-simulation form it is a single numpy array. In a true distributed/hardware
deployment, "stigmergy" would be realised either as a small shared map service
(each drone writes its deposits, reads the field) or as physical/virtual markers in
the environment — the ``send`` callback is intentionally never used, matching the
comms-free property of the original.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import StrategyAdapter

FREE = 0


class StigmergyAdapter(StrategyAdapter):
    name = "stigmergy"

    def __init__(self, deposit: float = 5.0, deposit_radius: int = 2,
                 evaporation: float = 0.985, window: int = 6,
                 outward_bias: float = 0.15, obstacle_value: float = 1e6,
                 seed: Optional[int] = None, **kwargs):
        self.deposit = float(deposit)
        self.deposit_radius = int(deposit_radius)
        self.evaporation = float(evaporation)
        self.window = int(window)
        self.outward_bias = float(outward_bias)
        self.obstacle_value = float(obstacle_value)
        self.rng = np.random.default_rng(seed)
        self.field = None
        self.free_mask = None
        self.start = None

    def reset(self, grid, drones, base_pos):
        self.field = np.zeros(grid.shape, dtype=float)
        self.free_mask = (grid == FREE)
        self.field[~self.free_mask] = self.obstacle_value
        living = [d for d in drones if d.alive]
        if living:
            self.start = np.mean([d.pos for d in living], axis=0)
        else:
            self.start = np.array(base_pos, dtype=float)

    def step(self, grid, drones, base_pos, send, t):
        # `send` is intentionally never called — coordination is comms-free.
        h, w = self.field.shape
        living = [d for d in drones if d.alive]
        if not living:
            return []

        # (b) evaporate; keep obstacles pinned high.
        self.field *= self.evaporation
        self.field[~self.free_mask] = self.obstacle_value

        # (a) deposit repellent around each drone.
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
            free_block = self.free_mask[ylo:yhi, xlo:xhi]
            block[free_block] += add[free_block]

        # (c) steer toward lowest-pheromone reachable free cell in a window.
        for d in living:
            cy, cx = d.cell()
            ylo, yhi = max(0, cy - self.window), min(h, cy + self.window + 1)
            xlo, xhi = max(0, cx - self.window), min(w, cx + self.window + 1)
            sub = self.field[ylo:yhi, xlo:xhi]
            ys, xs = np.mgrid[ylo:yhi, xlo:xhi]

            score = sub.astype(float).copy()
            sy, sx = self.start
            radial = np.sqrt((ys - sy) ** 2 + (xs - sx) ** 2)
            score = score - self.outward_bias * radial
            score = score + self.rng.random(score.shape) * 0.5

            free_sub = self.free_mask[ylo:yhi, xlo:xhi]
            score[~free_sub] = np.inf
            here = (ys == cy) & (xs == cx)
            score[here] = np.inf

            if not np.isfinite(score).any():
                continue
            flat = int(np.argmin(score))
            ty, tx = int(ys.flat[flat]), int(xs.flat[flat])
            d.target = (ty, tx)
        return []  # no messages
