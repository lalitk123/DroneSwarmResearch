"""Coordination strategies.

A strategy assigns each living drone a target cell each step. It receives the world,
the drones, the base station, and a `send` callback (sender, receiver, payload) that
routes through the active CommModel and returns True if the message was (eventually)
delivered. v0 ships the centralized baseline; decentralized-consensus and
bio-inspired-stigmergy strategies plug into the same interface (next phase).
"""

from __future__ import annotations

import numpy as np

from .world import FREE


class CoordinationStrategy:
    name = "base"

    def reset(self, world, drones, base):
        pass

    def assign(self, world, drones, base, send, step):
        """Set drone.target for each living drone."""
        raise NotImplementedError


def _frontier_cells(known_covered, grid):
    """Free, uncovered cells adjacent to covered cells (the exploration frontier).

    Falls back to all uncovered free cells if no frontier (e.g. start).
    """
    free_uncovered = (grid == FREE) & (~known_covered)
    if not free_uncovered.any():
        return np.empty((0, 2), dtype=int)
    # shift covered map in 4 directions to find neighbours
    cov = known_covered
    adj = np.zeros_like(cov)
    adj[1:, :] |= cov[:-1, :]
    adj[:-1, :] |= cov[1:, :]
    adj[:, 1:] |= cov[:, :-1]
    adj[:, :-1] |= cov[:, 1:]
    frontier = free_uncovered & adj
    cells = np.argwhere(frontier)
    if len(cells) == 0:
        cells = np.argwhere(free_uncovered)
    return cells


class CentralizedStrategy(CoordinationStrategy):
    """Base station holds the global map and assigns frontier targets to drones.

    Failure mode under comms loss: a drone only receives its new assignment if the
    base->drone message is delivered. When the link fails, the drone keeps its last
    target (and, once reached, idles) — so coverage collapses as comms degrade. This
    is the intended baseline that decentralized/bio-inspired strategies must beat.
    """
    name = "centralized"

    def __init__(self):
        self.global_covered = None

    def reset(self, world, drones, base):
        self.global_covered = np.zeros((world.cfg.height, world.cfg.width), dtype=bool)

    def assign(self, world, drones, base, send, step):
        # Base aggregates coverage from drones whose uplink is delivered.
        for d in drones:
            if d.alive and send(d, base, "map"):
                self.global_covered |= d.known_covered

        cells = _frontier_cells(self.global_covered, world.grid)
        if len(cells) == 0:
            return

        living = [d for d in drones if d.alive]
        # Greedy nearest-frontier assignment, discouraging duplicate targets.
        taken = []
        for d in living:
            # base computes assignment; only applies if downlink delivers
            d2 = ((cells[:, 0] - d.pos[0]) ** 2 + (cells[:, 1] - d.pos[1]) ** 2).astype(float)
            for t in taken:
                d2 += 50.0 * np.exp(-(((cells[:, 0] - t[0]) ** 2 +
                                       (cells[:, 1] - t[1]) ** 2)) / 9.0)
            best = cells[int(np.argmin(d2))]
            taken.append(best)
            if send(base, d, "assign"):
                d.target = tuple(best)
            # else: keep previous target (the failure mode)
