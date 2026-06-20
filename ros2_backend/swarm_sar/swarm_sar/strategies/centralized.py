"""Centralized planner adapter (baseline).

Ported from swarm_sim/strategies.py::CentralizedStrategy. The base station holds a
global coverage belief, aggregates uplinked drone maps, and assigns nearest
frontier cells with a duplicate-target penalty. A drone only receives its new
target if the base->drone link is delivered by the comm_bridge — otherwise it
keeps its last target and (once reached) idles. This is the failure mode the
resilient strategies must beat (coverage collapses 0.97->0.30 under range loss).

In ROS2 this logic lives in ``base_station_node`` (which owns the global map);
``drone_node`` instances simply uplink their maps and apply received assignments.
The adapter is written so it can run either inside the base station node or, for a
single-process co-simulation, inside the orchestrating node.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .base import StrategyAdapter, DroneView, frontier_cells


class CentralizedAdapter(StrategyAdapter):
    name = "centralized"

    def __init__(self, **kwargs):
        self.global_covered = None

    def reset(self, grid, drones, base_pos):
        self.global_covered = np.zeros(grid.shape, dtype=bool)

    def step(self, grid, drones, base_pos, send, t):
        outgoing = []
        living = [d for d in drones if d.alive]

        # 1) Base aggregates coverage from drones whose UPLINK delivers.
        for d in living:
            if send(d.id, -1, "map", {}):              # -1 == base station
                if d.known_covered is not None:
                    self.global_covered |= d.known_covered
                outgoing.append({"sender": d.id, "receiver": -1, "kind": "map"})

        cells = frontier_cells(self.global_covered, grid)
        if len(cells) == 0:
            return outgoing

        # 2) Greedy nearest-frontier assignment with a duplicate-target penalty.
        taken = []
        for d in living:
            d2 = ((cells[:, 0] - d.pos[0]) ** 2 +
                  (cells[:, 1] - d.pos[1]) ** 2).astype(float)
            for ty, tx in taken:
                d2 += 50.0 * np.exp(-(((cells[:, 0] - ty) ** 2 +
                                       (cells[:, 1] - tx) ** 2)) / 9.0)
            best = cells[int(np.argmin(d2))]
            taken.append((best[0], best[1]))
            # 3) DOWNLINK: only apply if the base->drone message is delivered.
            if send(-1, d.id, "assign", {"target": [int(best[0]), int(best[1])]}):
                d.target = (int(best[0]), int(best[1]))
                outgoing.append({"sender": -1, "receiver": d.id, "kind": "assign",
                                 "target": [int(best[0]), int(best[1])]})
            # else: keep previous target -> the centralized failure mode.
        return outgoing
