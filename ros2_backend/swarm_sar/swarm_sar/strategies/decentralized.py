"""Decentralized consensus / gossip adapter.

Ported from swarm_sim/decentralized.py::DecentralizedStrategy. Each drone keeps its
own coverage belief; living drones gossip maps to neighbours each tick, and
successful links (delivered by the comm_bridge) merge beliefs (OR coverage, union
victims). Targets are chosen independently from each drone's merged belief, with a
soft penalty around heard neighbours to deconflict. No base station required, so a
dropped link merely skips a merge — it never strands a drone. This degrades
gracefully under range / dead-zone loss (the project's resilience thesis).

In a true distributed ROS2 deployment each drone_node runs only the slice of this
loop for its own DroneView, publishing one CommMessage per neighbour and merging
whatever arrives in its inbox. The co-simulation form below performs the full
all-pairs gossip in one place for an exact match to the swarm_sim baseline.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .base import StrategyAdapter, DroneView, frontier_cells


class DecentralizedAdapter(StrategyAdapter):
    name = "decentralized"

    def __init__(self, **kwargs):
        pass

    def step(self, grid, drones, base_pos, send, t):
        outgoing = []
        living = [d for d in drones if d.alive]
        if not living:
            return outgoing

        # --- 1) Gossip / map-merge (one hop per tick) ---------------------
        # Snapshot first so merges within a tick don't cascade.
        snap_cov = {d.id: (d.known_covered.copy()
                           if d.known_covered is not None else None)
                    for d in living}
        snap_vic = {d.id: dict(d.known_victims) for d in living}
        neighbours = {d.id: [] for d in living}

        for sender in living:
            for receiver in living:
                if sender.id == receiver.id:
                    continue
                # delivery decided by the comm_bridge's CommModel
                if send(sender.id, receiver.id, "map", {}):
                    if snap_cov[sender.id] is not None and receiver.known_covered is not None:
                        receiver.known_covered |= snap_cov[sender.id]
                    for vid, pos in snap_vic[sender.id].items():
                        receiver.known_victims.setdefault(vid, pos)
                    neighbours[receiver.id].append(sender)
                    outgoing.append({"sender": sender.id, "receiver": receiver.id,
                                     "kind": "map"})

        # --- 2) Distributed assignment ------------------------------------
        for d in living:
            if d.known_covered is None:
                continue
            cells = frontier_cells(d.known_covered, grid)
            if len(cells) == 0:
                if d.target is None:
                    d.target = d.cell()
                continue

            cy = cells[:, 0].astype(float)
            cx = cells[:, 1].astype(float)
            cost = (cy - d.pos[0]) ** 2 + (cx - d.pos[1]) ** 2

            # Soft deconfliction around heard neighbours and their targets.
            for nb in neighbours[d.id]:
                refs = [nb.pos]
                if nb.target is not None:
                    refs.append(np.asarray(nb.target, dtype=float))
                for ry, rx in refs:
                    cost += 50.0 * np.exp(-(((cy - ry) ** 2 + (cx - rx) ** 2)) / 9.0)

            best = cells[int(np.argmin(cost))]
            d.target = (int(best[0]), int(best[1]))
        return outgoing
