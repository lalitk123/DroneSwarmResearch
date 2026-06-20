"""Decentralized consensus / gossip coverage strategy.

Each drone keeps its own local coverage belief. Every step, living drones gossip
their maps to neighbours within comm range; successful links merge maps (logical
OR) and victim dicts. Targets are then chosen independently by each drone from its
own merged belief, with a soft penalty around heard neighbours' positions/targets
to deconflict. No base station is required, so the swarm degrades gracefully as
comms fail: a dropped link merely skips a merge, it never strands a drone.
"""

from __future__ import annotations

import numpy as np

from .strategies import CoordinationStrategy, _frontier_cells
from .world import FREE


class DecentralizedStrategy(CoordinationStrategy):
    name = "decentralized"

    def reset(self, world, drones, base):
        # nothing global to hold; state lives on the drones themselves
        pass

    def assign(self, world, drones, base, send, step):
        living = [d for d in drones if d.alive]
        if not living:
            return

        # --- 1) Gossip / map-merge ----------------------------------------
        # For each ordered pair of distinct living drones, attempt a link.
        # On success the receiver merges the sender's belief (OR coverage,
        # update victims). Snapshot the senders' maps first so merges within a
        # single step don't cascade (consensus is one hop per step).
        snap_cov = {d.id: d.known_covered.copy() for d in living}
        snap_vic = {d.id: dict(d.known_victims) for d in living}
        # neighbours[d.id] = list of drone objects this drone successfully heard
        neighbours = {d.id: [] for d in living}

        for sender in living:
            for receiver in living:
                if sender.id == receiver.id:
                    continue
                if send(sender, receiver, "map"):
                    receiver.known_covered |= snap_cov[sender.id]
                    for vid, pos in snap_vic[sender.id].items():
                        receiver.known_victims.setdefault(vid, pos)
                    neighbours[receiver.id].append(sender)

        # --- 2) Distributed assignment ------------------------------------
        for d in living:
            cells = _frontier_cells(d.known_covered, world.grid)
            if len(cells) == 0:
                # fully explored belief: keep last target or hold position
                if d.target is None:
                    d.target = d.cell()
                continue

            cy = cells[:, 0].astype(float)
            cx = cells[:, 1].astype(float)
            cost = (cy - d.pos[0]) ** 2 + (cx - d.pos[1]) ** 2

            # Soft deconfliction: push away from heard neighbours' positions and
            # their current targets so nearby drones don't converge on one cell.
            for nb in neighbours[d.id]:
                refs = [nb.pos]
                if nb.target is not None:
                    refs.append(np.asarray(nb.target, dtype=float))
                for ry, rx in refs:
                    cost += 50.0 * np.exp(-(((cy - ry) ** 2 + (cx - rx) ** 2)) / 9.0)

            best = cells[int(np.argmin(cost))]
            d.target = (int(best[0]), int(best[1]))


if __name__ == "__main__":
    import numpy as np
    from swarm_sim.simulator import Simulator
    from swarm_sim.config import SimConfig
    from swarm_sim.comm import RangeComm

    def run(label, comm=None):
        cfg = SimConfig(seed=1)
        sim = Simulator(cfg, DecentralizedStrategy(), comm=comm)
        m = sim.run()
        print(f"{label:>28}: coverage={m.coverage:.3f} "
              f"detection_rate={m.detection_rate:.3f} "
              f"mean_conn={m.mean_connectivity:.3f}")
        return m

    print("DecentralizedStrategy self-test")
    run("perfect comms")
    rng = np.random.default_rng(0)
    run("RangeComm disk r=4 (degraded)", RangeComm(radius=4.0, rng=rng, mode="disk"))

    # Centralized baseline under the same degraded comms for comparison.
    from swarm_sim.strategies import CentralizedStrategy
    cfg = SimConfig(seed=1)
    rng2 = np.random.default_rng(0)
    base_m = Simulator(cfg, CentralizedStrategy(),
                       comm=RangeComm(radius=4.0, rng=rng2, mode="disk")).run()
    print(f"{'centralized r=4 (baseline)':>28}: coverage={base_m.coverage:.3f} "
          f"detection_rate={base_m.detection_rate:.3f} "
          f"mean_conn={base_m.mean_connectivity:.3f}")
