"""Common interface + shared helpers for ROS2 strategy adapters.

These are pure-Python (numpy only) and import nothing from rclpy, so they can be
unit-tested against the same expectations as the swarm_sim originals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

FREE, OBSTACLE = 0, 1  # matches swarm_sim/world.py


@dataclass
class DroneView:
    """The ROS2 node's local model of a drone, mirroring swarm_sim/drone.py.

    The adapter mutates ``target``; the owning node turns ``target`` into motion
    and keeps ``pos``/``alive`` fresh from odometry and the node-failure model.
    """

    id: int
    pos: np.ndarray                       # (y, x) float
    alive: bool = True
    target: Optional[Tuple[int, int]] = None
    known_covered: Optional[np.ndarray] = None   # bool (H, W)
    known_victims: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    # Messages delivered to this drone this tick (filled from /swarm/comm by the
    # node after the comm_bridge has applied the CommModel).
    inbox: List[dict] = field(default_factory=list)

    def cell(self) -> Tuple[int, int]:
        return (int(round(self.pos[0])), int(round(self.pos[1])))


def frontier_cells(known_covered: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Free, uncovered cells adjacent to covered cells (exploration frontier).

    Ported verbatim from swarm_sim/strategies.py::_frontier_cells so the ROS2
    backend explores identically. Falls back to all uncovered free cells.
    """
    free_uncovered = (grid == FREE) & (~known_covered)
    if not free_uncovered.any():
        return np.empty((0, 2), dtype=int)
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


class StrategyAdapter:
    """Base class for the ROS2-side coordination strategies.

    Lifecycle (driven by the node):
      reset(grid, drones, base_pos)         once, after the world is known
      step(drones, base_pos, send, t) -> {} called on each control-loop tick;
                                            sets ``drone.target`` and returns a
                                            list of outgoing CommMessage dicts to
                                            publish on /swarm/comm.

    ``send(sender_id, receiver_id, kind, payload) -> bool`` is supplied by the
    node. In a *fully distributed* ROS2 deployment each drone_node owns only its
    own DroneView and publishes messages; the synchronous ``send`` boolean shown
    here is the single-process / co-simulation form that mirrors swarm_sim exactly
    and is what the default launch uses for an apples-to-apples comparison with the
    Python results. See README "Stubbed vs functional".
    """

    name = "base"

    def reset(self, grid: np.ndarray, drones: List[DroneView],
              base_pos: Tuple[int, int]) -> None:
        pass

    def step(self, grid: np.ndarray, drones: List[DroneView],
             base_pos: Tuple[int, int], send, t: int):
        raise NotImplementedError
