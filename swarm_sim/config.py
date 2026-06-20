"""Simulation configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class SimConfig:
    # World
    width: int = 50
    height: int = 50
    n_obstacles: int = 40          # number of obstacle cells (random unless seeded layout)
    obstacle_clusters: int = 8     # obstacles are grown into clusters for realism
    n_victims: int = 15
    n_drones: int = 10

    # Drone
    sensing_radius: float = 3.0    # cells; detects victims & marks coverage within radius
    comm_radius: float = 12.0      # nominal max inter-drone / base comm range (cells)
    speed: float = 1.0             # cells per step (kinematic)

    # Base station (for centralized strategy)
    base_pos: Optional[Tuple[int, int]] = None  # defaults to (0, 0)

    # Episode
    max_steps: int = 600
    seed: int = 0

    # Bookkeeping
    extra: dict = field(default_factory=dict)

    def base(self) -> Tuple[int, int]:
        return self.base_pos if self.base_pos is not None else (0, 0)
