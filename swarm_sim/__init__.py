"""swarm_sim: lightweight simulator for resilient multi-drone SAR under comms failures.

See CLAUDE.md for the research framing. v0 deliverable: world + drones + pluggable
CommModel + CoordinationStrategy + metrics + Monte-Carlo runner, with a centralized
baseline demo.
"""

from .config import SimConfig
from .world import World
from .drone import Drone
from .simulator import Simulator
from . import comm, strategies, metrics

__all__ = [
    "SimConfig",
    "World",
    "Drone",
    "Simulator",
    "comm",
    "strategies",
    "metrics",
]
