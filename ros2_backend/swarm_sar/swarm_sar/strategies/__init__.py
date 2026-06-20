"""ROS2 strategy adapters.

Each adapter is a thin wrapper around the core ``assign()`` logic of the
corresponding ``swarm_sim`` strategy, restructured to be driven by a ROS2 node
rather than the synchronous simulator loop. The *algorithm* is identical; only the
plumbing differs:

  swarm_sim                         ROS2 backend
  ------------------------------    ----------------------------------------
  Simulator calls strategy.assign   drone_node timer calls adapter.step()
  send(a, b, payload) -> bool       publish CommMessage on /swarm/comm; the
                                    comm_bridge_node applies the CommModel and
                                    only delivered messages arrive in the inbox
  drone.target = (y, x)             adapter returns a target cell that the node
                                    converts into a velocity/waypoint command

See ``base.py`` for the common interface and ``frontier.py`` for the shared
frontier-cell helper ported from swarm_sim/strategies.py.
"""

from .base import StrategyAdapter, DroneView
from .centralized import CentralizedAdapter
from .decentralized import DecentralizedAdapter
from .stigmergy import StigmergyAdapter

ADAPTERS = {
    "centralized": CentralizedAdapter,
    "decentralized": DecentralizedAdapter,
    "stigmergy": StigmergyAdapter,
}


def build_strategy(name: str, **kwargs) -> StrategyAdapter:
    """Instantiate a strategy adapter by name (from config/params.yaml)."""
    key = str(name).lower()
    if key not in ADAPTERS:
        raise ValueError(
            f"unknown strategy {name!r}; choose from {sorted(ADAPTERS)}")
    return ADAPTERS[key](**kwargs)


__all__ = [
    "StrategyAdapter", "DroneView",
    "CentralizedAdapter", "DecentralizedAdapter", "StigmergyAdapter",
    "ADAPTERS", "build_strategy",
]
