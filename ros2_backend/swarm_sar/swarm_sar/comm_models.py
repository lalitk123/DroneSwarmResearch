"""Communication-failure models for the ROS2 backend.

This is a faithful, ROS2-agnostic port of ``swarm_sim/comm.py``. The
``comm_bridge_node`` instantiates one of these from ``config/params.yaml`` and
calls ``deliver()`` on every inter-drone CommMessage BEFORE relaying it — exactly
as ``swarm_sim/simulator.py`` routes every ``send()`` through the active model.

Endpoint convention (kept identical to swarm_sim): each endpoint is any object
exposing ``.pos`` (an iterable (y, x) or (y, x, z)) and ``.alive`` (bool). In the
bridge we wrap the sender/receiver coordinates carried in the CommMessage in a
tiny ``Endpoint`` namedtuple before calling ``deliver()``.

``deliver(sender, receiver, step)`` returns the delivery latency in *steps*
(0 = same tick) or ``None`` if the message is dropped. No numpy hard requirement
for the simple models; numpy is used where convenient and is a declared dep.
"""

from __future__ import annotations

import math
import random
from collections import namedtuple
from typing import List, Optional, Sequence, Tuple

# Lightweight endpoint used by the bridge. (y, x) is enough for all models; z is
# accepted and ignored by the planar range/dead-zone models (2.5D world).
Endpoint = namedtuple("Endpoint", ["pos", "alive"])


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(ai) - float(bi)) ** 2 for ai, bi in zip(a, b)))


class CommModel:
    """Base class — mirrors swarm_sim.comm.CommModel."""

    name = "base"

    def deliver(self, sender, receiver, step: int) -> Optional[int]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<CommModel {self.name}>"


class PerfectComm(CommModel):
    name = "perfect"

    def deliver(self, sender, receiver, step: int) -> Optional[int]:
        if not (sender.alive and receiver.alive):
            return None
        return 0


class PacketLoss(CommModel):
    """Bernoulli drop with probability p (S1)."""

    name = "packet_loss"

    def __init__(self, p: float, rng: Optional[random.Random] = None):
        self.p = float(p)
        self.rng = rng or random.Random()

    def deliver(self, sender, receiver, step: int) -> Optional[int]:
        if not (sender.alive and receiver.alive):
            return None
        return None if self.rng.random() < self.p else 0


class RangeComm(CommModel):
    """Distance-dependent delivery (S2).

    mode='disk'     : binary unit-disk cutoff at ``radius``.
    mode='pathloss' : p_deliver = min(1, (radius/d)^alpha) (Friis-like).
    """

    name = "range"

    def __init__(self, radius: float, rng: Optional[random.Random] = None,
                 mode: str = "disk", alpha: float = 2.0):
        self.radius = float(radius)
        self.rng = rng or random.Random()
        self.mode = mode
        self.alpha = float(alpha)

    def deliver(self, sender, receiver, step: int) -> Optional[int]:
        if not (sender.alive and receiver.alive):
            return None
        d = _dist(sender.pos, receiver.pos)
        if self.mode == "disk":
            return 0 if d <= self.radius else None
        p = 1.0 if d <= 1e-6 else min(1.0, (self.radius / d) ** self.alpha)
        return 0 if self.rng.random() < p else None


class DeadZone(CommModel):
    """Rectangular dead zones (S3). Any link with an endpoint inside a zone drops.

    zones: list of (y0, x0, y1, x1). Optionally wraps a base model for normal links
    and (block_crossing=True) also drops links whose straight path crosses a zone
    — line-of-sight obstruction, so a chain can't 'jump over' a dead zone.
    """

    name = "dead_zone"

    def __init__(self, zones: List[Tuple[float, float, float, float]],
                 base: Optional[CommModel] = None, block_crossing: bool = False):
        self.zones = zones
        self.base = base or PerfectComm()
        self.block_crossing = block_crossing

    def _in_zone(self, pos) -> bool:
        y, x = pos[0], pos[1]
        for (y0, x0, y1, x1) in self.zones:
            if y0 <= y <= y1 and x0 <= x <= x1:
                return True
        return False

    @staticmethod
    def _seg_hits_rect(p0, p1, rect, samples: int = 24) -> bool:
        y0, x0, y1, x1 = rect
        for i in range(samples):
            t = i / (samples - 1)
            y = p0[0] + t * (p1[0] - p0[0])
            x = p0[1] + t * (p1[1] - p0[1])
            if y0 <= y <= y1 and x0 <= x <= x1:
                return True
        return False

    def deliver(self, sender, receiver, step: int) -> Optional[int]:
        if not (sender.alive and receiver.alive):
            return None
        if self._in_zone(sender.pos) or self._in_zone(receiver.pos):
            return None
        if self.block_crossing:
            for rect in self.zones:
                if self._seg_hits_rect(sender.pos, receiver.pos, rect):
                    return None
        return self.base.deliver(sender, receiver, step)


class Latency(CommModel):
    """Adds random integer latency on top of a wrapped base model (S4)."""

    name = "latency"

    def __init__(self, max_delay: int, rng: Optional[random.Random] = None,
                 base: Optional[CommModel] = None):
        self.max_delay = int(max_delay)
        self.rng = rng or random.Random()
        self.base = base or PerfectComm()

    def deliver(self, sender, receiver, step: int) -> Optional[int]:
        d = self.base.deliver(sender, receiver, step)
        if d is None:
            return None
        return d + self.rng.randint(0, self.max_delay)


class Composite(CommModel):
    """Apply several models in sequence; survives only if all deliver (S6)."""

    name = "composite"

    def __init__(self, models: List[CommModel]):
        self.models = models
        self.name = "composite[" + "+".join(m.name for m in models) + "]"

    def deliver(self, sender, receiver, step: int) -> Optional[int]:
        total = 0
        for m in self.models:
            d = m.deliver(sender, receiver, step)
            if d is None:
                return None
            total += d
        return total


def build_comm_model(spec: dict, rng: Optional[random.Random] = None) -> CommModel:
    """Factory mirroring the failure scenarios S0..S6 from CLAUDE.md.

    ``spec`` comes straight from config/params.yaml, e.g.::

        comm:
          model: range        # perfect|packet_loss|range|dead_zone|latency|composite
          severity: 0.5       # interpretation depends on model (see below)
          range_radius: 12.0
          range_mode: disk
          packet_loss_p: 0.3
          dead_zones: [[10, 10, 20, 20]]
          max_delay: 3

    The ``severity`` knob lets the experiment runner sweep one scalar (as in the
    swarm_sim degradation curves) without rewriting the whole spec.
    """
    rng = rng or random.Random(spec.get("seed", 0))
    model = str(spec.get("model", "perfect")).lower()

    if model == "perfect":
        return PerfectComm()
    if model == "packet_loss":
        p = float(spec.get("packet_loss_p", spec.get("severity", 0.0)))
        return PacketLoss(p, rng)
    if model == "range":
        return RangeComm(float(spec.get("range_radius", 12.0)), rng,
                         mode=str(spec.get("range_mode", "disk")),
                         alpha=float(spec.get("range_alpha", 2.0)))
    if model == "dead_zone":
        zones = [tuple(z) for z in spec.get("dead_zones", [])]
        return DeadZone(zones, block_crossing=bool(spec.get("block_crossing", False)))
    if model == "latency":
        return Latency(int(spec.get("max_delay", 3)), rng)
    if model == "composite":
        zones = [tuple(z) for z in spec.get("dead_zones", [])]
        return Composite([
            RangeComm(float(spec.get("range_radius", 12.0)), rng,
                      mode=str(spec.get("range_mode", "disk"))),
            DeadZone(zones),
            PacketLoss(float(spec.get("packet_loss_p", 0.1)), rng),
        ])
    raise ValueError(f"unknown comm model: {model!r}")
