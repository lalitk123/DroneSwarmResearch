"""Pluggable communication-failure models.

A CommModel decides, for a directed (sender -> receiver) message at a given step,
whether it is delivered (and optionally with what latency). The simulator routes
all inter-agent messages through the active model, so swapping the failure regime
is a one-line change.

Conventions:
- `deliver(sender, receiver, world, step)` returns delivery latency in steps
  (0 = same step) or None if the message is dropped.
- Endpoints expose `.pos` (np array (y,x)) and `.alive`. The base station is passed
  as a lightweight object with the same attributes.
"""

from __future__ import annotations

import numpy as np


class CommModel:
    name = "base"

    def deliver(self, sender, receiver, world, step):
        raise NotImplementedError

    def __repr__(self):
        return f"<CommModel {self.name}>"


class PerfectComm(CommModel):
    name = "perfect"

    def deliver(self, sender, receiver, world, step):
        if not (sender.alive and receiver.alive):
            return None
        return 0


class PacketLoss(CommModel):
    """Each message dropped independently with probability p (Bernoulli)."""
    name = "packet_loss"

    def __init__(self, p: float, rng: np.random.Generator):
        self.p = float(p)
        self.rng = rng

    def deliver(self, sender, receiver, world, step):
        if not (sender.alive and receiver.alive):
            return None
        return None if self.rng.random() < self.p else 0


class RangeComm(CommModel):
    """Delivery probability depends on distance.

    mode='disk'    : binary unit-disk cutoff at `radius`.
    mode='pathloss': p_deliver = min(1, (radius/d)^alpha) for d > 0 (Friis-like).
    """
    name = "range"

    def __init__(self, radius: float, rng: np.random.Generator,
                 mode: str = "disk", alpha: float = 2.0):
        self.radius = float(radius)
        self.rng = rng
        self.mode = mode
        self.alpha = alpha

    def deliver(self, sender, receiver, world, step):
        if not (sender.alive and receiver.alive):
            return None
        d = float(np.linalg.norm(sender.pos - receiver.pos))
        if self.mode == "disk":
            return 0 if d <= self.radius else None
        p = 1.0 if d <= 1e-6 else min(1.0, (self.radius / d) ** self.alpha)
        return 0 if self.rng.random() < p else None


class DeadZone(CommModel):
    """Rectangular dead zones; any link with an endpoint inside a zone is dropped.

    zones: list of (y0, x0, y1, x1). Optionally wraps a base model for normal links.
    """
    name = "dead_zone"

    def __init__(self, zones, base: CommModel | None = None,
                 block_crossing: bool = False):
        self.zones = zones
        self.base = base or PerfectComm()
        # block_crossing=True also drops links whose straight path crosses a zone
        # (line-of-sight obstruction by terrain/buildings), so a chain can't "jump
        # over" the dead zone — relays must physically route around it.
        self.block_crossing = block_crossing

    def _in_zone(self, pos):
        y, x = pos
        for (y0, x0, y1, x1) in self.zones:
            if y0 <= y <= y1 and x0 <= x <= x1:
                return True
        return False

    @staticmethod
    def _seg_hits_rect(p0, p1, rect, samples=24):
        y0, x0, y1, x1 = rect
        for t in np.linspace(0.0, 1.0, samples):
            y = p0[0] + t * (p1[0] - p0[0])
            x = p0[1] + t * (p1[1] - p0[1])
            if y0 <= y <= y1 and x0 <= x <= x1:
                return True
        return False

    def deliver(self, sender, receiver, world, step):
        if not (sender.alive and receiver.alive):
            return None
        if self._in_zone(sender.pos) or self._in_zone(receiver.pos):
            return None
        if self.block_crossing:
            for rect in self.zones:
                if self._seg_hits_rect(sender.pos, receiver.pos, rect):
                    return None
        return self.base.deliver(sender, receiver, world, step)


class Latency(CommModel):
    """Adds random integer latency (delay) on top of a wrapped base model."""
    name = "latency"

    def __init__(self, max_delay: int, rng: np.random.Generator,
                 base: CommModel | None = None):
        self.max_delay = int(max_delay)
        self.rng = rng
        self.base = base or PerfectComm()

    def deliver(self, sender, receiver, world, step):
        d = self.base.deliver(sender, receiver, world, step)
        if d is None:
            return None
        return d + int(self.rng.integers(0, self.max_delay + 1))


class Composite(CommModel):
    """Apply several models in sequence; message survives only if all deliver.

    Latencies add up. Useful for the combined 'realistic disaster' scenario.
    """
    name = "composite"

    def __init__(self, models):
        self.models = models
        self.name = "composite[" + "+".join(m.name for m in models) + "]"

    def deliver(self, sender, receiver, world, step):
        total = 0
        for m in self.models:
            d = m.deliver(sender, receiver, world, step)
            if d is None:
                return None
            total += d
        return total
