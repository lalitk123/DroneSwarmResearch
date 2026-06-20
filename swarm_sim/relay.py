"""Adaptive relay-drone extension.

Wraps any base CoordinationStrategy and reserves K drones as *relays* that do not
search — instead they reposition each step to bridge the base station to the searcher
swarm, maximizing the connected component so the base can still reach searchers under
range / dead-zone comms failure. Requires `Simulator(multihop=True)` so that messages
actually route through the relay chain.

Hypothesis: relays rescue the connectivity-dependent strategies (esp. centralized)
under geographic comms loss, at the cost of K fewer searchers.
"""

from __future__ import annotations

import numpy as np

from .strategies import CoordinationStrategy


# ---- geometry helpers (shared by the multi-branch router) -------------------

def _seg_hits_rect(p0, p1, rect, samples=24):
    y0, x0, y1, x1 = rect
    for t in np.linspace(0.0, 1.0, samples):
        y = p0[0] + t * (p1[0] - p0[0])
        x = p0[1] + t * (p1[1] - p0[1])
        if y0 <= y <= y1 and x0 <= x <= x1:
            return True
    return False


def _seg_hits_any(p0, p1, zones):
    return any(_seg_hits_rect(p0, p1, z) for z in zones)


def _expanded_corners(rect, margin):
    y0, x0, y1, x1 = rect
    return [np.array([y0 - margin, x0 - margin]), np.array([y0 - margin, x1 + margin]),
            np.array([y1 + margin, x0 - margin]), np.array([y1 + margin, x1 + margin])]


def _route_around(p0, p1, zones, margin=2.0):
    """Return a polyline [p0, ..., p1] that detours around any zone the direct
    segment crosses. Picks the zone corner giving the shortest valid detour."""
    if not _seg_hits_any(p0, p1, zones):
        return [p0, p1]
    best, best_len = None, np.inf
    for z in zones:
        if not _seg_hits_rect(p0, p1, z):
            continue
        for c in _expanded_corners(z, margin):
            if _seg_hits_any(p0, c, zones) or _seg_hits_any(c, p1, zones):
                continue
            L = np.linalg.norm(p0 - c) + np.linalg.norm(c - p1)
            if L < best_len:
                best, best_len = c, L
    return [p0, best, p1] if best is not None else [p0, p1]


def _place_along(polyline, k):
    """k points spread evenly by arc length along the polyline (interior fractions)."""
    pts = [np.asarray(p, dtype=float) for p in polyline]
    seg = [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
    total = sum(seg) or 1.0
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    out = []
    for i in range(k):
        target = total * (i + 1) / (k + 1)
        j = int(np.searchsorted(cum, target) - 1)
        j = max(0, min(j, len(seg) - 1))
        local = (target - cum[j]) / (seg[j] or 1.0)
        out.append(tuple(pts[j] + local * (pts[j + 1] - pts[j])))
    return out


def _kmeans(points, k, iters=8, rng=None):
    """Lightweight k-means with farthest-first init. points: (N,2)."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    k = min(k, n)
    # farthest-first seeding
    idx = [0 if rng is None else int(rng.integers(n))]
    while len(idx) < k:
        d = np.min([np.linalg.norm(pts - pts[c], axis=1) for c in idx], axis=0)
        idx.append(int(np.argmax(d)))
    cent = pts[idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        labels = np.argmin(
            np.stack([np.linalg.norm(pts - c, axis=1) for c in cent], axis=1), axis=1)
        for c in range(k):
            m = labels == c
            if m.any():
                cent[c] = pts[m].mean(axis=0)
    return labels, cent


class RelayStrategy(CoordinationStrategy):
    def __init__(self, base_strategy: CoordinationStrategy, n_relays: int = 3):
        self.base = base_strategy
        self.n_relays = n_relays
        self.name = f"relay({base_strategy.name},k={n_relays})"
        self._relay_ids = None

    def reset(self, world, drones, base):
        # reserve the highest-id drones as relays
        ids = sorted(d.id for d in drones)
        self._relay_ids = set(ids[-self.n_relays:]) if self.n_relays > 0 else set()
        self.base.reset(world, drones, base)

    def assign(self, world, drones, base, send, step):
        searchers = [d for d in drones if d.id not in self._relay_ids]
        relays = [d for d in drones if d.id in self._relay_ids and d.alive]

        # 1) searchers run the wrapped strategy (relays excluded from search duty)
        self.base.assign(world, searchers, base, send, step)

        # 2) relays form an adaptive chain from base -> searcher centroid.
        living_searchers = [d for d in searchers if d.alive]
        if not relays:
            return
        if not living_searchers:
            for r in relays:
                r.target = tuple(base.pos)
            return

        # Anchor on the farthest searcher from base — that is the link most likely to
        # be broken, so bridging it yields the biggest connectivity gain. (Centroid is
        # a poor anchor: under a central dead zone it sits inside the blackout.)
        dists = [np.linalg.norm(d.pos - base.pos) for d in living_searchers]
        anchor = living_searchers[int(np.argmax(dists))].pos
        # spread relays evenly along the base->anchor segment to maximize bridging hops
        k = len(relays)
        for i, r in enumerate(relays):
            frac = (i + 1) / (k + 1)
            r.target = tuple(base.pos + frac * (anchor - base.pos))


class MultiBranchRelayStrategy(CoordinationStrategy):
    """Dead-zone-aware relays. Clusters searchers into branches and routes each branch
    as its own relay chain from the base to that cluster, detouring around known dead
    zones (instead of one straight chain to the single farthest searcher, which would
    cross a central dead zone and stay broken)."""

    def __init__(self, base_strategy: CoordinationStrategy, n_relays: int = 4,
                 dead_zones=None, n_branches: int = 2, margin: float = 2.0):
        self.base = base_strategy
        self.n_relays = n_relays
        self.dead_zones = list(dead_zones or [])
        self.n_branches = n_branches
        self.margin = margin
        self.name = f"multibranch({base_strategy.name},k={n_relays},b={n_branches})"
        self._relay_ids = None
        self._rng = np.random.default_rng(0)

    def reset(self, world, drones, base):
        ids = sorted(d.id for d in drones)
        self._relay_ids = set(ids[-self.n_relays:]) if self.n_relays > 0 else set()
        self.base.reset(world, drones, base)

    def assign(self, world, drones, base, send, step):
        searchers = [d for d in drones if d.id not in self._relay_ids]
        relays = [d for d in drones if d.id in self._relay_ids and d.alive]
        self.base.assign(world, searchers, base, send, step)

        living = [d for d in searchers if d.alive]
        if not relays:
            return
        if not living:
            for r in relays:
                r.target = tuple(base.pos)
            return

        # 1) cluster searchers into branches
        b = min(self.n_branches, len(living), len(relays))
        pts = np.array([d.pos for d in living])
        labels, cents = _kmeans(pts, b, rng=self._rng)

        # 2) allocate relays to branches, proportional to branch distance from base
        #    (longer reach needs more hops), at least 1 each
        bdist = np.array([np.linalg.norm(cents[c] - base.pos) for c in range(b)])
        alloc = np.ones(b, dtype=int)
        remaining = len(relays) - b
        if remaining > 0:
            w = bdist / bdist.sum()
            extra = np.floor(w * remaining).astype(int)
            alloc += extra
            # hand out any leftover to the farthest branches
            for c in np.argsort(-bdist)[:int(len(relays) - alloc.sum())]:
                alloc[c] += 1

        # 3) place each branch's relays along a route that detours around dead zones
        ri = 0
        for c in range(b):
            route = _route_around(np.asarray(base.pos, float), cents[c],
                                  self.dead_zones, self.margin)
            for p in _place_along(route, int(alloc[c])):
                if ri < len(relays):
                    relays[ri].target = p
                    ri += 1
