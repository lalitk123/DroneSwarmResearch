"""Metrics computation and resilience scoring."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List

import numpy as np


@dataclass
class EpisodeMetrics:
    coverage: float = 0.0
    detection_rate: float = 0.0
    time_to_first_victim: int = -1     # -1 = none found
    mission_time: int = -1             # step at which all victims found, else -1
    redundancy: float = 0.0
    mean_connectivity: float = 0.0     # mean fraction in largest connected component
    total_path_length: float = 0.0
    n_alive_end: int = 0

    def as_dict(self):
        return asdict(self)


def largest_connected_fraction(positions, alive, comm_radius) -> float:
    """Fraction of living drones in the largest comm-connected component (unit-disk)."""
    idx = [i for i, a in enumerate(alive) if a]
    n = len(idx)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    pos = np.array([positions[i] for i in idx])
    # adjacency by range
    dy = pos[:, 0:1] - pos[:, 0:1].T
    dx = pos[:, 1:2] - pos[:, 1:2].T
    adj = (dy ** 2 + dx ** 2) <= comm_radius ** 2
    # connected components via BFS
    seen = [False] * n
    best = 0
    for s in range(n):
        if seen[s]:
            continue
        stack, size = [s], 0
        seen[s] = True
        while stack:
            u = stack.pop()
            size += 1
            for v in range(n):
                if adj[u, v] and not seen[v]:
                    seen[v] = True
                    stack.append(v)
        best = max(best, size)
    return best / n


def resilience_auc(severities: List[float], values: List[float],
                   baseline: float | None = None) -> float:
    """Normalized area under the performance-vs-severity curve (0..1).

    severities in [0,1] (e.g. packet-loss prob). values is the metric at each.
    If baseline given, values are normalized by it (performance retention).
    A perfectly resilient strategy -> flat curve -> AUC ~ 1.0.
    """
    s = np.asarray(severities, dtype=float)
    v = np.asarray(values, dtype=float)
    order = np.argsort(s)
    s, v = s[order], v[order]
    if baseline:
        v = v / baseline
    if len(s) < 2:
        return float(v.mean()) if len(v) else 0.0
    span = s[-1] - s[0]
    if span <= 0:
        return float(v.mean())
    return float(np.trapz(v, s) / span)
