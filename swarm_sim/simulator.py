"""Simulator: ties world + drones + comm model + strategy together with metrics."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from .config import SimConfig
from .world import World
from .drone import Drone
from .comm import CommModel, PerfectComm
from .strategies import CoordinationStrategy
from .metrics import EpisodeMetrics, largest_connected_fraction


@dataclass
class BaseStation:
    pos: np.ndarray
    alive: bool = True
    id: int = -1


class Simulator:
    def __init__(self, cfg: SimConfig, strategy: CoordinationStrategy,
                 comm: CommModel | None = None,
                 node_failure_rate: float = 0.0,
                 scheduled_failures=None,
                 failure_mode: str = "random",
                 multihop: bool = False):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.world = World(cfg, self.rng)
        self.strategy = strategy
        self.comm = comm or PerfectComm()
        # multihop: a message is delivered if a *path* of deliverable links exists in
        # the comm graph (base + living drones), not just a direct link. Required for
        # relay drones to matter — they create bridging hops.
        self.multihop = multihop
        self._component = {}   # node id -> component label, recomputed each step
        self.node_failure_rate = node_failure_rate  # per-drone prob of failing per step
        # scheduled_failures: dict {step: n_drones} — kill n living drones at `step`.
        # failure_mode: 'random' (uniform) or 'targeted' (kill highest comm-degree drones,
        # i.e. the best-connected hubs — the adversarial worst case for connectivity).
        self.scheduled_failures = dict(scheduled_failures or {})
        self.failure_mode = failure_mode

        by, bx = cfg.base()
        self.base = BaseStation(pos=np.array([by, bx], dtype=float))
        self.drones = self._spawn_drones()
        self.strategy.reset(self.world, self.drones, self.base)

        self._delayed = []          # min-heap of (deliver_step, seq, receiver_id)
        self._seq = 0
        self._connectivity_samples = []

    def _spawn_drones(self):
        drones = []
        by, bx = self.cfg.base()
        for i in range(self.cfg.n_drones):
            # spawn near base on a free cell
            for _ in range(100):
                y = int(np.clip(by + self.rng.integers(-3, 4), 0, self.cfg.height - 1))
                x = int(np.clip(bx + self.rng.integers(-3, 4), 0, self.cfg.width - 1))
                if self.world.is_free(y, x):
                    break
            drones.append(Drone(i, (y, x), self.cfg, self.world))
        return drones

    # ---- comm routing ----------------------------------------------------
    def _send(self, sender, receiver, payload):
        """Route a message through the comm model. Returns True if it will be
        delivered (immediately or later). Latency is recorded but for v0 the
        boolean delivery is what strategies use."""
        if self.multihop:
            # delivered iff sender and receiver are in the same connected component
            # of this step's comm graph (path of deliverable links may pass via relays)
            return self._component.get(sender.id) == self._component.get(receiver.id) \
                and sender.id in self._component
        latency = self.comm.deliver(sender, receiver, self.world, self.step_idx)
        if latency is None:
            return False
        if latency > 0:
            self._seq += 1
            heapq.heappush(self._delayed,
                           (self.step_idx + latency, self._seq, receiver.id))
        return True

    def _compute_components(self):
        """Label connected components of the comm graph over {base, living drones}."""
        nodes = [self.base] + [d for d in self.drones if d.alive]
        self._component = {}
        n = len(nodes)
        # adjacency via the active comm model (sampled once this step)
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if self.comm.deliver(nodes[i], nodes[j], self.world, self.step_idx) is not None:
                    adj[i].append(j)
                    adj[j].append(i)
        label = 0
        seen = [False] * n
        comp_sizes = {}
        for s in range(n):
            if seen[s]:
                continue
            stack = [s]
            seen[s] = True
            sz = 0
            while stack:
                u = stack.pop()
                self._component[nodes[u].id] = label
                sz += 1
                for v in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            comp_sizes[label] = sz
            label += 1
        # true largest-connected fraction over drones (exclude base node)
        n_drones = n - 1
        self._last_conn = (max(comp_sizes.values()) / n) if n_drones > 0 else 0.0

    # ---- main loop --------------------------------------------------------
    def run(self, snapshot_steps=None) -> EpisodeMetrics:
        cfg = self.cfg
        mission_time = -1
        snapshot_steps = set(snapshot_steps or [])
        self.snapshots = {}  # step -> (coverage, detection_rate)
        for self.step_idx in range(cfg.max_steps):
            self._maybe_fail_nodes()

            # 1) sense
            for d in self.drones:
                d.step_sense(self.world, self.step_idx)

            # 2) coordinate (assign targets, exchanging messages via comm)
            if self.multihop:
                self._compute_components()
            self.strategy.assign(self.world, self.drones, self.base,
                                 self._send, self.step_idx)

            # 3) move
            for d in self.drones:
                d.move_towards(d.target, self.world)

            # 4) sample connectivity — use the true comm graph when multihop is on,
            # otherwise the nominal unit-disk over the configured comm radius
            if self.multihop:
                self._connectivity_samples.append(getattr(self, "_last_conn", 0.0))
            else:
                self._connectivity_samples.append(
                    largest_connected_fraction(
                        [d.pos for d in self.drones],
                        [d.alive for d in self.drones],
                        cfg.comm_radius))

            # 4b) optional snapshots (1-indexed step count)
            if (self.step_idx + 1) in snapshot_steps:
                self.snapshots[self.step_idx + 1] = (
                    self.world.coverage_fraction(), self.world.detection_rate())

            # 5) mission completion check
            if mission_time < 0 and self.world.victim_found.all() and len(self.world.victims):
                mission_time = self.step_idx
                break

        return self._collect_metrics(mission_time)

    def _maybe_fail_nodes(self):
        # scheduled bulk failure at a specific step
        n_sched = self.scheduled_failures.get(self.step_idx, 0)
        if n_sched:
            living = [d for d in self.drones if d.alive]
            if self.failure_mode == "targeted" and len(living) > 1:
                # kill the best-connected drones (highest in-range comm degree)
                pos = np.array([d.pos for d in living])
                d2 = (((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
                degree = ((d2 <= self.cfg.comm_radius ** 2).sum(1) - 1)
                order = list(np.argsort(-degree))
                living = [living[i] for i in order]
            else:
                self.rng.shuffle(living)
            for d in living[:int(n_sched)]:
                d.alive = False
                d.target = None
        # per-step stochastic failure
        if self.node_failure_rate <= 0:
            return
        for d in self.drones:
            if d.alive and self.rng.random() < self.node_failure_rate:
                d.alive = False
                d.target = None

    def _collect_metrics(self, mission_time) -> EpisodeMetrics:
        w = self.world
        found_steps = w.victim_found_step[w.victim_found]
        ttf = int(found_steps.min()) if len(found_steps) else -1
        return EpisodeMetrics(
            coverage=w.coverage_fraction(),
            detection_rate=w.detection_rate(),
            time_to_first_victim=ttf,
            mission_time=mission_time,
            redundancy=w.redundancy_ratio(),
            mean_connectivity=float(np.mean(self._connectivity_samples or [0.0])),
            total_path_length=float(sum(d.path_length for d in self.drones)),
            n_alive_end=int(sum(d.alive for d in self.drones)),
        )
