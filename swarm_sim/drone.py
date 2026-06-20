"""Drone agent: kinematic motion, local sensing, local map, bookkeeping."""

from __future__ import annotations

import numpy as np

from .world import World


class Drone:
    def __init__(self, drone_id: int, pos, cfg, world: World):
        self.id = drone_id
        self.cfg = cfg
        self.pos = np.array(pos, dtype=float)   # (y, x)
        self.alive = True
        self.path_length = 0.0
        self.target = None                       # (y, x) goal, set by strategy

        # Local belief: each drone keeps its own coverage map + known victims.
        # Shared/merged via comm depending on strategy.
        self.known_covered = np.zeros((cfg.height, cfg.width), dtype=bool)
        self.known_victims = {}                  # victim_id -> (y, x)
        # Inbox for messages delivered this step (filled by simulator/comm).
        self.inbox = []

    def step_sense(self, world: World, step: int):
        if not self.alive:
            return []
        newly = world.sense(self.pos, self.cfg.sensing_radius, step)
        # update local belief from own sensing
        self._mark_local_coverage(world)
        for vid in newly:
            self.known_victims[vid] = tuple(world.victims[vid])
        return newly

    def _mark_local_coverage(self, world: World):
        r = int(np.ceil(self.cfg.sensing_radius))
        y0, x0 = self.pos
        ylo, yhi = max(0, int(y0) - r), min(self.cfg.height, int(y0) + r + 1)
        xlo, xhi = max(0, int(x0) - r), min(self.cfg.width, int(x0) + r + 1)
        ys, xs = np.mgrid[ylo:yhi, xlo:xhi]
        d2 = (ys - y0) ** 2 + (xs - x0) ** 2
        self.known_covered[ylo:yhi, xlo:xhi] |= (d2 <= self.cfg.sensing_radius ** 2)

    def move_towards(self, target, world: World):
        """Kinematic step of length <= speed towards target, avoiding obstacles."""
        if not self.alive or target is None:
            return
        target = np.array(target, dtype=float)
        delta = target - self.pos
        dist = np.linalg.norm(delta)
        if dist < 1e-6:
            return
        step_vec = delta / dist * min(self.cfg.speed, dist)
        new_pos = self.pos + step_vec
        ny, nx = int(round(new_pos[0])), int(round(new_pos[1]))
        if world.is_free(ny, nx):
            self.path_length += np.linalg.norm(step_vec)
            self.pos = new_pos
        else:
            # simple obstacle avoidance: try axis-aligned slides
            for cand in (np.array([new_pos[0], self.pos[1]]),
                         np.array([self.pos[0], new_pos[1]])):
                cy, cx = int(round(cand[0])), int(round(cand[1]))
                if world.is_free(cy, cx):
                    self.path_length += np.linalg.norm(cand - self.pos)
                    self.pos = cand
                    return

    def cell(self):
        return (int(round(self.pos[0])), int(round(self.pos[1])))
