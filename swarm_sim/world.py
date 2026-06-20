"""2.5D grid world: obstacles, victims, coverage tracking."""

from __future__ import annotations

import numpy as np

from .config import SimConfig

FREE, OBSTACLE = 0, 1


class World:
    """Grid world. Cells are FREE or OBSTACLE. Victims sit on FREE cells.

    Tracks which cells have been covered (sensed by any drone) and which victims
    have been detected.
    """

    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.grid = np.zeros((cfg.height, cfg.width), dtype=np.int8)
        self._place_obstacles()
        self.victims = self._place_victims()           # (N,2) int array of (y,x)
        self.victim_found = np.zeros(len(self.victims), dtype=bool)
        self.victim_found_step = np.full(len(self.victims), -1, dtype=int)
        # coverage over free cells
        self.covered = np.zeros_like(self.grid, dtype=bool)
        self.coverage_count = np.zeros_like(self.grid, dtype=np.int32)
        self.n_free = int((self.grid == FREE).sum())

    # ---- setup -----------------------------------------------------------
    def _place_obstacles(self):
        cfg = self.cfg
        placed = 0
        clusters = max(1, cfg.obstacle_clusters)
        per = max(1, cfg.n_obstacles // clusters)
        for _ in range(clusters):
            cy = self.rng.integers(0, cfg.height)
            cx = self.rng.integers(0, cfg.width)
            for _ in range(per):
                if placed >= cfg.n_obstacles:
                    return
                # random walk from cluster seed
                y = int(np.clip(cy + self.rng.integers(-2, 3), 0, cfg.height - 1))
                x = int(np.clip(cx + self.rng.integers(-2, 3), 0, cfg.width - 1))
                if (y, x) == tuple(cfg.base()):
                    continue
                if self.grid[y, x] == FREE:
                    self.grid[y, x] = OBSTACLE
                    placed += 1

    def _place_victims(self):
        cfg = self.cfg
        free_cells = np.argwhere(self.grid == FREE)
        idx = self.rng.choice(len(free_cells), size=min(cfg.n_victims, len(free_cells)),
                              replace=False)
        return free_cells[idx].copy()

    # ---- queries ---------------------------------------------------------
    def is_free(self, y: int, x: int) -> bool:
        return (0 <= y < self.cfg.height and 0 <= x < self.cfg.width
                and self.grid[y, x] == FREE)

    def sense(self, pos: np.ndarray, radius: float, step: int):
        """Mark coverage and detect victims within `radius` of pos=(y,x).

        Returns list of victim indices newly detected this call.
        """
        y0, x0 = float(pos[0]), float(pos[1])
        r = int(np.ceil(radius))
        ylo, yhi = max(0, int(y0) - r), min(self.cfg.height, int(y0) + r + 1)
        xlo, xhi = max(0, int(x0) - r), min(self.cfg.width, int(x0) + r + 1)
        ys, xs = np.mgrid[ylo:yhi, xlo:xhi]
        d2 = (ys - y0) ** 2 + (xs - x0) ** 2
        mask = (d2 <= radius ** 2) & (self.grid[ylo:yhi, xlo:xhi] == FREE)
        self.covered[ylo:yhi, xlo:xhi] |= mask
        self.coverage_count[ylo:yhi, xlo:xhi] += mask

        newly = []
        if len(self.victims):
            vd2 = ((self.victims[:, 0] - y0) ** 2 + (self.victims[:, 1] - x0) ** 2)
            hit = (vd2 <= radius ** 2) & (~self.victim_found)
            for i in np.argwhere(hit).ravel():
                self.victim_found[i] = True
                self.victim_found_step[i] = step
                newly.append(int(i))
        return newly

    # ---- metrics helpers -------------------------------------------------
    def coverage_fraction(self) -> float:
        return float(self.covered.sum()) / max(1, self.n_free)

    def detection_rate(self) -> float:
        if len(self.victims) == 0:
            return 1.0
        return float(self.victim_found.sum()) / len(self.victims)

    def redundancy_ratio(self) -> float:
        """Total sensing events over covered cells / covered cells (>=1; 1 = no overlap)."""
        cov = self.covered.sum()
        if cov == 0:
            return 0.0
        return float(self.coverage_count.sum()) / float(cov)
