"""metrics_node: live mission metrics, mirroring swarm_sim/metrics.py.

Subscribes to every drone's DroneState (``/swarm/state``) and holds the victim
ground-truth (loaded from the same scenario that generated the Gazebo world). It
computes and periodically logs / publishes:
  * area coverage %        (union of per-drone covered counts vs free cells)
  * victim detection rate  (victims within sensing_radius of any live drone)
  * time-to-first-victim   (first tick a victim is detected)
  * mission_time           (tick when all victims detected -> episode can end)
  * largest-connected-component fraction (the connectivity metric)
  * total path length      (energy proxy)

``largest_connected_fraction`` is ported verbatim from swarm_sim/metrics.py so the
connectivity numbers are computed identically. Results are logged and also written
to ``/tmp/swarm_sar_metrics.json`` for offline degradation-curve plotting alongside
the swarm_sim ``results/matrix.csv``.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Tuple

import numpy as np

from ._rclpy_guard import HAVE_RCLPY, Node, require_rclpy

if HAVE_RCLPY:  # pragma: no cover - only inside ROS2
    import rclpy
    from rclpy.qos import QoSProfile
    from std_msgs.msg import Float64MultiArray
    from swarm_sar_interfaces.msg import DroneState


def largest_connected_fraction(positions, alive, comm_radius) -> float:
    """Fraction of living drones in the largest comm-connected component.

    Verbatim port of swarm_sim/metrics.py::largest_connected_fraction.
    """
    idx = [i for i, a in enumerate(alive) if a]
    n = len(idx)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    pos = np.array([positions[i] for i in idx])
    dy = pos[:, 0:1] - pos[:, 0:1].T
    dx = pos[:, 1:2] - pos[:, 1:2].T
    adj = (dy ** 2 + dx ** 2) <= comm_radius ** 2
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


class MetricsNode(Node):
    def __init__(self):
        require_rclpy()
        super().__init__("metrics_node")

        self.declare_parameter("map_height", 50)
        self.declare_parameter("map_width", 50)
        self.declare_parameter("n_drones", 10)
        self.declare_parameter("n_victims", 15)
        self.declare_parameter("sensing_radius", 3.0)
        self.declare_parameter("comm_radius", 12.0)
        self.declare_parameter("report_period_s", 1.0)
        # victim ground-truth as a flat [y0,x0, y1,x1, ...] list from params.yaml
        self.declare_parameter("victims_flat", [0.0])

        self.h = int(self.get_parameter("map_height").value)
        self.w = int(self.get_parameter("map_width").value)
        self.n_drones = int(self.get_parameter("n_drones").value)
        self.sensing_radius = float(self.get_parameter("sensing_radius").value)
        self.comm_radius = float(self.get_parameter("comm_radius").value)
        self.victims = self._unpack_victims(
            self.get_parameter("victims_flat").value)
        self.victim_found = np.zeros(len(self.victims), dtype=bool)
        self.victim_found_tick = np.full(len(self.victims), -1, dtype=int)

        self.n_free = self.h * self.w   # refined if an obstacle map is supplied
        self.covered = np.zeros((self.h, self.w), dtype=bool)
        self.pos: Dict[int, np.ndarray] = {}
        self.alive: Dict[int, bool] = {}
        self.path_len: Dict[int, float] = {}
        self.tick = 0
        self.t0 = time.time()
        self.first_victim_tick = -1
        self.mission_tick = -1

        qos = QoSProfile(depth=50)
        self.create_subscription(DroneState, "/swarm/state", self._on_state, qos)
        self.summary_pub = self.create_publisher(
            Float64MultiArray, "/swarm/metrics", qos)
        self.create_timer(
            float(self.get_parameter("report_period_s").value), self._report)
        self.get_logger().info(
            f"metrics up; {len(self.victims)} victims, {self.n_drones} drones")

    @staticmethod
    def _unpack_victims(flat) -> List[Tuple[float, float]]:
        flat = list(flat or [])
        return [(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]

    def _on_state(self, msg):
        p = np.array([msg.pos.y, msg.pos.x], dtype=float)
        self.pos[msg.drone_id] = p
        self.alive[msg.drone_id] = msg.alive
        self.path_len[msg.drone_id] = msg.path_length
        if not msg.alive:
            return
        # mark coverage + detect victims geometrically (ground truth lives here)
        self._mark_coverage(p)
        self._detect_victims(p)

    def _mark_coverage(self, p):
        r = int(np.ceil(self.sensing_radius))
        y0, x0 = p
        ylo, yhi = max(0, int(y0) - r), min(self.h, int(y0) + r + 1)
        xlo, xhi = max(0, int(x0) - r), min(self.w, int(x0) + r + 1)
        ys, xs = np.mgrid[ylo:yhi, xlo:xhi]
        d2 = (ys - y0) ** 2 + (xs - x0) ** 2
        self.covered[ylo:yhi, xlo:xhi] |= (d2 <= self.sensing_radius ** 2)

    def _detect_victims(self, p):
        if not len(self.victims):
            return
        v = np.array(self.victims, dtype=float)
        d2 = (v[:, 0] - p[0]) ** 2 + (v[:, 1] - p[1]) ** 2
        hit = (d2 <= self.sensing_radius ** 2) & (~self.victim_found)
        for i in np.argwhere(hit).ravel():
            self.victim_found[i] = True
            self.victim_found_tick[i] = self.tick
            if self.first_victim_tick < 0:
                self.first_victim_tick = self.tick

    def _report(self):
        self.tick += 1
        coverage = float(self.covered.sum()) / max(1, self.n_free)
        det = (float(self.victim_found.sum()) / len(self.victims)
               if len(self.victims) else 1.0)
        ids = sorted(self.pos)
        conn = largest_connected_fraction(
            [self.pos[i] for i in ids],
            [self.alive.get(i, False) for i in ids],
            self.comm_radius) if ids else 0.0
        total_path = float(sum(self.path_len.values()))

        if self.mission_tick < 0 and len(self.victims) and self.victim_found.all():
            self.mission_tick = self.tick
            self.get_logger().info(f"MISSION COMPLETE at tick {self.tick}")

        self.get_logger().info(
            f"[t={self.tick}] coverage={coverage:.3f} detection={det:.3f} "
            f"connectivity={conn:.3f} path_len={total_path:.1f}")

        out = Float64MultiArray()
        out.data = [float(self.tick), coverage, det, conn, total_path]
        self.summary_pub.publish(out)
        self._dump_json(coverage, det, conn, total_path)

    def _dump_json(self, coverage, det, conn, total_path):
        try:
            with open("/tmp/swarm_sar_metrics.json", "w") as fh:
                json.dump({
                    "tick": self.tick,
                    "coverage": coverage,
                    "detection_rate": det,
                    "mean_connectivity": conn,
                    "total_path_length": total_path,
                    "time_to_first_victim": int(self.first_victim_tick),
                    "mission_time": int(self.mission_tick),
                    "n_alive": int(sum(self.alive.values())),
                }, fh, indent=2)
        except OSError:
            pass


def main(args=None):
    require_rclpy()
    rclpy.init(args=args)
    node = MetricsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
