"""base_station_node: central planner for the centralized strategy.

Only relevant when ``strategy == centralized`` (mirrors the BaseStation in
swarm_sim/simulator.py + CentralizedStrategy). It:
  * holds the GLOBAL coverage belief, aggregated from drone uplinks that the
    comm_bridge actually delivers (``/swarm/comm/delivered/-1``);
  * computes greedy nearest-frontier assignments with duplicate-target penalty;
  * publishes each assignment as a CommMessage on ``/swarm/comm`` addressed to the
    target drone — so it, too, is subject to the comm_bridge's CommModel. When the
    base->drone downlink drops, the drone never gets its new target and idles. This
    is the centralized failure mode the project is built to expose.

For decentralized / stigmergy strategies this node is inert (it simply parks at the
base position) and can be omitted from the launch; it is included so the launch file
is strategy-agnostic.
"""

from __future__ import annotations

import json
from typing import Dict

import numpy as np

from ._rclpy_guard import HAVE_RCLPY, Node, require_rclpy
from .strategies.base import frontier_cells

if HAVE_RCLPY:  # pragma: no cover - only inside ROS2
    import rclpy
    from rclpy.qos import QoSProfile
    from swarm_sar_interfaces.msg import CommMessage, DroneState

BASE_ID = -1


class BaseStationNode(Node):
    def __init__(self):
        require_rclpy()
        super().__init__("base_station_node")

        self.declare_parameter("strategy", "centralized")
        self.declare_parameter("map_height", 50)
        self.declare_parameter("map_width", 50)
        self.declare_parameter("plan_period_s", 0.2)

        self.strategy = str(self.get_parameter("strategy").value)
        h = int(self.get_parameter("map_height").value)
        w = int(self.get_parameter("map_width").value)
        self.global_covered = np.zeros((h, w), dtype=bool)
        self.grid = np.zeros((h, w), dtype=np.int8)
        self.drone_pos: Dict[int, np.ndarray] = {}
        self.drone_alive: Dict[int, bool] = {}

        qos = QoSProfile(depth=50)
        self.comm_pub = self.create_publisher(CommMessage, "/swarm/comm", qos)
        self.create_subscription(DroneState, "/swarm/state", self._on_state, qos)
        # uplinked maps that survived the comm_bridge arrive here
        self.create_subscription(
            CommMessage, f"/swarm/comm/delivered/{BASE_ID}", self._on_uplink, qos)

        if self.strategy == "centralized":
            self.create_timer(
                float(self.get_parameter("plan_period_s").value), self._plan)
            self.get_logger().info("base_station: centralized planner active")
        else:
            self.get_logger().info(
                f"base_station: inert for strategy={self.strategy}")

    def _on_state(self, msg):
        self.drone_pos[msg.drone_id] = np.array([msg.pos.y, msg.pos.x], dtype=float)
        self.drone_alive[msg.drone_id] = msg.alive

    def _on_uplink(self, msg):
        # Payload carries a list of newly-covered cells from the drone's belief.
        payload = json.loads(msg.payload_json or "{}")
        for (y, x) in payload.get("covered", []):
            self.global_covered[int(y), int(x)] = True

    def _plan(self):
        cells = frontier_cells(self.global_covered, self.grid)
        if len(cells) == 0:
            return
        living = [i for i, a in self.drone_alive.items() if a]
        taken = []
        for did in living:
            p = self.drone_pos[did]
            d2 = ((cells[:, 0] - p[0]) ** 2 + (cells[:, 1] - p[1]) ** 2).astype(float)
            for ty, tx in taken:
                d2 += 50.0 * np.exp(-(((cells[:, 0] - ty) ** 2 +
                                       (cells[:, 1] - tx) ** 2)) / 9.0)
            best = cells[int(np.argmin(d2))]
            taken.append((best[0], best[1]))
            self._publish_assign(did, int(best[0]), int(best[1]))

    def _publish_assign(self, drone_id: int, ty: int, tx: int):
        msg = CommMessage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.sender_id = BASE_ID
        msg.receiver_id = drone_id
        msg.kind = "assign"
        msg.sender_pos.y = 0.0
        msg.sender_pos.x = 0.0
        if drone_id in self.drone_pos:
            msg.receiver_pos.y = float(self.drone_pos[drone_id][0])
            msg.receiver_pos.x = float(self.drone_pos[drone_id][1])
        msg.payload_json = json.dumps({"target": [ty, tx]})
        self.comm_pub.publish(msg)


def main(args=None):
    require_rclpy()
    rclpy.init(args=args)
    node = BaseStationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
