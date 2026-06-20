"""drone_node: one ROS2 node per drone.

Responsibilities (mirrors swarm_sim/drone.py + the per-drone slice of the
strategy loop):
  * Subscribe to this drone's pose from Gazebo (``/<ns>/odom``, nav_msgs/Odometry).
  * Maintain a local coverage belief + known-victim list by "sensing" cells within
    ``sensing_radius`` of the current pose (Gazebo would supply real sensor data;
    here we mark coverage geometrically, exactly as swarm_sim does).
  * Run the pluggable coordination strategy adapter on a fixed-rate control timer.
  * Publish a velocity command (``/<ns>/cmd_vel``, geometry_msgs/Twist) toward the
    strategy's target cell — the kinematic "move_towards" of swarm_sim/drone.py.
  * Send/receive inter-drone messages on ``/swarm/comm`` (CommMessage). Outgoing
    messages go to the comm_bridge_node, which applies the CommModel and republishes
    only the delivered ones onto ``/swarm/comm/delivered/<id>`` for this drone.
  * Publish its DroneState on ``/swarm/state`` for the metrics / base-station nodes.

All ROS2 use is guarded so the module imports without rclpy.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

from ._rclpy_guard import HAVE_RCLPY, Node, require_rclpy
from .strategies import build_strategy
from .strategies.base import DroneView, FREE

if HAVE_RCLPY:  # pragma: no cover - only inside ROS2
    import rclpy
    from rclpy.qos import QoSProfile
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from swarm_sar_interfaces.msg import CommMessage, DroneState


class DroneNode(Node):
    """One drone. Namespaced as ``drone_<id>`` by the launch file."""

    def __init__(self):
        require_rclpy()
        super().__init__("drone_node")

        # ---- parameters (from params.yaml via the launch file) -----------
        self.declare_parameter("drone_id", 0)
        self.declare_parameter("strategy", "decentralized")
        self.declare_parameter("map_height", 50)
        self.declare_parameter("map_width", 50)
        self.declare_parameter("sensing_radius", 3.0)
        self.declare_parameter("speed", 1.0)
        self.declare_parameter("control_period_s", 0.2)

        self.drone_id = int(self.get_parameter("drone_id").value)
        h = int(self.get_parameter("map_height").value)
        w = int(self.get_parameter("map_width").value)
        self.sensing_radius = float(self.get_parameter("sensing_radius").value)
        self.speed = float(self.get_parameter("speed").value)

        # ---- world model / local belief ---------------------------------
        # Obstacle grid is loaded from the same world the Gazebo SDF was generated
        # from (a companion .npy beside the .world); here we assume all-free until
        # that map is published. The metrics node owns ground truth.
        self.grid = np.zeros((h, w), dtype=np.int8)
        self.view = DroneView(
            id=self.drone_id,
            pos=np.zeros(2, dtype=float),
            known_covered=np.zeros((h, w), dtype=bool),
        )

        # ---- strategy adapter -------------------------------------------
        strat_name = str(self.get_parameter("strategy").value)
        self.strategy = build_strategy(strat_name, seed=self.drone_id)
        self.strategy.reset(self.grid, [self.view], self._base_pos())

        # ---- pub/sub -----------------------------------------------------
        qos = QoSProfile(depth=10)
        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", qos)
        self.state_pub = self.create_publisher(DroneState, "/swarm/state", qos)
        self.comm_pub = self.create_publisher(CommMessage, "/swarm/comm", qos)

        self.create_subscription(Odometry, "odom", self._on_odom, qos)
        # delivered messages for THIS drone, republished by the comm_bridge
        self.create_subscription(
            CommMessage, f"/swarm/comm/delivered/{self.drone_id}",
            self._on_comm, qos)

        self.tick = 0
        period = float(self.get_parameter("control_period_s").value)
        self.create_timer(period, self._control_loop)
        self.get_logger().info(
            f"drone {self.drone_id} up; strategy={strat_name}")

    # ---- callbacks -------------------------------------------------------
    def _base_pos(self):
        return (0, 0)

    def _on_odom(self, msg):
        # Gazebo world frame -> grid (y, x). z ignored in the 2.5D model.
        self.view.pos = np.array([msg.pose.pose.position.y,
                                  msg.pose.pose.position.x], dtype=float)

    def _on_comm(self, msg):
        # A message the comm_bridge decided to deliver to us.
        self.view.inbox.append({
            "sender": msg.sender_id, "kind": msg.kind,
            "payload": json.loads(msg.payload_json or "{}"),
        })

    # ---- main control loop ----------------------------------------------
    def _control_loop(self):
        self.tick += 1
        self._sense()

        # The strategy sets self.view.target and returns messages to publish.
        # In the per-drone distributed deployment `send` just enqueues a
        # CommMessage (delivery is decided downstream by the bridge); we treat
        # the publish as "attempted" and let the bridge filter. For the
        # co-simulation comparison the orchestrator node passes a real boolean.
        outgoing = self.strategy.step(
            self.grid, [self.view], self._base_pos(), self._send, self.tick)
        for m in outgoing:
            self._publish_comm(m)

        self._drive_toward_target()
        self._publish_state()
        self.view.inbox.clear()

    def _sense(self):
        """Mark coverage + detect victims within sensing_radius (geometric).

        Mirrors swarm_sim/drone.py::_mark_local_coverage. Real victim detection
        would come from a Gazebo sensor; the metrics node holds victim ground truth
        and credits a detection when this drone reports coverage of a victim cell.
        """
        r = int(np.ceil(self.sensing_radius))
        y0, x0 = self.view.pos
        h, w = self.grid.shape
        ylo, yhi = max(0, int(y0) - r), min(h, int(y0) + r + 1)
        xlo, xhi = max(0, int(x0) - r), min(w, int(x0) + r + 1)
        ys, xs = np.mgrid[ylo:yhi, xlo:xhi]
        d2 = (ys - y0) ** 2 + (xs - x0) ** 2
        self.view.known_covered[ylo:yhi, xlo:xhi] |= (d2 <= self.sensing_radius ** 2)

    def _send(self, sender_id, receiver_id, kind, payload) -> bool:
        """Enqueue an outgoing message. Returns True (attempted).

        Delivery success is determined by the comm_bridge in the distributed
        deployment, so from a single drone's view a publish always "attempts".
        The co-simulation orchestrator overrides this with a synchronous bridge
        call for an exact swarm_sim-style boolean.
        """
        self._publish_comm({"sender": sender_id, "receiver": receiver_id,
                            "kind": kind, **(payload or {})})
        return True

    def _publish_comm(self, m: dict):
        msg = CommMessage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.sender_id = int(m.get("sender", self.drone_id))
        msg.receiver_id = int(m.get("receiver", -2))
        msg.kind = str(m.get("kind", ""))
        msg.sender_pos.y = float(self.view.pos[0])
        msg.sender_pos.x = float(self.view.pos[1])
        payload = {k: v for k, v in m.items()
                   if k not in ("sender", "receiver", "kind")}
        msg.payload_json = json.dumps(payload)
        self.comm_pub.publish(msg)

    def _drive_toward_target(self):
        """Proportional velocity command toward the target cell (kinematic)."""
        if self.view.target is None:
            self.cmd_pub.publish(Twist())
            return
        target = np.array(self.view.target, dtype=float)
        delta = target - self.view.pos
        dist = float(np.linalg.norm(delta))
        cmd = Twist()
        if dist > 1e-3:
            v = delta / dist * min(self.speed, dist)
            cmd.linear.y = float(v[0])   # grid-y
            cmd.linear.x = float(v[1])   # grid-x
        self.cmd_pub.publish(cmd)

    def _publish_state(self):
        st = DroneState()
        st.header.stamp = self.get_clock().now().to_msg()
        st.drone_id = self.drone_id
        st.alive = self.view.alive
        st.pos.y = float(self.view.pos[0])
        st.pos.x = float(self.view.pos[1])
        if self.view.target is not None:
            st.target.y = float(self.view.target[0])
            st.target.x = float(self.view.target[1])
        st.path_length = float(getattr(self.view, "path_length", 0.0))
        st.covered_cell_count = int(self.view.known_covered.sum())
        st.known_victim_count = int(len(self.view.known_victims))
        self.state_pub.publish(st)


def main(args=None):
    require_rclpy()
    rclpy.init(args=args)
    node = DroneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
