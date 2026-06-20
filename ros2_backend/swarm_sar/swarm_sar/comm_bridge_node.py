"""comm_bridge_node: the communication-degradation injection point.

This is the ROS2 analogue of how swarm_sim/simulator.py routes every message
through the active CommModel. ALL inter-drone and drone<->base messages are
published by the drones onto ``/swarm/comm``; this node intercepts them, evaluates
the configured CommModel (packet loss / range / dead zone / latency / composite),
and ONLY republishes the messages the model decides to deliver — onto
``/swarm/comm/delivered/<receiver_id>`` (or fan-out for broadcasts).

This models the *network layer* — the same role ns-3 plays when integrated with
ROS2/Gazebo (see CLAUDE.md: "ROS 2 + Gazebo ... has no built-in model of network
degradation. You bolt on a network layer (e.g. ns-3 via integration)"). Here that
network layer is a single lightweight node sharing the exact CommModel code path as
the Python simulator, so degradation results are directly comparable.

Range / dead-zone models need endpoint positions. The drones stamp their own pose
into ``sender_pos``; the bridge keeps a live table of every drone's last pose from
``/swarm/state`` so it can fill in the receiver position. Latency is realised by
scheduling delayed republication with a one-shot timer.
"""

from __future__ import annotations

import json
import random
from typing import Dict, Optional

from ._rclpy_guard import HAVE_RCLPY, Node, require_rclpy
from .comm_models import CommModel, Endpoint, build_comm_model

if HAVE_RCLPY:  # pragma: no cover - only inside ROS2
    import rclpy
    from rclpy.qos import QoSProfile
    from swarm_sar_interfaces.msg import CommMessage, DroneState

BROADCAST = -2
BASE_ID = -1


class CommBridgeNode(Node):
    def __init__(self):
        require_rclpy()
        super().__init__("comm_bridge_node")

        # ---- comm model from params -------------------------------------
        self.declare_parameter("comm_model", "perfect")
        self.declare_parameter("severity", 0.0)
        self.declare_parameter("range_radius", 12.0)
        self.declare_parameter("range_mode", "disk")
        self.declare_parameter("packet_loss_p", 0.0)
        self.declare_parameter("max_delay", 3)
        self.declare_parameter("seed", 0)
        self.declare_parameter("n_drones", 10)
        self.declare_parameter("control_period_s", 0.2)
        # dead zones passed as a flat list [y0,x0,y1,x1, y0,x0,y1,x1, ...]
        self.declare_parameter("dead_zones_flat", [0.0])

        spec = {
            "model": str(self.get_parameter("comm_model").value),
            "severity": float(self.get_parameter("severity").value),
            "range_radius": float(self.get_parameter("range_radius").value),
            "range_mode": str(self.get_parameter("range_mode").value),
            "packet_loss_p": float(self.get_parameter("packet_loss_p").value),
            "max_delay": int(self.get_parameter("max_delay").value),
            "seed": int(self.get_parameter("seed").value),
            "dead_zones": self._unpack_zones(
                self.get_parameter("dead_zones_flat").value),
        }
        self.rng = random.Random(spec["seed"])
        self.comm: CommModel = build_comm_model(spec, self.rng)
        self.control_period = float(self.get_parameter("control_period_s").value)
        self.n_drones = int(self.get_parameter("n_drones").value)

        # last known pose of every endpoint (id -> (y, x)); base at (0,0)
        self.poses: Dict[int, tuple] = {BASE_ID: (0.0, 0.0)}
        self.alive: Dict[int, bool] = {BASE_ID: True}
        self.tick = 0

        qos = QoSProfile(depth=50)
        self.create_subscription(CommMessage, "/swarm/comm", self._on_comm, qos)
        self.create_subscription(DroneState, "/swarm/state", self._on_state, qos)
        # per-receiver delivered topics
        self.delivered_pubs = {
            i: self.create_publisher(
                CommMessage, f"/swarm/comm/delivered/{i}", qos)
            for i in list(range(self.n_drones)) + [BASE_ID]
        }
        # advance a logical tick on the control period so latency is in "steps"
        self.create_timer(self.control_period, self._advance_tick)
        self.get_logger().info(f"comm_bridge up; model={self.comm.name}")

    @staticmethod
    def _unpack_zones(flat):
        flat = list(flat or [])
        zones = []
        for i in range(0, len(flat) - 3, 4):
            zones.append(tuple(flat[i:i + 4]))
        return zones

    def _advance_tick(self):
        self.tick += 1

    def _on_state(self, msg):
        self.poses[msg.drone_id] = (msg.pos.y, msg.pos.x)
        self.alive[msg.drone_id] = msg.alive

    def _endpoint(self, eid: int, fallback=None) -> Endpoint:
        pos = self.poses.get(eid, fallback or (0.0, 0.0))
        return Endpoint(pos=pos, alive=self.alive.get(eid, True))

    def _on_comm(self, msg):
        """Apply the CommModel and (maybe, after latency) republish."""
        sender = self._endpoint(msg.sender_id,
                                fallback=(msg.sender_pos.y, msg.sender_pos.x))
        receivers = (list(range(self.n_drones)) if msg.receiver_id == BROADCAST
                     else [msg.receiver_id])

        for rid in receivers:
            if rid == msg.sender_id:
                continue
            receiver = self._endpoint(rid,
                                      fallback=(msg.receiver_pos.y, msg.receiver_pos.x))
            latency = self.comm.deliver(sender, receiver, self.tick)
            if latency is None:
                continue  # DROPPED by the comm model
            if latency <= 0:
                self._deliver(rid, msg)
            else:
                # schedule delayed delivery (latency in control-period steps)
                delay_s = latency * self.control_period
                self._schedule(rid, msg, delay_s)

    def _schedule(self, rid: int, msg, delay_s: float):
        # one-shot timer that fires once then cancels itself
        timer = {"t": None}

        def _fire():
            self._deliver(rid, msg)
            timer["t"].cancel()

        timer["t"] = self.create_timer(delay_s, _fire)

    def _deliver(self, rid: int, src):
        pub = self.delivered_pubs.get(rid)
        if pub is None:
            return
        out = CommMessage()
        out.header = src.header
        out.sender_id = src.sender_id
        out.receiver_id = rid
        out.kind = src.kind
        out.sender_pos = src.sender_pos
        out.payload_json = src.payload_json
        pub.publish(out)


def main(args=None):
    require_rclpy()
    rclpy.init(args=args)
    node = CommBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
