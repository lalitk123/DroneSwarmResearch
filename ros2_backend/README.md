# swarm_sar — ROS2 + Gazebo backend

A high-fidelity ROS2 (Humble) + Gazebo backend that mirrors the lightweight Python
research simulator in [`../swarm_sim`](../swarm_sim). The **same three coordination
strategies** (centralized / decentralized / stigmergy) and the **same communication
failure models** (packet loss / range / dead zone / latency / composite) run here, so
results obtained in the fast Python harness can be validated in a physics-grade
simulator on the path to PX4/MAVROS hardware deployment.

> ROS2 is not required to read or lint this package — every node guards `import rclpy`
> via `swarm_sar/_rclpy_guard.py`, so all modules `py_compile` on a plain machine. ROS2
> + Gazebo is required only to actually *run* it.

## Why this exists
Per the project survey ([../CLAUDE.md](../CLAUDE.md)): ROS2 + Gazebo gives physical/SITL
fidelity but **has no built-in model of network degradation** — you bolt on a network
layer (e.g. ns-3). Here that network layer is a single node (`comm_bridge_node`) that
routes *every* inter-agent message through the exact same `CommModel` abstraction as the
Python sim, so degradation experiments are directly comparable across both backends.

## Packages
- **`swarm_sar`** (ament_python) — nodes, strategy adapters, comm models, launch, world, config.
- **`swarm_sar_interfaces`** (ament_cmake) — custom messages `CommMessage` and `DroneState`
  (a separate interfaces package because `rosidl` message generation needs ament_cmake).

## Architecture: Python sim → ROS2 mapping

| `swarm_sim` concept | ROS2 / Gazebo equivalent |
|---|---|
| `World` (grid, obstacles, victims) | `worlds/disaster.world` (Gazebo SDF) + victim ground-truth in `metrics_node` |
| `Drone` (pose, sensing, kinematics) | `drone_node` (one node/drone): `/<ns>/odom` in, `/<ns>/cmd_vel` out, geometric sensing |
| `Drone.move_towards` | velocity command (`geometry_msgs/Twist`) toward the strategy target |
| `CommModel` (perfect/loss/range/dead-zone/latency/composite) | `comm_bridge_node` + `swarm_sar/comm_models.py` — intercepts `/swarm/comm`, delivers only passing messages to `/swarm/comm/delivered/<id>` |
| `send(sender, receiver, payload)` | publish `CommMessage` on `/swarm/comm` (bridge applies the model) |
| `CoordinationStrategy` interface | `swarm_sar/strategies/base.py` + `centralized.py` / `decentralized.py` / `stigmergy.py` |
| `BaseStation` + `CentralizedStrategy` | `base_station_node` (global belief, greedy frontier assign, downlink subject to comm) |
| `Simulator` step loop | ROS2 executor + per-node rate timers (`control_period_s`, `plan_period_s`) |
| `metrics.py` (coverage, detection, connectivity, AUC) | `metrics_node` — `largest_connected_fraction` ported verbatim |
| `SimConfig` | `config/params.yaml` (ROS2 params, `/**` wildcard) |
| Multi-hop relay routing (`Simulator(multihop=True)`) | emergent from the real radio mesh; the bridge can be extended to path-based delivery |
| Node failure (`scheduled_failures`) | `DroneState.alive=false` / killing a `drone_node` |

## Prerequisites
- ROS2 **Humble** (or newer), `colcon`, `rosdep`
- **Gazebo** (Garden/Harmonic via `ros_gz`, or Ignition) with `ros_gz_sim`
- A quadrotor model (e.g. PX4 **x500**/iris, or a simple quadrotor URDF). For full SITL,
  PX4 + MAVROS (optional; the kinematic `cmd_vel` path works without it).
- Python `numpy`

## Build
```bash
# from a colcon workspace, e.g. ~/ros2_ws/src
ln -s /path/to/DroneSwarmResearch/ros2_backend/swarm_sar            .
ln -s /path/to/DroneSwarmResearch/ros2_backend/swarm_sar_interfaces .
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select swarm_sar_interfaces swarm_sar
source install/setup.bash
```

## Run
```bash
# perfect comms, 10 drones, centralized
ros2 launch swarm_sar swarm_sar.launch.py

# inject failures / change strategy by overriding params on the CLI
ros2 launch swarm_sar swarm_sar.launch.py params_file:=/abs/path/my_params.yaml
# or edit config/params.yaml: strategy / comm_model / severity / n_drones / ...
```
`metrics_node` logs coverage, detection rate, time-to-first-victim, connectivity
(largest connected component) and path length, mirroring the Python harness so the two
backends produce comparable degradation curves.

## What is functional vs stubbed
- **Functional logic / idiomatic structure:** node graph, topic wiring, `CommMessage`/
  `DroneState` schemas, the comm-degradation bridge, strategy adapters, metrics, launch,
  params, SDF world.
- **Stubbed / environment-dependent:** actual Gazebo model spawning and `/odom` feedback
  depend on the chosen drone model and your `ros_gz` version; the obstacle layout in
  `disaster.world` is a representative cluster (generate it from `SimConfig` for parametric
  maps). Strategy adapters port the *core* `assign()` logic; tune controller gains for the
  real vehicle dynamics.

## Path to hardware
Swap the kinematic `cmd_vel` integration for PX4 offboard control via **MAVROS**
(`/mavros/setpoint_velocity/cmd_vel`), keep `comm_bridge_node` (or replace it with a real
radio + ns-3 emulation), and the same strategies/metrics carry over unchanged.
