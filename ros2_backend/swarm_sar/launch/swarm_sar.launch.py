"""Top-level launch for the swarm_sar ROS2 backend.

Brings up, all parameterised from ``config/params.yaml``:
  * Gazebo with ``worlds/disaster.world`` (obstacles + ground plane),
  * N drone_node instances (one per drone, each in its own ``drone_<i>`` namespace,
    each spawned as a quadrotor model in Gazebo),
  * the comm_bridge_node (network-degradation layer),
  * the base_station_node (centralized planner; inert for other strategies),
  * the metrics_node.

Usage:
    ros2 launch swarm_sar swarm_sar.launch.py
    ros2 launch swarm_sar swarm_sar.launch.py n_drones:=20 strategy:=stigmergy \
        comm_model:=range range_radius:=6.0

The N drones are created from the ``n_drones`` parameter so the swarm scales the
same way swarm_sim's SimConfig.n_drones does.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("swarm_sar")
    default_params = os.path.join(pkg, "config", "params.yaml")
    world_path = os.path.join(pkg, "worlds", "disaster.world")

    # ---- launch arguments (override params.yaml on the CLI) -------------
    args = [
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("n_drones", default_value="10"),
        DeclareLaunchArgument("strategy", default_value="decentralized"),
        DeclareLaunchArgument("comm_model", default_value="perfect"),
        DeclareLaunchArgument("range_radius", default_value="12.0"),
        DeclareLaunchArgument("use_gazebo", default_value="true"),
    ]

    params_file = LaunchConfiguration("params_file")
    n_drones = LaunchConfiguration("n_drones")
    strategy = LaunchConfiguration("strategy")
    comm_model = LaunchConfiguration("comm_model")
    range_radius = LaunchConfiguration("range_radius")

    nodes = []

    # ---- Gazebo (Classic 11 via gazebo_ros on Humble) -------------------
    # For newer Gazebo (Garden/Harmonic) swap this for the ros_gz_sim launch.
    gazebo_launch = os.path.join(
        get_package_share_directory("gazebo_ros"), "launch", "gazebo.launch.py")
    nodes.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch),
        launch_arguments={"world": world_path}.items(),
    ))

    common = [params_file,
              {"n_drones": n_drones, "strategy": strategy,
               "comm_model": comm_model, "range_radius": range_radius}]

    # ---- infrastructure nodes -------------------------------------------
    nodes.append(Node(package="swarm_sar", executable="comm_bridge_node",
                      name="comm_bridge", output="screen", parameters=common))
    nodes.append(Node(package="swarm_sar", executable="base_station_node",
                      name="base_station", output="screen", parameters=common))
    nodes.append(Node(package="swarm_sar", executable="metrics_node",
                      name="metrics", output="screen", parameters=common))

    # ---- N drones -------------------------------------------------------
    # OpaqueFunction would let us read n_drones at runtime; for a static, readable
    # scaffold we spawn a fixed pool sized by MAX_DRONES and disable extras via the
    # `drone_id < n_drones` guard inside each node would over-complicate things, so
    # we instead expand here against a max and rely on the launch arg default. To
    # keep the file declarative we spawn MAX_DRONES nodes; set n_drones to match.
    MAX_DRONES = 50
    for i in range(MAX_DRONES):
        # only spawn drone i if i < n_drones (evaluated at launch)
        condition_expr = PythonExpression([str(i), " < ", n_drones])
        from launch.conditions import IfCondition
        drone_params = common + [{"drone_id": i}]
        nodes.append(Node(
            package="swarm_sar", executable="drone_node",
            namespace=PythonExpression(["'drone_", str(i), "'"]),
            name="drone_node", output="screen",
            parameters=drone_params,
            condition=IfCondition(condition_expr),
        ))
        # NOTE: each drone should also be spawned into Gazebo via spawn_entity.py
        # with the quadrotor SDF/URDF at a start pose near the base. That spawn
        # action is documented in the README ("Spawning drone models") and omitted
        # here to keep the launch focused on the coordination/comm scaffold.

    return LaunchDescription(args + nodes)
