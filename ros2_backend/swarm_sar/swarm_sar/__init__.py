"""swarm_sar: ROS2 + Gazebo backend for resilient drone-swarm SAR.

This package mirrors the lightweight Python simulator in ``swarm_sim/`` onto
ROS2 nodes so the same coordination strategies and communication-failure models
can run in a high-fidelity (Gazebo / PX4) simulator.

The ``rclpy`` import is guarded throughout so that these modules can be imported
(for testing, linting, or documentation) on a machine without ROS2 installed.
"""

__version__ = "0.1.0"
