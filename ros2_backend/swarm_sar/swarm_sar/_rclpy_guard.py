"""Centralised guard for the optional ``rclpy`` dependency.

ROS2 (``rclpy``) is only present inside a sourced ROS2 environment. To keep every
node module importable on a plain machine (so they can be unit-tested, linted, and
``py_compile``-checked without ROS2), we import rclpy here behind a try/except and
expose a ``HAVE_RCLPY`` flag plus a ``require_rclpy()`` helper.

Node ``main()`` functions call ``require_rclpy()`` first; the pure-Python strategy
logic and comm models do not need ROS2 at all and import nothing from here.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised only inside a ROS2 environment
    import rclpy  # noqa: F401
    from rclpy.node import Node  # noqa: F401
    HAVE_RCLPY = True
    _IMPORT_ERROR = None
except Exception as exc:  # ImportError on non-ROS2 machines
    HAVE_RCLPY = False
    _IMPORT_ERROR = exc

    class Node:  # type: ignore
        """Stand-in so ``class FooNode(Node)`` is still a valid definition.

        Instantiating it raises, so importing a module is always safe but running
        a node without ROS2 fails loudly with an actionable message.
        """

        def __init__(self, *args, **kwargs):
            require_rclpy()


def require_rclpy() -> None:
    """Raise a clear error if ROS2 / rclpy is not available."""
    if not HAVE_RCLPY:
        raise NotImplementedError(
            "rclpy is not available — this node must be run inside a sourced "
            "ROS2 (Humble) environment, e.g. `source /opt/ros/humble/setup.bash` "
            "then `colcon build` and `ros2 launch swarm_sar swarm_sar.launch.py`. "
            f"Original import error: {_IMPORT_ERROR!r}"
        )
