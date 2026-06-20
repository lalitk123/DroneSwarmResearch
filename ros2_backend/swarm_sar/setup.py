"""ament_python setup for the swarm_sar package.

Installs the ROS2 nodes (drone / comm_bridge / base_station / metrics), the
strategy adapter submodule, and the launch / world / config data files so that
`ros2 launch swarm_sar swarm_sar.launch.py` resolves everything from the install
space.
"""

import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'swarm_sar'

setup(
    name=package_name,
    version='0.1.0',
    # include the strategies submodule package as well
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament resource index marker
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        # package manifest
        ('share/' + package_name, ['package.xml']),
        # launch files
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
        # gazebo worlds
        (os.path.join('share', package_name, 'worlds'),
         glob('worlds/*.world')),
        # parameter / config files
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='DroneSwarmResearch',
    maintainer_email='vaishik@gmail.com',
    description='ROS2 + Gazebo backend for resilient drone-swarm SAR under comm failure.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # `ros2 run swarm_sar <name>` -> module:main
            'drone_node = swarm_sar.drone_node:main',
            'comm_bridge_node = swarm_sar.comm_bridge_node:main',
            'base_station_node = swarm_sar.base_station_node:main',
            'metrics_node = swarm_sar.metrics_node:main',
        ],
    },
)
