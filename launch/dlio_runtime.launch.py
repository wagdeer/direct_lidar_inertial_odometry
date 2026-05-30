#
#   Copyright (c)     
#
#   The Verifiable & Control-Theoretic Robotics (VECTR) Lab
#   University of California, Los Angeles
#
#   Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez
#   Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu
#
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression, PathJoinSubstitution
from launch_ros.actions import PushRosNamespace

def generate_launch_description():
    # Launch the Livox MID360 driver
    livox_dir = get_package_share_directory('livox_ros_driver2')
    livox_launch_file = os.path.join(livox_dir, 'launch_ROS2', 'msg_MID360_launch.py')
    livox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(livox_launch_file)
    )

    # Launch the Direct Lidar Inertial Odometry (DLIO) node
    dlio_dir = get_package_share_directory('direct_lidar_inertial_odometry')
    dlio_launch_file = os.path.join(dlio_dir, 'launch', 'dlio.launch.py')
    dlio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(dlio_launch_file),
        launch_arguments={
            'rviz': 'false'
        }.items()
    )

    return LaunchDescription([
        livox_launch,
        dlio_launch
    ])