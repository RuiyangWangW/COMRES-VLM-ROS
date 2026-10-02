"""The central COMRES node.

    ros2 launch comres_ros comres.launch.py [config:=<yaml>] [hint:="..."] [vlm_base_url:=http://host:18765/v1]

hint and vlm_base_url override the config's when given (taken as plain text)."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _node(context):
    params = [LaunchConfiguration("config").perform(context)]
    extra = {}
    for k in ("hint", "vlm_base_url"):
        v = LaunchConfiguration(k).perform(context)
        if v:
            extra[k] = ParameterValue(v, value_type=str)
    if extra:
        params.append(extra)
    return [Node(package="comres_ros", executable="comres_node", name="comres_node", output="screen",
                 parameters=params)]


def generate_launch_description():
    default = os.path.join(get_package_share_directory("comres_ros"), "config", "comres.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default),
        DeclareLaunchArgument("hint", default_value=""),
        DeclareLaunchArgument("vlm_base_url", default_value=""),
        OpaqueFunction(function=_node),
    ])
