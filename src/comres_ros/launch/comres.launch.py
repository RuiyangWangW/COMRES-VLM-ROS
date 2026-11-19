"""The central COMRES node and its link to the dogs (dog_link: the dogs run dog_bridge.py).

    ros2 launch comres_ros comres.launch.py [config:=<yaml>] [hint:="..."] [vlm_base_url:=http://host:18765/v1]
        [dog_link:=false]

hint and vlm_base_url override the config's when given (taken as plain text). dog_link:=false
when the robots' topics reach this computer directly (the same ROS 2 version). A config with a
map_align section (several robots, comres_two_dogs.yaml) starts map_align too: it places the
robots' maps in world_frame."""
import os

import yaml

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
    nodes = [Node(package="comres_ros", executable="comres_node", name="comres_node", output="screen",
                  parameters=params)]
    if LaunchConfiguration("dog_link").perform(context) == "true":
        nodes.append(Node(package="comres_ros", executable="comres_dog_link", name="dog_link", output="screen",
                          parameters=params[:1]))
    with open(params[0]) as f:
        if "map_align" in (yaml.safe_load(f) or {}):
            nodes.append(Node(package="comres_ros", executable="comres_map_align", name="map_align", output="screen",
                              parameters=params[:1]))
    return nodes


def generate_launch_description():
    default = os.path.join(get_package_share_directory("comres_ros"), "config", "comres.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default),
        DeclareLaunchArgument("hint", default_value=""),
        DeclareLaunchArgument("vlm_base_url", default_value=""),
        DeclareLaunchArgument("dog_link", default_value="true", choices=["true", "false"]),
        OpaqueFunction(function=_node),
    ])
