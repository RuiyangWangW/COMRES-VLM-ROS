from glob import glob

from setuptools import setup

package_name = "comres_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ruiyang Wang",
    maintainer_email="ruiyang.wang@duke.edu",
    description="COMRES on real robots: the central node.",
    license="TODO",
    entry_points={
        "console_scripts": [
            "comres_node = comres_ros.comres_node:main",
        ],
    },
)
