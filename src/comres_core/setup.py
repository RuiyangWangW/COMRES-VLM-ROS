from setuptools import setup

package_name = "comres_core"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name, package_name + ".algo"],
    package_data={package_name + ".algo": ["CORE_VERSION"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "scipy", "scikit-image", "pillow", "openai"],
    zip_safe=True,
    maintainer="Ruiyang Wang",
    maintainer_email="ruiyang.wang@duke.edu",
    description="The COMRES algorithm and its real-time engine (no ROS dependency).",
    license="TODO",
    tests_require=["pytest"],
)
