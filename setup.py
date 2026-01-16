from setuptools import setup
import os
from glob import glob

package_name = 'orb_slam_py'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Farhan',
    maintainer_email='farhan@example.com',
    description='Python-based Monocular ORB-SLAM for ROS 2 Humble',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'orb_slam_node = orb_slam_py.node:main',
            'stereo_orb_slam_node = orb_slam_py.stereo_node:main',
            'kitti_camera_info = orb_slam_py.kitti_camera_info:main',
        ],
    },
)
