from setuptools import setup

package_name = 'test_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yinghan Sun',
    maintainer_email='yinghansun2@gmail.com',
    description='first package for perception-ros2 to make test',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hello_ros2_node = test_package.hello_ros2:main'
        ],
    },
)
