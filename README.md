# Perception-ROS2



## ROS2
Install ROS2:
~~~
$ sudo apt update && sudo apt install curl gnupg lsb-release 
$ sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg 
$ echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

$ sudo apt update
$ sudo apt upgrade
$ sudo apt install ros-humble-desktop

$ source /opt/ros/humble/setup.bash
$ echo " source /opt/ros/humble/setup.bash" >> ~/.bashrc
~~~

Test ROS2: open 2 terminal and run the two command below, respectively.
~~~
ros2 run demo_nodes_cpp talker
ros2 run demo_nodes_py listener
~~~

If you can read Chinese, [here](https://book.guyuehome.com/) is a good tutorial for ROS2.