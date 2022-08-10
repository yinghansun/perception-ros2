import time

import rclpy
from rclpy.node import Node


class HelloROS2(Node):
    def __init__(self, name: str):
        super().__init__(name)
        while rclpy.ok():
            self.get_logger().info("Hello ROS2")
            time.sleep(0.5)

def main(args=None):
    rclpy.init(args=args)
    node = HelloROS2("hello_ros2_node")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()