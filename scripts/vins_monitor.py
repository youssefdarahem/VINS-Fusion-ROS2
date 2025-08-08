#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud
import time


class VINSMonitor(Node):
    def __init__(self):
        super().__init__('vins_monitor')

        # Subscribe to VINS output topics
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry', self.odom_callback, 10)
        self.path_sub = self.create_subscription(
            Path, '/path', self.path_callback, 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/camera_pose', self.pose_callback, 10)
        self.points_sub = self.create_subscription(
            PointCloud, '/point_cloud', self.points_callback, 10)

        # Counters
        self.odom_count = 0
        self.path_count = 0
        self.pose_count = 0
        self.points_count = 0

        # Timer for status updates
        self.timer = self.create_timer(2.0, self.print_status)
        self.start_time = time.time()

        self.get_logger().info("VINS Monitor started. Listening for VINS output...")

    def odom_callback(self, msg):
        self.odom_count += 1
        if self.odom_count == 1:
            self.get_logger().info("✅ First odometry message received!")

    def path_callback(self, msg):
        self.path_count += 1
        if self.path_count == 1:
            self.get_logger().info("✅ First path message received!")

    def pose_callback(self, msg):
        self.pose_count += 1
        if self.pose_count == 1:
            self.get_logger().info("✅ First camera pose message received!")

    def points_callback(self, msg):
        self.points_count += 1
        if self.points_count == 1:
            self.get_logger().info("✅ First point cloud message received!")

    def print_status(self):
        runtime = time.time() - self.start_time
        self.get_logger().info(
            f"Runtime: {runtime:.1f}s | "
            f"Odom: {self.odom_count} | Path: {self.path_count} | "
            f"Pose: {self.pose_count} | Points: {self.points_count}"
        )


def main():
    rclpy.init()
    monitor = VINSMonitor()

    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        print("\nMonitor stopped by user")
    finally:
        monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
