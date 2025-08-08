#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
import csv
import os
from datetime import datetime


class VINSLogger(Node):
    def __init__(self):
        super().__init__('vins_logger')

        # Create output directory
        self.output_dir = "/tmp/vins_output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Create output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.odom_file = open(
            f"{self.output_dir}/odometry_{timestamp}.csv", 'w', newline='')
        self.path_file = open(
            f"{self.output_dir}/path_{timestamp}.csv", 'w', newline='')
        self.camera_pose_file = open(
            f"{self.output_dir}/camera_pose_{timestamp}.csv", 'w', newline='')

        # CSV writers
        self.odom_writer = csv.writer(self.odom_file)
        self.path_writer = csv.writer(self.path_file)
        self.camera_pose_writer = csv.writer(self.camera_pose_file)

        # Write headers
        self.odom_writer.writerow(
            ['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        self.path_writer.writerow(
            ['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        self.camera_pose_writer.writerow(
            ['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry', self.odom_callback, 10)
        self.path_sub = self.create_subscription(
            Path, '/path', self.path_callback, 10)
        self.camera_pose_sub = self.create_subscription(
            PoseStamped, '/camera_pose', self.camera_pose_callback, 10)

        self.get_logger().info(
            f"VINS Logger started. Saving to {self.output_dir}")

    def odom_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        self.odom_writer.writerow([
            timestamp, pos.x, pos.y, pos.z,
            ori.x, ori.y, ori.z, ori.w
        ])
        self.odom_file.flush()

    def path_callback(self, msg):
        for pose_stamped in msg.poses:
            timestamp = pose_stamped.header.stamp.sec + \
                pose_stamped.header.stamp.nanosec * 1e-9
            pos = pose_stamped.pose.position
            ori = pose_stamped.pose.orientation

            self.path_writer.writerow([
                timestamp, pos.x, pos.y, pos.z,
                ori.x, ori.y, ori.z, ori.w
            ])
        self.path_file.flush()

    def camera_pose_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pos = msg.pose.position
        ori = msg.pose.orientation

        self.camera_pose_writer.writerow([
            timestamp, pos.x, pos.y, pos.z,
            ori.x, ori.y, ori.z, ori.w
        ])
        self.camera_pose_file.flush()

    def __del__(self):
        if hasattr(self, 'odom_file'):
            self.odom_file.close()
        if hasattr(self, 'path_file'):
            self.path_file.close()
        if hasattr(self, 'camera_pose_file'):
            self.camera_pose_file.close()


def main():
    rclpy.init()
    logger = VINSLogger()

    try:
        rclpy.spin(logger)
    except KeyboardInterrupt:
        print("\nLogger stopped by user")
    finally:
        logger.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
