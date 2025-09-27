#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped
import pandas as pd
import numpy as np
import os
from datetime import datetime
import threading
import signal
import sys


class VINSLiveComparison(Node):
    def __init__(self):
        super().__init__('vins_live_comparison')

        # Data storage
        self.vins_data = []
        self.ground_truth_data = []
        self.gps_data = []

        # Output configuration
        self.output_dir = "/tmp/vins_live_output"
        os.makedirs(self.output_dir, exist_ok=True)

        self.start_time = None
        self.recording = True

        # Subscribers for live VINS output
        self.vins_odom_sub = self.create_subscription(
            Odometry, '/vins_estimator/odometry', self.vins_callback, 10)

        # Subscribers for ground truth (from bag playback)
        self.gt_odom_sub = self.create_subscription(
            Odometry, '/Odometry', self.ground_truth_callback, 10)

        # GPS subscriber (for reference)
        self.gps_sub = self.create_subscription(
            NavSatFix, '/fix', self.gps_callback, 10)

        # Statistics
        self.vins_count = 0
        self.gt_count = 0
        self.gps_count = 0

        self.get_logger().info("🎯 VINS Live Comparison Node Started")
        self.get_logger().info(f"📁 Output directory: {self.output_dir}")

        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        self.get_logger().info("\n🛑 Shutdown signal received. Saving data...")
        self.save_all_data()
        sys.exit(0)

    def vins_callback(self, msg):
        """Record live VINS estimator output"""
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.start_time is None:
            self.start_time = timestamp
            self.get_logger().info("✅ Started recording VINS data")

        rel_time = timestamp - self.start_time

        self.vins_data.append({
            'timestamp': rel_time,
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'z': msg.pose.pose.position.z,
            'qx': msg.pose.pose.orientation.x,
            'qy': msg.pose.pose.orientation.y,
            'qz': msg.pose.pose.orientation.z,
            'qw': msg.pose.pose.orientation.w,
            'vx': msg.twist.twist.linear.x,
            'vy': msg.twist.twist.linear.y,
            'vz': msg.twist.twist.linear.z,
            'wx': msg.twist.twist.angular.x,
            'wy': msg.twist.twist.angular.y,
            'wz': msg.twist.twist.angular.z
        })

        self.vins_count += 1

        if self.vins_count % 100 == 0:
            self.get_logger().info(f"📊 VINS messages: {self.vins_count}")

    def ground_truth_callback(self, msg):
        """Record ground truth odometry from bag"""
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.start_time is None:
            self.start_time = timestamp
            self.get_logger().info("✅ Started recording ground truth data")

        rel_time = timestamp - self.start_time

        self.ground_truth_data.append({
            'timestamp': rel_time,
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'z': msg.pose.pose.position.z,
            'qx': msg.pose.pose.orientation.x,
            'qy': msg.pose.pose.orientation.y,
            'qz': msg.pose.pose.orientation.z,
            'qw': msg.pose.pose.orientation.w,
            'vx': msg.twist.twist.linear.x,
            'vy': msg.twist.twist.linear.y,
            'vz': msg.twist.twist.linear.z,
            'wx': msg.twist.twist.angular.x,
            'wy': msg.twist.twist.angular.y,
            'wz': msg.twist.twist.angular.z
        })

        self.gt_count += 1

        if self.gt_count % 100 == 0:
            self.get_logger().info(f"📍 Ground truth messages: {self.gt_count}")

    def gps_callback(self, msg):
        """Record GPS data for reference"""
        if msg.status.status < 0:
            return

        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.start_time is None:
            self.start_time = timestamp

        rel_time = timestamp - self.start_time

        self.gps_data.append({
            'timestamp': rel_time,
            'latitude': msg.latitude,
            'longitude': msg.longitude,
            'altitude': msg.altitude,
            'status': msg.status.status
        })

        self.gps_count += 1

        if self.gps_count % 50 == 0:
            self.get_logger().info(f"🛰️  GPS messages: {self.gps_count}")

    def save_all_data(self):
        """Save all recorded data to CSV files"""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Save VINS data
            if len(self.vins_data) > 0:
                vins_df = pd.DataFrame(self.vins_data)
                vins_file = os.path.join(
                    self.output_dir, f'live_vins_output_{timestamp_str}.csv')
                vins_df.to_csv(vins_file, index=False)
                self.get_logger().info(
                    f"✅ VINS data saved: {vins_file} ({len(self.vins_data)} points)")

            # Save ground truth data
            if len(self.ground_truth_data) > 0:
                gt_df = pd.DataFrame(self.ground_truth_data)
                gt_file = os.path.join(
                    self.output_dir, f'ground_truth_{timestamp_str}.csv')
                gt_df.to_csv(gt_file, index=False)
                self.get_logger().info(
                    f"✅ Ground truth saved: {gt_file} ({len(self.ground_truth_data)} points)")

            # Save GPS data
            if len(self.gps_data) > 0:
                gps_df = pd.DataFrame(self.gps_data)
                gps_file = os.path.join(
                    self.output_dir, f'gps_reference_{timestamp_str}.csv')
                gps_df.to_csv(gps_file, index=False)
                self.get_logger().info(
                    f"✅ GPS data saved: {gps_file} ({len(self.gps_data)} points)")

            # Create analysis script call
            if len(self.vins_data) > 0 and len(self.ground_truth_data) > 0:
                analysis_command = f"""
# Run offline analysis with the recorded data:
cd /home/joey/Desktop/dev/VINS-Fusion-ROS2/scripts
python3 offline_vins_gps_analysis.py \\
    --vins-file {vins_file} \\
    --gps-file {gt_file} \\
    --output-dir {self.output_dir}/analysis_{timestamp_str} \\
    --align-trajectories \\
    --align-altitude
"""

                script_file = os.path.join(
                    self.output_dir, f'run_analysis_{timestamp_str}.sh')
                with open(script_file, 'w') as f:
                    f.write("#!/bin/bash\n")
                    f.write(analysis_command)

                os.chmod(script_file, 0o755)
                self.get_logger().info(
                    f"📝 Analysis script created: {script_file}")

        except Exception as e:
            self.get_logger().error(f"❌ Error saving data: {e}")

    def print_status(self):
        """Print current recording status"""
        duration = 0
        if self.start_time is not None:
            current_time = self.get_clock().now().nanoseconds * 1e-9
            duration = current_time - self.start_time

        self.get_logger().info(f"""
📊 Live Recording Status:
   ⏱️  Duration: {duration:.1f}s
   🧭 VINS messages: {self.vins_count}
   📍 Ground truth: {self.gt_count}
   🛰️  GPS messages: {self.gps_count}
   📁 Output: {self.output_dir}
""")


def main():
    rclpy.init()

    try:
        node = VINSLiveComparison()

        print("🎯 VINS Live Comparison Recorder")
        print("=" * 40)
        print("Recording live VINS output vs ground truth")
        print("Press Ctrl+C to stop recording and save data")
        print("")

        # Print status every 10 seconds
        def status_timer():
            while rclpy.ok():
                node.print_status()
                threading.Event().wait(10)

        status_thread = threading.Thread(target=status_timer, daemon=True)
        status_thread.start()

        # Spin the node
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\n🛑 Recording stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'node' in locals():
            node.save_all_data()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
