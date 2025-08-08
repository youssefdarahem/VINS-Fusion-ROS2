#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import NavSatFix
import numpy as np
import math
import json
import csv
from datetime import datetime
import os


class VINSErrorAnalyzer(Node):
    def __init__(self):
        super().__init__('vins_error_analyzer')

        # Data storage
        self.vins_data = []
        self.gps_data = []
        self.global_path_data = []

        # GPS reference point
        self.gps_ref_lat = None
        self.gps_ref_lon = None
        self.gps_ref_alt = None

        # Heading alignment
        self.heading_aligned = False
        self.alignment_rotation = 0.0
        self.min_points_for_alignment = 50  # Minimum points needed for alignment

        # Statistics
        self.vins_count = 0
        self.gps_count = 0
        self.global_count = 0

        # Output directory
        self.output_dir = "vins_error_analysis"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Subscribers
        self.vins_sub = self.create_subscription(
            Odometry, '/vins_estimator/odometry', self.vins_callback, 10)
        self.gps_sub = self.create_subscription(
            NavSatFix, '/fix', self.gps_callback, 10)
        self.global_path_sub = self.create_subscription(
            Path, '/globalEstimator/global_path', self.global_path_callback, 10)

        self.get_logger().info("VINS Error Analyzer started")
        self.get_logger().info(f"Output directory: {self.output_dir}")

    def gps_to_local(self, lat, lon, alt):
        """Convert GPS coordinates to local ENU coordinates"""
        if self.gps_ref_lat is None:
            self.gps_ref_lat = lat
            self.gps_ref_lon = lon
            self.gps_ref_alt = alt
            self.get_logger().info(
                f"GPS reference: {lat:.6f}, {lon:.6f}, {alt:.2f}")
            return 0.0, 0.0, 0.0

        R_earth = 6378137.0
        dlat = lat - self.gps_ref_lat
        dlon = lon - self.gps_ref_lon
        dalt = alt - self.gps_ref_alt

        x = dlon * R_earth * \
            math.cos(math.radians(self.gps_ref_lat)) * math.pi / 180.0
        y = dlat * R_earth * math.pi / 180.0
        z = dalt

        return x, y, z

    def align_trajectories_heading(self):
        """Align VINS trajectory heading with GPS using initial direction"""
        if (self.heading_aligned or
            len(self.vins_data) < self.min_points_for_alignment or
                len(self.gps_data) < self.min_points_for_alignment):
            return False

        # Get initial trajectory segments for alignment
        vins_segment = self.vins_data[:self.min_points_for_alignment]
        gps_segment = self.gps_data[:self.min_points_for_alignment]

        # Calculate initial movement directions
        vins_start = np.array([vins_segment[0]['x'], vins_segment[0]['y']])
        vins_end = np.array([vins_segment[-1]['x'], vins_segment[-1]['y']])
        vins_direction = vins_end - vins_start
        vins_heading = math.atan2(vins_direction[1], vins_direction[0])

        gps_start = np.array([gps_segment[0]['x'], gps_segment[0]['y']])
        gps_end = np.array([gps_segment[-1]['x'], gps_segment[-1]['y']])
        gps_direction = gps_end - gps_start
        gps_heading = math.atan2(gps_direction[1], gps_direction[0])

        # Calculate rotation needed to align headings
        heading_diff = gps_heading - vins_heading

        # Normalize angle to [-pi, pi]
        while heading_diff > math.pi:
            heading_diff -= 2 * math.pi
        while heading_diff < -math.pi:
            heading_diff += 2 * math.pi

        self.alignment_rotation = heading_diff
        self.heading_aligned = True

        # Apply rotation to all existing VINS data
        cos_a, sin_a = math.cos(heading_diff), math.sin(heading_diff)

        for data_point in self.vins_data:
            x_orig = data_point['x']
            y_orig = data_point['y']

            # Rotate around origin
            data_point['x'] = cos_a * x_orig - sin_a * y_orig
            data_point['y'] = sin_a * x_orig + cos_a * y_orig

        self.get_logger().info(
            f"🎯 Heading aligned: {math.degrees(heading_diff):.1f}° rotation applied")
        return True

    def vins_callback(self, msg):
        """Store VINS odometry data"""
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Apply heading alignment if already determined
        if self.heading_aligned:
            cos_a, sin_a = math.cos(self.alignment_rotation), math.sin(
                self.alignment_rotation)
            x_rotated = cos_a * x - sin_a * y
            y_rotated = sin_a * x + cos_a * y
            x, y = x_rotated, y_rotated

        data_point = {
            'timestamp': timestamp,
            'x': x,
            'y': y,
            'z': msg.pose.pose.position.z,
            'qx': msg.pose.pose.orientation.x,
            'qy': msg.pose.pose.orientation.y,
            'qz': msg.pose.pose.orientation.z,
            'qw': msg.pose.pose.orientation.w,
            'vx': msg.twist.twist.linear.x,
            'vy': msg.twist.twist.linear.y,
            'vz': msg.twist.twist.linear.z
        }

        self.vins_data.append(data_point)
        self.vins_count += 1

        if self.vins_count == 1:
            self.get_logger().info("✅ First VINS message received")
        elif self.vins_count % 100 == 0:
            self.get_logger().info(f"📊 VINS messages: {self.vins_count}")

        # Try to align heading when we have enough data
        if not self.heading_aligned:
            self.align_trajectories_heading()

    def gps_callback(self, msg):
        """Store GPS data"""
        if msg.status.status < 0:
            return

        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x, y, z = self.gps_to_local(msg.latitude, msg.longitude, msg.altitude)

        data_point = {
            'timestamp': timestamp,
            'latitude': msg.latitude,
            'longitude': msg.longitude,
            'altitude': msg.altitude,
            'x': x,
            'y': y,
            'z': z,
            'status': msg.status.status,
            'service': msg.status.service
        }

        self.gps_data.append(data_point)
        self.gps_count += 1

        if self.gps_count == 1:
            self.get_logger().info("✅ First GPS message received")
        elif self.gps_count % 50 == 0:
            self.get_logger().info(f"📡 GPS messages: {self.gps_count}")

        # Try to align heading when we have enough data
        if not self.heading_aligned:
            self.align_trajectories_heading()

    def global_path_callback(self, msg):
        """Store global estimator path data"""
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Store the latest pose from the path
        if len(msg.poses) > 0:
            latest_pose = msg.poses[-1].pose
            data_point = {
                'timestamp': timestamp,
                'x': latest_pose.position.x,
                'y': latest_pose.position.y,
                'z': latest_pose.position.z,
                'qx': latest_pose.orientation.x,
                'qy': latest_pose.orientation.y,
                'qz': latest_pose.orientation.z,
                'qw': latest_pose.orientation.w
            }

            self.global_path_data.append(data_point)
            self.global_count += 1

            if self.global_count == 1:
                self.get_logger().info("✅ First Global Path message received")

    def interpolate_data(self, data1, data2, target_timestamps):
        """Interpolate data to common timestamps"""
        if not data1 or not data2:
            return [], []

        # Convert to numpy arrays for interpolation
        timestamps1 = np.array([d['timestamp'] for d in data1])
        timestamps2 = np.array([d['timestamp'] for d in data2])

        # Find common time range
        start_time = max(timestamps1[0], timestamps2[0])
        end_time = min(timestamps1[-1], timestamps2[-1])

        # Filter target timestamps to common range
        valid_targets = [
            t for t in target_timestamps if start_time <= t <= end_time]

        if not valid_targets:
            return [], []

        # Interpolate data1
        interp1 = []
        for target_t in valid_targets:
            # Find closest data points
            idx = np.searchsorted(timestamps1, target_t)
            if idx == 0:
                interp1.append(data1[0])
            elif idx >= len(data1):
                interp1.append(data1[-1])
            else:
                # Linear interpolation
                t1, t2 = timestamps1[idx-1], timestamps1[idx]
                alpha = (target_t - t1) / (t2 - t1) if t2 != t1 else 0

                point = {}
                for key in ['x', 'y', 'z']:
                    if key in data1[idx-1] and key in data1[idx]:
                        point[key] = data1[idx-1][key] + alpha * \
                            (data1[idx][key] - data1[idx-1][key])
                point['timestamp'] = target_t
                interp1.append(point)

        # Interpolate data2 similarly
        interp2 = []
        for target_t in valid_targets:
            idx = np.searchsorted(timestamps2, target_t)
            if idx == 0:
                interp2.append(data2[0])
            elif idx >= len(data2):
                interp2.append(data2[-1])
            else:
                t1, t2 = timestamps2[idx-1], timestamps2[idx]
                alpha = (target_t - t1) / (t2 - t1) if t2 != t1 else 0

                point = {}
                for key in ['x', 'y', 'z']:
                    if key in data2[idx-1] and key in data2[idx]:
                        point[key] = data2[idx-1][key] + alpha * \
                            (data2[idx][key] - data2[idx-1][key])
                point['timestamp'] = target_t
                interp2.append(point)

        return interp1, interp2

    def calculate_errors(self):
        """Calculate various error metrics"""
        if not self.vins_data or not self.gps_data:
            self.get_logger().warning("Insufficient data for error calculation")
            return

        # Use VINS timestamps as reference for interpolation
        vins_timestamps = [d['timestamp'] for d in self.vins_data]

        # Interpolate GPS data to VINS timestamps
        vins_interp, gps_interp = self.interpolate_data(
            self.vins_data, self.gps_data, vins_timestamps)

        if not vins_interp or not gps_interp:
            self.get_logger().warning("No overlapping data for error calculation")
            return

        # Calculate errors
        errors = []
        for v, g in zip(vins_interp, gps_interp):
            if 'x' in v and 'y' in v and 'x' in g and 'y' in g:
                error_2d = math.sqrt(
                    (v['x'] - g['x'])**2 + (v['y'] - g['y'])**2)
                error_3d = math.sqrt(
                    (v['x'] - g['x'])**2 + (v['y'] - g['y'])**2 + (v.get('z', 0) - g.get('z', 0))**2)

                error_data = {
                    'timestamp': v['timestamp'],
                    'vins_x': v['x'],
                    'vins_y': v['y'],
                    'vins_z': v.get('z', 0),
                    'gps_x': g['x'],
                    'gps_y': g['y'],
                    'gps_z': g.get('z', 0),
                    'error_2d': error_2d,
                    'error_3d': error_3d,
                    'error_x': abs(v['x'] - g['x']),
                    'error_y': abs(v['y'] - g['y']),
                    'error_z': abs(v.get('z', 0) - g.get('z', 0))
                }
                errors.append(error_data)

        return errors

    def save_analysis(self):
        """Save comprehensive error analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Calculate errors
        errors = self.calculate_errors()

        if not errors:
            self.get_logger().error("No errors calculated - insufficient data")
            return

        # Convert to numpy for statistics
        errors_2d = np.array([e['error_2d'] for e in errors])
        errors_3d = np.array([e['error_3d'] for e in errors])
        errors_x = np.array([e['error_x'] for e in errors])
        errors_y = np.array([e['error_y'] for e in errors])
        errors_z = np.array([e['error_z'] for e in errors])

        # Calculate statistics
        stats = {
            'data_summary': {
                'vins_messages': len(self.vins_data),
                'gps_messages': len(self.gps_data),
                'global_path_messages': len(self.global_path_data),
                'synchronized_points': len(errors),
                'analysis_timestamp': timestamp
            },
            'error_statistics_2d': {
                'mean': float(np.mean(errors_2d)),
                'std': float(np.std(errors_2d)),
                'rmse': float(np.sqrt(np.mean(errors_2d**2))),
                'max': float(np.max(errors_2d)),
                'min': float(np.min(errors_2d)),
                'median': float(np.median(errors_2d)),
                'percentile_95': float(np.percentile(errors_2d, 95))
            },
            'error_statistics_3d': {
                'mean': float(np.mean(errors_3d)),
                'std': float(np.std(errors_3d)),
                'rmse': float(np.sqrt(np.mean(errors_3d**2))),
                'max': float(np.max(errors_3d)),
                'min': float(np.min(errors_3d)),
                'median': float(np.median(errors_3d)),
                'percentile_95': float(np.percentile(errors_3d, 95))
            },
            'component_errors': {
                'x_axis': {
                    'mean': float(np.mean(errors_x)),
                    'std': float(np.std(errors_x)),
                    'rmse': float(np.sqrt(np.mean(errors_x**2)))
                },
                'y_axis': {
                    'mean': float(np.mean(errors_y)),
                    'std': float(np.std(errors_y)),
                    'rmse': float(np.sqrt(np.mean(errors_y**2)))
                },
                'z_axis': {
                    'mean': float(np.mean(errors_z)),
                    'std': float(np.std(errors_z)),
                    'rmse': float(np.sqrt(np.mean(errors_z**2)))
                }
            }
        }

        # Save detailed error data to CSV
        csv_file = os.path.join(
            self.output_dir, f'error_analysis_{timestamp}.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=errors[0].keys())
            writer.writeheader()
            writer.writerows(errors)

        # Save statistics to JSON
        json_file = os.path.join(
            self.output_dir, f'error_statistics_{timestamp}.json')
        with open(json_file, 'w') as f:
            json.dump(stats, f, indent=2)

        # Save human-readable summary
        summary_file = os.path.join(
            self.output_dir, f'error_summary_{timestamp}.txt')
        with open(summary_file, 'w') as f:
            f.write("VINS-GPS Error Analysis Summary\n")
            f.write("===============================\n\n")
            f.write(
                f"Analysis generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Bag: lighthouse_francis_sample\n\n")

            f.write("Data Summary:\n")
            f.write(
                f"  VINS messages: {stats['data_summary']['vins_messages']}\n")
            f.write(
                f"  GPS messages: {stats['data_summary']['gps_messages']}\n")
            f.write(
                f"  Global path messages: {stats['data_summary']['global_path_messages']}\n")
            f.write(
                f"  Synchronized points: {stats['data_summary']['synchronized_points']}\n\n")

            f.write("2D Position Error Statistics:\n")
            f.write(
                f"  Mean error: {stats['error_statistics_2d']['mean']:.3f} m\n")
            f.write(f"  RMSE: {stats['error_statistics_2d']['rmse']:.3f} m\n")
            f.write(
                f"  Standard deviation: {stats['error_statistics_2d']['std']:.3f} m\n")
            f.write(
                f"  Maximum error: {stats['error_statistics_2d']['max']:.3f} m\n")
            f.write(
                f"  Minimum error: {stats['error_statistics_2d']['min']:.3f} m\n")
            f.write(
                f"  Median error: {stats['error_statistics_2d']['median']:.3f} m\n")
            f.write(
                f"  95th percentile: {stats['error_statistics_2d']['percentile_95']:.3f} m\n\n")

            f.write("3D Position Error Statistics:\n")
            f.write(
                f"  Mean error: {stats['error_statistics_3d']['mean']:.3f} m\n")
            f.write(f"  RMSE: {stats['error_statistics_3d']['rmse']:.3f} m\n")
            f.write(
                f"  Standard deviation: {stats['error_statistics_3d']['std']:.3f} m\n")
            f.write(
                f"  Maximum error: {stats['error_statistics_3d']['max']:.3f} m\n")
            f.write(
                f"  Minimum error: {stats['error_statistics_3d']['min']:.3f} m\n")
            f.write(
                f"  Median error: {stats['error_statistics_3d']['median']:.3f} m\n")
            f.write(
                f"  95th percentile: {stats['error_statistics_3d']['percentile_95']:.3f} m\n\n")

            f.write("Component-wise RMSE:\n")
            f.write(
                f"  X-axis: {stats['component_errors']['x_axis']['rmse']:.3f} m\n")
            f.write(
                f"  Y-axis: {stats['component_errors']['y_axis']['rmse']:.3f} m\n")
            f.write(
                f"  Z-axis: {stats['component_errors']['z_axis']['rmse']:.3f} m\n")

        # Print summary
        print(f"\n📊 Error Analysis Complete!")
        print(f"===========================")
        print(f"Synchronized points: {len(errors)}")
        print(f"2D RMSE: {stats['error_statistics_2d']['rmse']:.3f} m")
        print(f"3D RMSE: {stats['error_statistics_3d']['rmse']:.3f} m")
        print(f"Mean 2D error: {stats['error_statistics_2d']['mean']:.3f} m")
        print(f"Max 2D error: {stats['error_statistics_2d']['max']:.3f} m")
        print(f"\nFiles saved:")
        print(f"  📈 {csv_file}")
        print(f"  📋 {json_file}")
        print(f"  📄 {summary_file}")


def main():
    rclpy.init()

    try:
        analyzer = VINSErrorAnalyzer()

        print("🎯 VINS Error Analyzer")
        print("======================")
        print("Analyzing VINS vs GPS trajectory errors...")
        print("This will collect:")
        print("- VINS estimator odometry")
        print("- GPS fix data")
        print("- Global estimator path (if available)")
        print("\nPress Ctrl+C to stop and generate analysis...")
        print("")

        rclpy.spin(analyzer)

    except KeyboardInterrupt:
        print("\n⏹️  Analysis stopped by user")
        print("Generating final error analysis...")
        if 'analyzer' in locals():
            analyzer.save_analysis()
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
