#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
import numpy as np
import csv
import os
from datetime import datetime
import math
from collections import deque
import threading
import time


class VINSGPSErrorAnalyzer(Node):
    def __init__(self):
        super().__init__('vins_gps_error_analyzer')

        # Data storage
        self.vins_data = deque()
        self.gps_data = deque()
        self.synchronized_pairs = []

        # Reference point for local coordinate conversion
        self.gps_ref_lat = None
        self.gps_ref_lon = None
        self.gps_ref_alt = None

        # Trajectory alignment
        self.trajectory_aligned = False
        self.alignment_rotation = 0.0

        # Statistics
        self.total_vins_messages = 0
        self.total_gps_messages = 0

        # Subscribers
        self.vins_sub = self.create_subscription(
            Odometry, '/odometry', self.vins_callback, 10)
        self.gps_sub = self.create_subscription(
            NavSatFix, '/fix', self.gps_callback, 10)

        # Output directory
        self.output_dir = "/tmp/vins_gps_analysis"
        os.makedirs(self.output_dir, exist_ok=True)

        # Analysis timer
        self.analysis_timer = self.create_timer(5.0, self.analyze_errors)

        self.get_logger().info("VINS-GPS Error Analyzer started")
        self.get_logger().info(f"Output directory: {self.output_dir}")

    def gps_to_local(self, lat, lon, alt):
        """Convert GPS coordinates to local ENU coordinates"""
        if self.gps_ref_lat is None:
            # Set first GPS point as reference
            self.gps_ref_lat = lat
            self.gps_ref_lon = lon
            self.gps_ref_alt = alt
            self.get_logger().info(
                f"GPS reference set: lat={lat:.6f}, lon={lon:.6f}, alt={alt:.2f}")
            return 0.0, 0.0, 0.0

        # Convert to ENU coordinates (East-North-Up)
        # Approximate conversion for small distances
        R_earth = 6378137.0  # Earth radius in meters

        dlat = lat - self.gps_ref_lat
        dlon = lon - self.gps_ref_lon
        dalt = alt - self.gps_ref_alt

        # Convert to meters
        x = dlon * R_earth * \
            math.cos(math.radians(self.gps_ref_lat)) * math.pi / 180.0  # East
        y = dlat * R_earth * math.pi / 180.0  # North
        z = dalt  # Up

        return x, y, z

    def align_trajectories_realtime(self, vins_positions, gps_positions):
        """Quick trajectory alignment for real-time processing (X-Y only)"""
        if len(vins_positions) < 10 or len(gps_positions) < 10:
            return vins_positions  # Need enough points for alignment

        vins_pos = np.array(vins_positions)
        gps_pos = np.array(gps_positions)

        # Only work with X-Y coordinates
        vins_xy = vins_pos[:, :2]
        gps_xy = gps_pos[:, :2]

        # Center trajectories (X-Y only)
        vins_xy_centered = vins_xy - np.mean(vins_xy, axis=0)
        gps_xy_centered = gps_xy - np.mean(gps_xy, axis=0)

        # Test common angles (0, 90, 180, 270 degrees)
        best_error = float('inf')
        best_angle = 0

        for angle_deg in [0, 90, 180, 270]:
            angle_rad = np.radians(angle_deg)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

            # 2D rotation matrix (X-Y only)
            rotation_matrix_2d = np.array([
                [cos_a, -sin_a],
                [sin_a,  cos_a]
            ])

            vins_xy_rotated = vins_xy_centered @ rotation_matrix_2d.T
            error = np.sum((vins_xy_rotated - gps_xy_centered)**2)

            if error < best_error:
                best_error = error
                best_angle = angle_deg

        # Apply best rotation to X-Y only
        if best_angle != 0:
            angle_rad = np.radians(best_angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix_2d = np.array([
                [cos_a, -sin_a],
                [sin_a,  cos_a]
            ])

            vins_xy_aligned = (
                vins_xy - np.mean(vins_xy, axis=0)) @ rotation_matrix_2d.T
            vins_xy_aligned += np.mean(gps_xy, axis=0)

            # Combine aligned X-Y with original Z
            vins_aligned = np.column_stack([
                vins_xy_aligned[:, 0],  # Aligned X
                vins_xy_aligned[:, 1],  # Aligned Y
                vins_pos[:, 2]          # Original Z (unchanged)
            ])

            self.alignment_rotation = best_angle
            self.trajectory_aligned = True

            return vins_aligned.tolist()

        return vins_positions

    def vins_callback(self, msg):
        """Store VINS odometry data"""
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        self.vins_data.append({
            'timestamp': timestamp,
            'x': x, 'y': y, 'z': z
        })

        self.total_vins_messages += 1

        # Keep only recent data (last 60 seconds)
        cutoff_time = timestamp - 60.0
        while self.vins_data and self.vins_data[0]['timestamp'] < cutoff_time:
            self.vins_data.popleft()

    def gps_callback(self, msg):
        """Store GPS data and convert to local coordinates"""
        if msg.status.status < 0:  # Skip if no GPS fix
            return

        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        lat = msg.latitude
        lon = msg.longitude
        alt = msg.altitude

        # Convert to local coordinates
        x, y, z = self.gps_to_local(lat, lon, alt)

        self.gps_data.append({
            'timestamp': timestamp,
            'x': x, 'y': y, 'z': z,
            'lat': lat, 'lon': lon, 'alt': alt
        })

        self.total_gps_messages += 1

        # Keep only recent data (last 60 seconds)
        cutoff_time = timestamp - 60.0
        while self.gps_data and self.gps_data[0]['timestamp'] < cutoff_time:
            self.gps_data.popleft()

    def find_closest_gps(self, vins_timestamp, max_time_diff=0.1):
        """Find the closest GPS measurement to a VINS timestamp"""
        best_match = None
        min_time_diff = float('inf')

        for gps_point in self.gps_data:
            time_diff = abs(gps_point['timestamp'] - vins_timestamp)
            if time_diff < min_time_diff and time_diff <= max_time_diff:
                min_time_diff = time_diff
                best_match = gps_point

        return best_match, min_time_diff

    def synchronize_data(self):
        """Synchronize VINS and GPS data by timestamp"""
        synchronized = []

        for vins_point in self.vins_data:
            gps_match, time_diff = self.find_closest_gps(
                vins_point['timestamp'])

            if gps_match is not None:
                synchronized.append({
                    'timestamp': vins_point['timestamp'],
                    'vins_x': vins_point['x'],
                    'vins_y': vins_point['y'],
                    'vins_z': vins_point['z'],
                    'gps_x': gps_match['x'],
                    'gps_y': gps_match['y'],
                    'gps_z': gps_match['z'],
                    'time_diff': time_diff
                })

        # Apply trajectory alignment if we have enough data
        if len(synchronized) >= 10 and not self.trajectory_aligned:
            vins_positions = [[p['vins_x'], p['vins_y'], p['vins_z']]
                              for p in synchronized]
            gps_positions = [[p['gps_x'], p['gps_y'], p['gps_z']]
                             for p in synchronized]

            aligned_positions = self.align_trajectories_realtime(
                vins_positions, gps_positions)

            # Update synchronized data with aligned positions
            for i, pos in enumerate(aligned_positions):
                synchronized[i]['vins_x'] = pos[0]
                synchronized[i]['vins_y'] = pos[1]
                synchronized[i]['vins_z'] = pos[2]

            if self.trajectory_aligned:
                print(
                    f"🎯 Trajectories aligned with {self.alignment_rotation}° rotation")

        return synchronized

    def calculate_errors(self, synchronized_data):
        """Calculate various error metrics"""
        if len(synchronized_data) < 2:
            return None

        errors_x = [p['vins_x'] - p['gps_x'] for p in synchronized_data]
        errors_y = [p['vins_y'] - p['gps_y'] for p in synchronized_data]
        errors_z = [p['vins_z'] - p['gps_z'] for p in synchronized_data]

        # 2D and 3D position errors
        errors_2d = [math.sqrt(ex**2 + ey**2)
                     for ex, ey in zip(errors_x, errors_y)]
        errors_3d = [math.sqrt(ex**2 + ey**2 + ez**2)
                     for ex, ey, ez in zip(errors_x, errors_y, errors_z)]

        # Statistics
        metrics = {
            'n_points': len(synchronized_data),
            'mean_error_x': np.mean(errors_x),
            'mean_error_y': np.mean(errors_y),
            'mean_error_z': np.mean(errors_z),
            'std_error_x': np.std(errors_x),
            'std_error_y': np.std(errors_y),
            'std_error_z': np.std(errors_z),
            'rmse_x': np.sqrt(np.mean([e**2 for e in errors_x])),
            'rmse_y': np.sqrt(np.mean([e**2 for e in errors_y])),
            'rmse_z': np.sqrt(np.mean([e**2 for e in errors_z])),
            'mean_error_2d': np.mean(errors_2d),
            'mean_error_3d': np.mean(errors_3d),
            'rmse_2d': np.sqrt(np.mean([e**2 for e in errors_2d])),
            'rmse_3d': np.sqrt(np.mean([e**2 for e in errors_3d])),
            'max_error_2d': max(errors_2d),
            'max_error_3d': max(errors_3d),
        }

        return metrics

    def save_detailed_analysis(self, synchronized_data, metrics):
        """Save detailed analysis to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save synchronized data
        csv_file = f"{self.output_dir}/vins_gps_comparison_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'vins_x', 'vins_y', 'vins_z',
                'gps_x', 'gps_y', 'gps_z',
                'error_x', 'error_y', 'error_z', 'error_2d', 'error_3d', 'time_diff'
            ])

            for point in synchronized_data:
                error_x = point['vins_x'] - point['gps_x']
                error_y = point['vins_y'] - point['gps_y']
                error_z = point['vins_z'] - point['gps_z']
                error_2d = math.sqrt(error_x**2 + error_y**2)
                error_3d = math.sqrt(error_x**2 + error_y**2 + error_z**2)

                writer.writerow([
                    point['timestamp'],
                    point['vins_x'], point['vins_y'], point['vins_z'],
                    point['gps_x'], point['gps_y'], point['gps_z'],
                    error_x, error_y, error_z, error_2d, error_3d,
                    point['time_diff']
                ])

        # Save metrics summary
        metrics_file = f"{self.output_dir}/error_metrics_{timestamp}.txt"
        with open(metrics_file, 'w') as f:
            f.write("VINS-GPS Error Analysis Summary\n")
            f.write("================================\n\n")
            f.write(f"Analysis time: {datetime.now()}\n")
            f.write(
                f"Number of synchronized points: {metrics['n_points']}\n\n")

            f.write("Position Errors (meters):\n")
            f.write(
                f"  Mean X error: {metrics['mean_error_x']:.3f} ± {metrics['std_error_x']:.3f}\n")
            f.write(
                f"  Mean Y error: {metrics['mean_error_y']:.3f} ± {metrics['std_error_y']:.3f}\n")
            f.write(
                f"  Mean Z error: {metrics['mean_error_z']:.3f} ± {metrics['std_error_z']:.3f}\n\n")

            f.write("Root Mean Square Errors (RMSE):\n")
            f.write(f"  RMSE X: {metrics['rmse_x']:.3f} m\n")
            f.write(f"  RMSE Y: {metrics['rmse_y']:.3f} m\n")
            f.write(f"  RMSE Z: {metrics['rmse_z']:.3f} m\n")
            f.write(f"  RMSE 2D: {metrics['rmse_2d']:.3f} m\n")
            f.write(f"  RMSE 3D: {metrics['rmse_3d']:.3f} m\n\n")

            f.write("Maximum Errors:\n")
            f.write(f"  Max 2D error: {metrics['max_error_2d']:.3f} m\n")
            f.write(f"  Max 3D error: {metrics['max_error_3d']:.3f} m\n\n")

            f.write("Mean Errors:\n")
            f.write(f"  Mean 2D error: {metrics['mean_error_2d']:.3f} m\n")
            f.write(f"  Mean 3D error: {metrics['mean_error_3d']:.3f} m\n")

        self.get_logger().info(f"Analysis saved to: {csv_file}")
        self.get_logger().info(f"Metrics saved to: {metrics_file}")

    def analyze_errors(self):
        """Perform error analysis"""
        if not self.vins_data or not self.gps_data:
            return

        # Synchronize data
        synchronized = self.synchronize_data()

        if len(synchronized) < 5:
            self.get_logger().info(
                f"Waiting for more data... (VINS: {self.total_vins_messages}, GPS: {self.total_gps_messages}, Sync: {len(synchronized)})")
            return

        # Calculate errors
        metrics = self.calculate_errors(synchronized)

        if metrics is None:
            return

        # Log current status
        self.get_logger().info(
            f"Error Analysis - Points: {metrics['n_points']}, "
            f"2D RMSE: {metrics['rmse_2d']:.2f}m, "
            f"3D RMSE: {metrics['rmse_3d']:.2f}m, "
            f"Max 2D: {metrics['max_error_2d']:.2f}m"
        )

        # Save detailed analysis every 30 seconds
        current_time = time.time()
        if not hasattr(self, 'last_save_time') or (current_time - self.last_save_time) > 30:
            self.save_detailed_analysis(synchronized, metrics)
            self.last_save_time = current_time


def main():
    rclpy.init()

    try:
        analyzer = VINSGPSErrorAnalyzer()

        print("🔍 VINS-GPS Error Analyzer Started")
        print("==================================")
        print("This tool compares VINS odometry with GPS ground truth.")
        print("- VINS data: /odometry topic")
        print("- GPS data: /fix topic")
        print("- Analysis output: /tmp/vins_gps_analysis/")
        print("")
        print("Real-time error metrics will be displayed every 5 seconds.")
        print("Detailed analysis saved every 30 seconds.")
        print("Press Ctrl+C to stop...")
        print("")

        rclpy.spin(analyzer)

    except KeyboardInterrupt:
        print("\nAnalysis stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'analyzer' in locals():
            analyzer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
