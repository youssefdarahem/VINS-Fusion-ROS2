#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
from collections import deque
import threading


class VINSGPSTrajectoryVisualizer(Node):
    def __init__(self):
        super().__init__('vins_gps_visualizer')

        # Data storage
        self.max_points = 500
        self.vins_x = deque(maxlen=self.max_points)
        self.vins_y = deque(maxlen=self.max_points)
        self.gps_x = deque(maxlen=self.max_points)
        self.gps_y = deque(maxlen=self.max_points)
        self.errors_2d = deque(maxlen=self.max_points)
        self.timestamps = deque(maxlen=self.max_points)

        # GPS reference point
        self.gps_ref_lat = None
        self.gps_ref_lon = None
        self.gps_ref_alt = None

        # Trajectory alignment
        self.trajectory_aligned = False
        self.alignment_rotation = 0.0
        self.last_alignment_check = 0
        self.heading_aligned = False
        self.min_points_for_heading_alignment = 30

        # Statistics
        self.vins_count = 0
        self.gps_count = 0
        self.start_time = None

        # Subscribers
        self.vins_sub = self.create_subscription(
            Odometry, '/vins_estimator/odometry', self.vins_callback, 10)
        self.gps_sub = self.create_subscription(
            NavSatFix, '/fix', self.gps_callback, 10)

        # Setup matplotlib
        self.setup_plot()

        self.get_logger().info("VINS-GPS Trajectory Visualizer started")

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

    def align_initial_heading(self):
        """Align VINS initial heading with GPS using first trajectory segment"""
        if (self.heading_aligned or
            len(self.vins_x) < self.min_points_for_heading_alignment or
                len(self.gps_x) < self.min_points_for_heading_alignment):
            return False

        # Get initial trajectory segments for heading alignment
        vins_x_segment = list(self.vins_x)[
            :self.min_points_for_heading_alignment]
        vins_y_segment = list(self.vins_y)[
            :self.min_points_for_heading_alignment]
        gps_x_segment = list(self.gps_x)[
            :self.min_points_for_heading_alignment]
        gps_y_segment = list(self.gps_y)[
            :self.min_points_for_heading_alignment]

        # Calculate initial movement directions
        vins_start = np.array([vins_x_segment[0], vins_y_segment[0]])
        vins_end = np.array([vins_x_segment[-1], vins_y_segment[-1]])
        vins_direction = vins_end - vins_start
        vins_heading = math.atan2(vins_direction[1], vins_direction[0])

        gps_start = np.array([gps_x_segment[0], gps_y_segment[0]])
        gps_end = np.array([gps_x_segment[-1], gps_y_segment[-1]])
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

        rotated_x = []
        rotated_y = []

        for i in range(len(self.vins_x)):
            x_orig = self.vins_x[i]
            y_orig = self.vins_y[i]

            # Rotate around origin
            x_new = cos_a * x_orig - sin_a * y_orig
            y_new = sin_a * x_orig + cos_a * y_orig

            rotated_x.append(x_new)
            rotated_y.append(y_new)

        # Update deques with rotated positions
        self.vins_x.clear()
        self.vins_y.clear()
        for x, y in zip(rotated_x, rotated_y):
            self.vins_x.append(x)
            self.vins_y.append(y)

        print(
            f"🎯 Initial heading aligned: {math.degrees(heading_diff):.1f}° rotation applied")
        return True

    def check_trajectory_alignment(self):
        """Check and apply trajectory alignment if needed"""
        if (self.trajectory_aligned or
            len(self.vins_x) < 20 or len(self.gps_x) < 20 or
                len(self.vins_x) - self.last_alignment_check < 10):
            return

        # Convert to numpy arrays for alignment (X-Y only)
        vins_xy = np.array(list(zip(self.vins_x, self.vins_y)))
        gps_xy = np.array(list(zip(self.gps_x, self.gps_y)))

        # Make sure we have the same number of points
        min_len = min(len(vins_xy), len(gps_xy))
        vins_xy = vins_xy[-min_len:]
        gps_xy = gps_xy[-min_len:]

        # Center trajectories (X-Y only)
        vins_xy_centered = vins_xy - np.mean(vins_xy, axis=0)
        gps_xy_centered = gps_xy - np.mean(gps_xy, axis=0)

        # Test common rotations
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

        # Apply alignment if significant rotation needed
        if best_angle != 0:
            self.alignment_rotation = best_angle
            self.trajectory_aligned = True
            self.last_alignment_check = len(self.vins_x)

            # Apply rotation to all existing VINS points
            angle_rad = np.radians(best_angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

            aligned_x = []
            aligned_y = []

            for i in range(len(self.vins_x)):
                x_centered = self.vins_x[i] - np.mean(self.vins_x)
                y_centered = self.vins_y[i] - np.mean(self.vins_y)

                x_rotated = cos_a * x_centered - sin_a * y_centered
                y_rotated = sin_a * x_centered + cos_a * y_centered

                aligned_x.append(x_rotated + np.mean(self.gps_x))
                aligned_y.append(y_rotated + np.mean(self.gps_y))

            # Update deques with aligned positions
            self.vins_x.clear()
            self.vins_y.clear()
            for x, y in zip(aligned_x, aligned_y):
                self.vins_x.append(x)
                self.vins_y.append(y)

            print(f"🎯 Trajectories aligned with {best_angle}° rotation")

    def vins_callback(self, msg):
        """Store VINS odometry data"""
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.start_time is None:
            self.start_time = timestamp

        rel_time = timestamp - self.start_time

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Apply heading alignment if already determined
        if self.heading_aligned:
            cos_a, sin_a = math.cos(self.alignment_rotation), math.sin(
                self.alignment_rotation)
            x_rotated = cos_a * x - sin_a * y
            y_rotated = sin_a * x + cos_a * y
            x, y = x_rotated, y_rotated

        # Apply trajectory alignment if already determined (for legacy support)
        if self.trajectory_aligned and not self.heading_aligned:
            angle_rad = np.radians(self.alignment_rotation)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

            x_rotated = cos_a * x - sin_a * y
            y_rotated = sin_a * x + cos_a * y
            x, y = x_rotated, y_rotated

        self.vins_x.append(x)
        self.vins_y.append(y)
        self.timestamps.append(rel_time)
        self.vins_count += 1

        if self.vins_count == 1:
            self.get_logger().info("✅ First VINS message received")

        # Try initial heading alignment first
        if not self.heading_aligned:
            self.align_initial_heading()

        # Check for trajectory alignment opportunity (fallback method)
        elif not self.trajectory_aligned:
            self.check_trajectory_alignment()

    def gps_callback(self, msg):
        """Store GPS data"""
        if msg.status.status < 0:
            return

        x, y, z = self.gps_to_local(msg.latitude, msg.longitude, msg.altitude)
        self.gps_x.append(x)
        self.gps_y.append(y)
        self.gps_count += 1

        if self.gps_count == 1:
            self.get_logger().info("✅ First GPS message received")

        # Try initial heading alignment when we have enough data
        if not self.heading_aligned:
            self.align_initial_heading()

        # Calculate 2D error if we have VINS data
        if len(self.vins_x) > 0 and len(self.gps_x) > 0:
            error_2d = math.sqrt(
                (self.vins_x[-1] - x)**2 + (self.vins_y[-1] - y)**2)
            self.errors_2d.append(error_2d)

    def setup_plot(self):
        """Setup matplotlib figure"""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(15, 10))

        # Create subplots
        self.ax1 = plt.subplot(2, 2, 1)  # Trajectory comparison
        self.ax2 = plt.subplot(2, 2, 2)  # Error over time
        self.ax3 = plt.subplot(2, 2, 3)  # Error histogram
        self.ax4 = plt.subplot(2, 2, 4)  # Statistics

        # Trajectory plot
        self.ax1.set_title('Flight Trajectories Comparison',
                           fontsize=12, color='white')
        self.ax1.set_xlabel('X Position (m)', color='white')
        self.ax1.set_ylabel('Y Position (m)', color='white')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_aspect('equal')

        # Initialize empty plots
        self.vins_line, = self.ax1.plot(
            [], [], 'cyan', linewidth=2, label='VINS', alpha=0.8)
        self.gps_line, = self.ax1.plot(
            [], [], 'yellow', linewidth=2, label='GPS', alpha=0.8)
        self.vins_current, = self.ax1.plot(
            [], [], 'co', markersize=8, label='VINS Current')
        self.gps_current, = self.ax1.plot(
            [], [], 'yo', markersize=8, label='GPS Current')
        self.ax1.legend(loc='upper right')

        # Error plot
        self.ax2.set_title('2D Position Error Over Time',
                           fontsize=12, color='white')
        self.ax2.set_xlabel('Time (s)', color='white')
        self.ax2.set_ylabel('Error (m)', color='white')
        self.ax2.grid(True, alpha=0.3)
        self.error_line, = self.ax2.plot([], [], 'red', linewidth=2, alpha=0.8)

        # Error histogram
        self.ax3.set_title('Error Distribution', fontsize=12, color='white')
        self.ax3.set_xlabel('2D Error (m)', color='white')
        self.ax3.set_ylabel('Frequency', color='white')
        self.ax3.grid(True, alpha=0.3)

        # Statistics text
        self.ax4.set_title('Real-time Statistics', fontsize=12, color='white')
        self.ax4.axis('off')
        self.stats_text = self.ax4.text(0.05, 0.95, '', transform=self.ax4.transAxes,
                                        fontsize=10, color='white', verticalalignment='top',
                                        fontfamily='monospace')

        plt.tight_layout()

    def update_plot(self, frame):
        """Update the plot with new data"""
        try:
            # Update trajectory plots
            if len(self.vins_x) > 0:
                vins_x_array = np.array(self.vins_x)
                vins_y_array = np.array(self.vins_y)
                self.vins_line.set_data(vins_x_array, vins_y_array)

                if len(vins_x_array) > 0:
                    self.vins_current.set_data(
                        [vins_x_array[-1]], [vins_y_array[-1]])

            if len(self.gps_x) > 0:
                gps_x_array = np.array(self.gps_x)
                gps_y_array = np.array(self.gps_y)
                self.gps_line.set_data(gps_x_array, gps_y_array)

                if len(gps_x_array) > 0:
                    self.gps_current.set_data(
                        [gps_x_array[-1]], [gps_y_array[-1]])

            # Auto-scale trajectory plot
            if len(self.vins_x) > 1 or len(self.gps_x) > 1:
                all_x = list(self.vins_x) + list(self.gps_x)
                all_y = list(self.vins_y) + list(self.gps_y)

                if all_x and all_y:
                    margin = max((max(all_x) - min(all_x)) * 0.1,
                                 (max(all_y) - min(all_y)) * 0.1, 1.0)

                    self.ax1.set_xlim(min(all_x) - margin, max(all_x) + margin)
                    self.ax1.set_ylim(min(all_y) - margin, max(all_y) + margin)

            # Update error plot
            if len(self.errors_2d) > 0 and len(self.timestamps) > 0:
                error_array = np.array(self.errors_2d)
                time_array = np.array(
                    list(self.timestamps)[-len(error_array):])

                self.error_line.set_data(time_array, error_array)

                if len(time_array) > 1:
                    self.ax2.set_xlim(0, max(time_array) + 1)
                    self.ax2.set_ylim(0, max(max(error_array) * 1.1, 1.0))

            # Update error histogram
            if len(self.errors_2d) > 10:
                self.ax3.clear()
                self.ax3.hist(self.errors_2d, bins=20, alpha=0.7,
                              color='red', edgecolor='white')
                self.ax3.set_title('Error Distribution',
                                   fontsize=12, color='white')
                self.ax3.set_xlabel('2D Error (m)', color='white')
                self.ax3.set_ylabel('Frequency', color='white')
                self.ax3.grid(True, alpha=0.3)

            # Update statistics
            if len(self.errors_2d) > 0:
                errors = np.array(self.errors_2d)
                stats_text = f"""
VINS-GPS Trajectory Comparison
==============================

Data Points:
  VINS messages: {self.vins_count}
  GPS messages:  {self.gps_count}
  Error samples: {len(errors)}

Current Position:
  VINS: ({self.vins_x[-1]:.2f}, {self.vins_y[-1]:.2f}) m
  GPS:  ({self.gps_x[-1] if self.gps_x else 0:.2f}, {self.gps_y[-1] if self.gps_y else 0:.2f}) m

Error Statistics:
  Current:  {errors[-1]:.3f} m
  Mean:     {np.mean(errors):.3f} m
  Std Dev:  {np.std(errors):.3f} m
  RMSE:     {np.sqrt(np.mean(errors**2)):.3f} m
  Max:      {np.max(errors):.3f} m
  Min:      {np.min(errors):.3f} m

Runtime: {self.timestamps[-1] if self.timestamps else 0:.1f} s
                """.strip()

                self.stats_text.set_text(stats_text)

        except Exception as e:
            self.get_logger().error(f"Error updating plot: {e}")

        return (self.vins_line, self.gps_line, self.vins_current,
                self.gps_current, self.error_line)

    def start_visualization(self):
        """Start the real-time visualization"""
        try:
            self.ani = animation.FuncAnimation(
                self.fig, self.update_plot, interval=200, blit=False, cache_frame_data=False)
            plt.show()
        except KeyboardInterrupt:
            self.get_logger().info("Visualization stopped by user")
        except Exception as e:
            self.get_logger().error(f"Error in visualization: {e}")


def main():
    rclpy.init()

    try:
        visualizer = VINSGPSTrajectoryVisualizer()

        # Start ROS2 spinning in a separate thread
        ros_thread = threading.Thread(
            target=rclpy.spin, args=(visualizer,), daemon=True)
        ros_thread.start()

        print("🎯 VINS-GPS Trajectory Visualizer")
        print("=================================")
        print("Real-time comparison of VINS odometry vs GPS ground truth")
        print("- VINS trajectory: Cyan line")
        print("- GPS trajectory: Yellow line")
        print("- Real-time error statistics displayed")
        print("Close the plot window to exit...")
        print("")

        # Start visualization (blocks until window is closed)
        visualizer.start_visualization()

    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
