#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, CompressedImage, PointCloud2
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import numpy as np
import math
import cv2
import os
from collections import deque
import threading
from datetime import datetime


class VINSGPSTrajectoryVisualizer(Node):
    def __init__(self):
        super().__init__('vins_gps_visualizer')

        # Data storage
        self.max_points = 500
        self.vins_x = deque(maxlen=self.max_points)
        self.vins_y = deque(maxlen=self.max_points)
        self.vins_z = deque(maxlen=self.max_points)
        self.gps_x = deque(maxlen=self.max_points)
        self.gps_y = deque(maxlen=self.max_points)
        self.gps_z = deque(maxlen=self.max_points)
        self.errors_2d = deque(maxlen=self.max_points)
        self.errors_3d = deque(maxlen=self.max_points)
        self.timestamps = deque(maxlen=self.max_points)

        # Camera and feature data
        self.current_image = None
        self.current_features = None
        self.image_timestamp = None
        self.feature_points = []
        self.tracking_quality = 0.0

        # GPS reference point
        self.gps_ref_lat = None
        self.gps_ref_lon = None
        self.gps_ref_alt = None

        # Trajectory alignment
        self.trajectory_aligned = False
        self.alignment_rotation = 0.0
        self.last_alignment_check = 0
        self.heading_aligned = False
        self.altitude_aligned = False
        self.altitude_offset = 0.0
        self.min_points_for_heading_alignment = 30
        self.min_points_for_altitude_alignment = 20

        # Figure saving
        self.save_figures = True
        self.output_dir = './visualization_output'
        self.figure_counter = 0
        self.save_interval = 100  # Save every 100 updates

        # Statistics
        self.vins_count = 0
        self.gps_count = 0
        self.image_count = 0
        self.feature_count = 0
        self.start_time = None

        # System status
        self.vins_active = False
        self.gps_active = False
        self.camera_active = False

        # Subscribers
        self.vins_sub = self.create_subscription(
            Odometry, '/vins_estimator/odometry', self.vins_callback, 10)
        self.gps_sub = self.create_subscription(
            NavSatFix, '/fix', self.gps_callback, 10)
        self.image_sub = self.create_subscription(
            CompressedImage, '/camera/image_color/compressed', self.image_callback, 10)
        self.feature_sub = self.create_subscription(
            PointCloud2, '/vins_estimator/point_cloud', self.feature_callback, 10)

        # Setup matplotlib
        self.setup_enhanced_plot()

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

    def align_altitude(self):
        """Align VINS altitude with GPS altitude using mean offset"""
        if (self.altitude_aligned or
            len(self.vins_z) < self.min_points_for_altitude_alignment or
                len(self.gps_z) < self.min_points_for_altitude_alignment):
            return False

        # Calculate mean altitude difference
        vins_z_array = np.array(list(self.vins_z))
        gps_z_array = np.array(list(self.gps_z)[-len(vins_z_array):])

        # Calculate offset to align mean altitudes
        mean_vins_z = np.mean(vins_z_array)
        mean_gps_z = np.mean(gps_z_array)
        self.altitude_offset = mean_gps_z - mean_vins_z

        # Apply offset to all existing VINS altitude data
        aligned_z = []
        for z in self.vins_z:
            aligned_z.append(z + self.altitude_offset)

        # Update deque with aligned altitudes
        self.vins_z.clear()
        for z in aligned_z:
            self.vins_z.append(z)

        self.altitude_aligned = True
        print(
            f"📏 Altitude aligned: {self.altitude_offset:.2f}m offset applied")
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
        z = msg.pose.pose.position.z

        # Apply heading alignment if already determined
        if self.heading_aligned:
            cos_a, sin_a = math.cos(self.alignment_rotation), math.sin(
                self.alignment_rotation)
            x_rotated = cos_a * x - sin_a * y
            y_rotated = sin_a * x + cos_a * y
            x, y = x_rotated, y_rotated

        # Apply altitude alignment if already determined
        if self.altitude_aligned:
            z = z + self.altitude_offset

        # Apply trajectory alignment if already determined (for legacy support)
        if self.trajectory_aligned and not self.heading_aligned:
            angle_rad = np.radians(self.alignment_rotation)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

            x_rotated = cos_a * x - sin_a * y
            y_rotated = sin_a * x + cos_a * y
            x, y = x_rotated, y_rotated

        self.vins_x.append(x)
        self.vins_y.append(y)
        self.vins_z.append(z)
        self.timestamps.append(rel_time)
        self.vins_count += 1

        if self.vins_count == 1:
            self.get_logger().info("✅ First VINS message received")

        # Try initial heading alignment first
        if not self.heading_aligned:
            self.align_initial_heading()

        # Try altitude alignment
        if not self.altitude_aligned:
            self.align_altitude()

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
        self.gps_z.append(z)
        self.gps_count += 1

        if self.gps_count == 1:
            self.get_logger().info("✅ First GPS message received")

        # Try initial heading alignment when we have enough data
        if not self.heading_aligned:
            self.align_initial_heading()

        # Try altitude alignment when we have enough data
        if not self.altitude_aligned:
            self.align_altitude()

        # Calculate 2D and 3D errors if we have VINS data
        if len(self.vins_x) > 0 and len(self.gps_x) > 0:
            error_2d = math.sqrt(
                (self.vins_x[-1] - x)**2 + (self.vins_y[-1] - y)**2)
            self.errors_2d.append(error_2d)

            if len(self.vins_z) > 0:
                error_3d = math.sqrt(
                    (self.vins_x[-1] - x)**2 + (self.vins_y[-1] - y)**2 + (self.vins_z[-1] - z)**2)
                self.errors_3d.append(error_3d)

    def image_callback(self, msg):
        """Process camera images"""
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_image is not None:
                # Convert BGR to RGB for matplotlib
                self.current_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                self.image_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                self.image_count += 1
                self.camera_active = True

                # Extract basic features using corner detection
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                corners = cv2.goodFeaturesToTrack(gray, maxCorners=100,
                                                  qualityLevel=0.01, minDistance=10)

                if corners is not None:
                    self.feature_points = corners.reshape(-1, 2)
                    self.feature_count = len(self.feature_points)
                    self.tracking_quality = min(
                        len(self.feature_points) / 100.0, 1.0)
                else:
                    self.feature_points = []
                    self.feature_count = 0
                    self.tracking_quality = 0.0

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def feature_callback(self, msg):
        """Process VINS feature points (if available)"""
        try:
            # This would parse PointCloud2 data for 3D features
            # For now, we'll rely on image-based feature detection
            pass
        except Exception as e:
            self.get_logger().error(f"Error processing features: {e}")

    def setup_enhanced_plot(self):
        """Setup separate matplotlib figures for independent saving"""
        plt.style.use('dark_background')

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Figure 1: Trajectory Comparison
        self.fig_trajectory = plt.figure(figsize=(12, 8))
        self.ax_trajectory = self.fig_trajectory.add_subplot(111)
        self.ax_trajectory.set_title(
            'Flight Trajectories Comparison', fontsize=14, color='white')
        self.ax_trajectory.set_xlabel(
            'X Position (m)', color='white', fontsize=12)
        self.ax_trajectory.set_ylabel(
            'Y Position (m)', color='white', fontsize=12)
        self.ax_trajectory.grid(True, alpha=0.3)
        self.ax_trajectory.set_aspect('equal')

        # Figure 2: Error Analysis
        self.fig_error = plt.figure(figsize=(12, 8))
        self.ax_error = self.fig_error.add_subplot(2, 1, 1)
        self.ax_error.set_title(
            '2D Position Error Over Time', fontsize=14, color='white')
        self.ax_error.set_xlabel('Time (s)', color='white', fontsize=12)
        self.ax_error.set_ylabel('2D Error (m)', color='white', fontsize=12)
        self.ax_error.grid(True, alpha=0.3)

        self.ax_error_3d = self.fig_error.add_subplot(2, 1, 2)
        self.ax_error_3d.set_title(
            '3D Position Error Over Time', fontsize=14, color='white')
        self.ax_error_3d.set_xlabel('Time (s)', color='white', fontsize=12)
        self.ax_error_3d.set_ylabel('3D Error (m)', color='white', fontsize=12)
        self.ax_error_3d.grid(True, alpha=0.3)

        # Figure 3: Altitude Comparison
        self.fig_altitude = plt.figure(figsize=(12, 8))
        self.ax_altitude = self.fig_altitude.add_subplot(111)
        self.ax_altitude.set_title(
            'Altitude Comparison', fontsize=14, color='white')
        self.ax_altitude.set_xlabel('Time (s)', color='white', fontsize=12)
        self.ax_altitude.set_ylabel('Altitude (m)', color='white', fontsize=12)
        self.ax_altitude.grid(True, alpha=0.3)

        # Figure 4: Camera Feed
        self.fig_camera = plt.figure(figsize=(10, 8))
        self.ax_camera = self.fig_camera.add_subplot(111)
        self.ax_camera.set_title(
            'Live Camera Feed with Features', fontsize=14, color='white')
        self.ax_camera.axis('off')

        # Figure 5: Statistics Dashboard
        self.fig_stats = plt.figure(figsize=(12, 10))
        self.ax_status = self.fig_stats.add_subplot(2, 2, 1)
        self.ax_status.set_title('System Status', fontsize=12, color='white')
        self.ax_status.axis('off')

        self.ax_hist = self.fig_stats.add_subplot(2, 2, 2)
        self.ax_hist.set_title('Error Distribution',
                               fontsize=12, color='white')
        self.ax_hist.set_xlabel('2D Error (m)', color='white', fontsize=10)
        self.ax_hist.set_ylabel('Frequency', color='white', fontsize=10)
        self.ax_hist.grid(True, alpha=0.3)

        self.ax_stats_display = self.fig_stats.add_subplot(2, 1, 2)
        self.ax_stats_display.set_title(
            'Real-time Statistics', fontsize=12, color='white')
        self.ax_stats_display.axis('off')

        # Initialize trajectory plots
        self.vins_line, = self.ax_trajectory.plot(
            [], [], 'cyan', linewidth=2, label='VINS', alpha=0.8)
        self.gps_line, = self.ax_trajectory.plot(
            [], [], 'yellow', linewidth=2, label='GPS', alpha=0.8)
        self.vins_current, = self.ax_trajectory.plot(
            [], [], 'co', markersize=6, label='VINS Current')
        self.gps_current, = self.ax_trajectory.plot(
            [], [], 'yo', markersize=6, label='GPS Current')
        self.ax_trajectory.legend(loc='upper right', fontsize=10)

        # Initialize error plots
        self.error_line_2d, = self.ax_error.plot(
            [], [], 'red', linewidth=2, alpha=0.8, label='2D Error')
        self.error_line_3d, = self.ax_error_3d.plot(
            [], [], 'purple', linewidth=2, alpha=0.8, label='3D Error')

        # Initialize altitude plots
        self.vins_altitude_line, = self.ax_altitude.plot(
            [], [], 'cyan', linewidth=2, label='VINS Altitude', alpha=0.8)
        self.gps_altitude_line, = self.ax_altitude.plot(
            [], [], 'yellow', linewidth=2, label='GPS Altitude', alpha=0.8)
        self.ax_altitude.legend(loc='upper right', fontsize=10)

        # Initialize status indicators
        self.status_text = self.ax_status.text(0.05, 0.95, '', transform=self.ax_status.transAxes,
                                               fontsize=9, color='white', verticalalignment='top',
                                               fontfamily='monospace')

        # Initialize statistics text
        self.stats_text = self.ax_stats_display.text(0.05, 0.95, '', transform=self.ax_stats_display.transAxes,
                                                     fontsize=9, color='white', verticalalignment='top',
                                                     fontfamily='monospace')

        # Adjust layouts
        self.fig_trajectory.tight_layout()
        self.fig_error.tight_layout()
        self.fig_altitude.tight_layout()
        self.fig_camera.tight_layout()
        self.fig_stats.tight_layout()

    def save_individual_figures(self):
        """Save each figure as a separate file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save trajectory comparison
            trajectory_file = os.path.join(
                self.output_dir, f'trajectory_comparison_{timestamp}.png')
            self.fig_trajectory.savefig(trajectory_file, dpi=300, bbox_inches='tight',
                                        facecolor='black', edgecolor='none')

            # Save error analysis
            error_file = os.path.join(
                self.output_dir, f'error_analysis_{timestamp}.png')
            self.fig_error.savefig(error_file, dpi=300, bbox_inches='tight',
                                   facecolor='black', edgecolor='none')

            # Save altitude comparison
            altitude_file = os.path.join(
                self.output_dir, f'altitude_comparison_{timestamp}.png')
            self.fig_altitude.savefig(altitude_file, dpi=300, bbox_inches='tight',
                                      facecolor='black', edgecolor='none')

            # Save camera feed
            camera_file = os.path.join(
                self.output_dir, f'camera_feed_{timestamp}.png')
            self.fig_camera.savefig(camera_file, dpi=300, bbox_inches='tight',
                                    facecolor='black', edgecolor='none')

            # Save statistics dashboard
            stats_file = os.path.join(
                self.output_dir, f'statistics_dashboard_{timestamp}.png')
            self.fig_stats.savefig(stats_file, dpi=300, bbox_inches='tight',
                                   facecolor='black', edgecolor='none')

            self.get_logger().info(f"📁 Figures saved to: {self.output_dir}")

        except Exception as e:
            self.get_logger().error(f"Error saving figures: {e}")

    def update_plot(self, frame):
        """Update all separate figures with new data"""
        try:
            self.figure_counter += 1

            # Update camera feed with features
            if self.current_image is not None:
                self.ax_camera.clear()
                self.ax_camera.imshow(self.current_image)
                self.ax_camera.set_title(
                    f'Live Camera Feed - Features: {self.feature_count} | Quality: {self.tracking_quality:.1%}',
                    fontsize=12, color='white')
                self.ax_camera.axis('off')

                # Draw feature points
                if len(self.feature_points) > 0:
                    # Color features based on tracking quality
                    colors = ['red' if self.tracking_quality < 0.3 else
                              'yellow' if self.tracking_quality < 0.7 else 'lime'
                              for _ in self.feature_points]

                    self.ax_camera.scatter(self.feature_points[:, 0],
                                           self.feature_points[:, 1],
                                           c=colors, s=20, alpha=0.8)

                # Add tracking quality indicator
                quality_color = 'red' if self.tracking_quality < 0.3 else \
                    'yellow' if self.tracking_quality < 0.7 else 'lime'
                self.ax_camera.add_patch(Circle((50, 50), 20,
                                                color=quality_color, alpha=0.7))

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
                    self.ax_trajectory.set_xlim(
                        min(all_x) - margin, max(all_x) + margin)
                    self.ax_trajectory.set_ylim(
                        min(all_y) - margin, max(all_y) + margin)

            # Update 2D error plot
            if len(self.errors_2d) > 0 and len(self.timestamps) > 0:
                error_array = np.array(self.errors_2d)
                time_array = np.array(
                    list(self.timestamps)[-len(error_array):])
                self.error_line_2d.set_data(time_array, error_array)

                if len(time_array) > 1:
                    self.ax_error.set_xlim(0, max(time_array) + 1)
                    self.ax_error.set_ylim(0, max(max(error_array) * 1.1, 1.0))

            # Update 3D error plot
            if len(self.errors_3d) > 0 and len(self.timestamps) > 0:
                error_3d_array = np.array(self.errors_3d)
                time_array = np.array(
                    list(self.timestamps)[-len(error_3d_array):])
                self.error_line_3d.set_data(time_array, error_3d_array)

                if len(time_array) > 1:
                    self.ax_error_3d.set_xlim(0, max(time_array) + 1)
                    self.ax_error_3d.set_ylim(
                        0, max(max(error_3d_array) * 1.1, 1.0))

            # Update altitude plots
            if len(self.vins_z) > 0 and len(self.timestamps) > 0:
                vins_z_array = np.array(self.vins_z)
                time_array = np.array(
                    list(self.timestamps)[-len(vins_z_array):])
                self.vins_altitude_line.set_data(time_array, vins_z_array)

            if len(self.gps_z) > 0 and len(self.timestamps) > 0:
                gps_z_array = np.array(self.gps_z)
                time_array = np.array(
                    list(self.timestamps)[-len(gps_z_array):])
                self.gps_altitude_line.set_data(time_array, gps_z_array)

            # Auto-scale altitude plot
            if len(self.vins_z) > 0 or len(self.gps_z) > 0:
                all_z = list(self.vins_z) + list(self.gps_z)
                all_times = list(self.timestamps)

                if all_z and all_times:
                    self.ax_altitude.set_xlim(0, max(all_times) + 1)
                    z_margin = (max(all_z) - min(all_z)) * \
                        0.1 if len(all_z) > 1 else 1.0
                    self.ax_altitude.set_ylim(
                        min(all_z) - z_margin, max(all_z) + z_margin)

            # Update error histogram
            if len(self.errors_2d) > 10:
                self.ax_hist.clear()
                self.ax_hist.hist(self.errors_2d, bins=15, alpha=0.7,
                                  color='red', edgecolor='white')
                self.ax_hist.set_title(
                    'Error Distribution', fontsize=12, color='white')
                self.ax_hist.set_xlabel(
                    '2D Error (m)', color='white', fontsize=10)
                self.ax_hist.set_ylabel(
                    'Frequency', color='white', fontsize=10)
                self.ax_hist.grid(True, alpha=0.3)

            # Update system status
            current_time = datetime.now().strftime("%H:%M:%S")

            # Check system activity
            self.vins_active = self.vins_count > 0
            self.gps_active = self.gps_count > 0

            status_text = f"""
SYSTEM STATUS ({current_time})
{'='*25}

Sensors:
  🎥 Camera:  {'🟢 ACTIVE' if self.camera_active else '🔴 INACTIVE'}
  🧭 VINS:    {'🟢 ACTIVE' if self.vins_active else '🔴 INACTIVE'}
  📡 GPS:     {'🟢 ACTIVE' if self.gps_active else '🔴 INACTIVE'}

Alignment Status:
  🎯 Heading: {'✅ ALIGNED' if self.heading_aligned else '⏳ PENDING'}
  📏 Altitude: {'✅ ALIGNED' if self.altitude_aligned else '⏳ PENDING'}
  🔄 Trajectory: {'✅ ALIGNED' if self.trajectory_aligned else '⏳ PENDING'}

Image Processing:
  📸 Images: {self.image_count:,}
  🎯 Features: {self.feature_count}
  📊 Quality: {self.tracking_quality:.1%}
            """.strip()

            self.status_text.set_text(status_text)

            # Update statistics
            if len(self.errors_2d) > 0:
                errors_2d = np.array(self.errors_2d)
                errors_3d = np.array(self.errors_3d) if len(
                    self.errors_3d) > 0 else np.array([])

                current_alt_vins = self.vins_z[-1] if len(
                    self.vins_z) > 0 else 0
                current_alt_gps = self.gps_z[-1] if len(self.gps_z) > 0 else 0

                stats_text = f"""
TRAJECTORY ANALYSIS
{'='*35}

Data Points:
  📊 VINS messages: {self.vins_count:,}
  📡 GPS messages:  {self.gps_count:,}
  📏 Error samples: {len(errors_2d):,}

Current Position:
  🧭 VINS: ({self.vins_x[-1]:.2f}, {self.vins_y[-1]:.2f}, {current_alt_vins:.2f}) m
  📡 GPS:  ({self.gps_x[-1] if self.gps_x else 0:.2f}, {self.gps_y[-1] if self.gps_y else 0:.2f}, {current_alt_gps:.2f}) m

2D Error Metrics:
  📏 Current:  {errors_2d[-1]:.3f} m
  📊 Mean:     {np.mean(errors_2d):.3f} m
  📈 Std Dev:  {np.std(errors_2d):.3f} m
  🎯 RMSE:     {np.sqrt(np.mean(errors_2d**2)):.3f} m
  ⬆️  Max:      {np.max(errors_2d):.3f} m

3D Error Metrics:
  📏 Current:  {errors_3d[-1]:.3f} m if len(errors_3d) > 0 else 'N/A'
  📊 RMSE 3D:  {np.sqrt(np.mean(errors_3d**2)):.3f} m if len(errors_3d) > 0 else 'N/A'

Altitude Info:
  📏 Offset:   {self.altitude_offset:.2f} m
  🎯 Aligned:  {'YES' if self.altitude_aligned else 'NO'}

⏱️  Runtime: {self.timestamps[-1] if self.timestamps else 0:.1f} s
                """.strip()
            else:
                stats_text = f"""
TRAJECTORY ANALYSIS
{'='*35}

Data Points:
  📊 VINS messages: {self.vins_count:,}
  📡 GPS messages:  {self.gps_count:,}
  📏 Error samples: 0

Status: Waiting for data...
⏱️  Runtime: {self.timestamps[-1] if self.timestamps else 0:.1f} s
                """.strip()

            self.stats_text.set_text(stats_text)

            # Save figures periodically
            if self.save_figures and self.figure_counter % self.save_interval == 0:
                self.save_individual_figures()

        except Exception as e:
            self.get_logger().error(f"Error updating plots: {e}")

        return (self.vins_line, self.gps_line, self.vins_current,
                self.gps_current, self.error_line_2d, self.error_line_3d,
                self.vins_altitude_line, self.gps_altitude_line)

    def start_visualization(self):
        """Start the real-time visualization with separate figures"""
        try:
            # Create animations for all figures
            self.ani_trajectory = animation.FuncAnimation(
                self.fig_trajectory, self.update_plot, interval=200, blit=False, cache_frame_data=False)
            self.ani_error = animation.FuncAnimation(
                self.fig_error, self.update_plot, interval=200, blit=False, cache_frame_data=False)
            self.ani_altitude = animation.FuncAnimation(
                self.fig_altitude, self.update_plot, interval=200, blit=False, cache_frame_data=False)
            self.ani_camera = animation.FuncAnimation(
                self.fig_camera, self.update_plot, interval=200, blit=False, cache_frame_data=False)
            self.ani_stats = animation.FuncAnimation(
                self.fig_stats, self.update_plot, interval=200, blit=False, cache_frame_data=False)

            # Show all figures
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

        print("🎯 Enhanced VINS-GPS Trajectory Visualizer")
        print("==========================================")
        print("Real-time comprehensive analysis with:")
        print("📸 Live camera feed with feature tracking")
        print("🧭 VINS vs GPS trajectory comparison")
        print("📊 Real-time error analysis and statistics")
        print("📏 Altitude comparison and alignment")
        print("🎯 Automatic heading alignment")
        print("📈 System status monitoring")
        print("💾 Automatic figure saving")
        print("")
        print("Features:")
        print("- Cyan line: VINS trajectory")
        print("- Yellow line: GPS trajectory")
        print("- Separate figures for different analyses")
        print("- Automatic altitude alignment")
        print("- Independent figure saving every 100 updates")
        print("- 2D and 3D error tracking")
        print("")
        print("Output directory: ./visualization_output")
        print("Close any plot window to exit...")
        print("")

        # Start visualization (blocks until window is closed)
        visualizer.start_visualization()

    except KeyboardInterrupt:
        print("\n⏹️  Visualization stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
