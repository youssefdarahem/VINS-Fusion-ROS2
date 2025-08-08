#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
import threading


class RealtimePathVisualizer(Node):
    def __init__(self):
        super().__init__('realtime_path_visualizer')

        # Data storage
        self.max_points = 2000  # Maximum number of points to display
        self.x_data = deque(maxlen=self.max_points)
        self.y_data = deque(maxlen=self.max_points)
        self.z_data = deque(maxlen=self.max_points)
        self.timestamps = deque(maxlen=self.max_points)

        # Statistics
        self.total_distance = 0.0
        self.last_position = None
        self.start_time = None

        # Subscriber
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry', self.odom_callback, 10)

        self.get_logger().info("Real-time Path Visualizer started. Waiting for odometry data...")

        # Initialize matplotlib
        self.setup_plot()

    def setup_plot(self):
        """Setup the matplotlib figure and axes"""
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Main path plot (top-down view)
        self.ax1.set_title('Flight Path (Top-Down View)',
                           fontsize=14, color='white')
        self.ax1.set_xlabel('X Position (m)', color='white')
        self.ax1.set_ylabel('Y Position (m)', color='white')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_aspect('equal')

        # Altitude plot
        self.ax2.set_title('Altitude Over Time', fontsize=14, color='white')
        self.ax2.set_xlabel('Time (s)', color='white')
        self.ax2.set_ylabel('Z Position (m)', color='white')
        self.ax2.grid(True, alpha=0.3)

        # Initialize empty plots
        self.path_line, = self.ax1.plot(
            [], [], 'cyan', linewidth=2, alpha=0.8, label='Flight Path')
        self.current_pos, = self.ax1.plot(
            [], [], 'ro', markersize=8, label='Current Position')
        self.start_pos, = self.ax1.plot(
            [], [], 'go', markersize=10, label='Start Position')

        self.altitude_line, = self.ax2.plot(
            [], [], 'yellow', linewidth=2, alpha=0.8)

        # Add legends
        self.ax1.legend(loc='lower right')

        # Add text for statistics
        self.stats_text = self.fig.text(0.02, 0.02, '', fontsize=10, color='white',
                                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

        plt.tight_layout()

    def odom_callback(self, msg):
        """Callback for odometry messages"""
        try:
            # Extract position
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            z = msg.pose.pose.position.z

            # Get timestamp
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            if self.start_time is None:
                self.start_time = timestamp
                self.get_logger().info(f"✅ First odometry received! Starting visualization...")

            # Relative time
            rel_time = timestamp - self.start_time

            # Add to data
            self.x_data.append(x)
            self.y_data.append(y)
            self.z_data.append(z)
            self.timestamps.append(rel_time)

            # Calculate distance traveled
            if self.last_position is not None:
                dx = x - self.last_position[0]
                dy = y - self.last_position[1]
                dz = z - self.last_position[2]
                distance = np.sqrt(dx**2 + dy**2 + dz**2)
                self.total_distance += distance

            self.last_position = (x, y, z)

        except Exception as e:
            self.get_logger().error(f"Error in odometry callback: {e}")

    def update_plot(self, frame):
        """Update the plot with new data"""
        if len(self.x_data) == 0:
            return self.path_line, self.current_pos, self.start_pos, self.altitude_line

        try:
            # Convert to numpy arrays
            x_array = np.array(self.x_data)
            y_array = np.array(self.y_data)
            z_array = np.array(self.z_data)
            time_array = np.array(self.timestamps)

            # Update path plot
            self.path_line.set_data(x_array, y_array)

            # Update current position
            if len(x_array) > 0:
                self.current_pos.set_data([x_array[-1]], [y_array[-1]])

                # Mark start position
                if len(x_array) > 1:
                    self.start_pos.set_data([x_array[0]], [y_array[0]])

            # Update altitude plot
            self.altitude_line.set_data(time_array, z_array)

            # Auto-scale axes
            if len(x_array) > 1:
                # Path plot bounds with margin
                x_margin = (np.max(x_array) - np.min(x_array)) * 0.1
                y_margin = (np.max(y_array) - np.min(y_array)) * 0.1

                self.ax1.set_xlim(np.min(x_array) - x_margin,
                                  np.max(x_array) + x_margin)
                self.ax1.set_ylim(np.min(y_array) - y_margin,
                                  np.max(y_array) + y_margin)

                # Altitude plot bounds
                z_margin = (np.max(z_array) - np.min(z_array)) * 0.1
                self.ax2.set_xlim(0, np.max(time_array) + 1)
                self.ax2.set_ylim(np.min(z_array) - z_margin,
                                  np.max(z_array) + z_margin)

            # Update statistics
            if len(x_array) > 0:
                stats = f"Points: {len(x_array)}\n"
                stats += f"Distance: {self.total_distance:.2f} m\n"
                stats += f"Current Position:\n"
                stats += f"  X: {x_array[-1]:.2f} m\n"
                stats += f"  Y: {y_array[-1]:.2f} m\n"
                stats += f"  Z: {z_array[-1]:.2f} m\n"
                stats += f"Runtime: {time_array[-1]:.1f} s"

                self.stats_text.set_text(stats)

        except Exception as e:
            self.get_logger().error(f"Error updating plot: {e}")

        return self.path_line, self.current_pos, self.start_pos, self.altitude_line

    def start_visualization(self):
        """Start the real-time visualization"""
        try:
            # Create animation
            self.ani = animation.FuncAnimation(
                self.fig, self.update_plot, interval=100, blit=False, cache_frame_data=False)

            # Show plot
            plt.show()

        except KeyboardInterrupt:
            self.get_logger().info("Visualization stopped by user")
        except Exception as e:
            self.get_logger().error(f"Error in visualization: {e}")


def main():
    rclpy.init()

    try:
        visualizer = RealtimePathVisualizer()

        # Start ROS2 spinning in a separate thread
        ros_thread = threading.Thread(
            target=rclpy.spin, args=(visualizer,), daemon=True)
        ros_thread.start()

        # Start visualization (this will block until window is closed)
        visualizer.start_visualization()

    except KeyboardInterrupt:
        print("\nVisualizer stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
