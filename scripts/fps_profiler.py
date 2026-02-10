#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud
import time
from collections import deque
from datetime import datetime
import json
import os


class FPSProfiler(Node):
    def __init__(self):
        super().__init__('fps_profiler')
        
        # Initialize subscribers for all VINS topics
        self.topic_subscriptions = {
            'image_mono': self.create_subscription(
                Image, '/camera/image_mono', self.image_callback, 10),
            'image_stereo': self.create_subscription(
                Image, '/camera/image_rect', self.image_stereo_callback, 10),
            'imu': self.create_subscription(
                Imu, '/imu', self.imu_callback, 10),
            'odometry': self.create_subscription(
                Odometry, '/odometry', self.odom_callback, 10),
            'path': self.create_subscription(
                Path, '/path', self.path_callback, 10),
            'pose': self.create_subscription(
                PoseStamped, '/camera_pose', self.pose_callback, 10),
            'point_cloud': self.create_subscription(
                PointCloud, '/point_cloud', self.points_callback, 10),
        }
        
        # FPS tracking structures
        self.topic_stats = {
            'image_mono': {'timestamps': deque(maxlen=300), 'fps': 0.0, 'count': 0, 'active': False},
            'image_stereo': {'timestamps': deque(maxlen=300), 'fps': 0.0, 'count': 0, 'active': False},
            'imu': {'timestamps': deque(maxlen=1000), 'fps': 0.0, 'count': 0, 'active': False},
            'odometry': {'timestamps': deque(maxlen=300), 'fps': 0.0, 'count': 0, 'active': False, 'latencies': deque(maxlen=100)},
            'path': {'timestamps': deque(maxlen=300), 'fps': 0.0, 'count': 0, 'active': False},
            'pose': {'timestamps': deque(maxlen=300), 'fps': 0.0, 'count': 0, 'active': False, 'latencies': deque(maxlen=100)},
            'point_cloud': {'timestamps': deque(maxlen=300), 'fps': 0.0, 'count': 0, 'active': False},
        }
        
        self.start_time = time.time()
        self.last_print_time = time.time()
        self.print_interval = 2.0  # Print stats every 2 seconds
        
        # Setup output directory
        self.output_dir = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'fps_profiling_results'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # File logging
        self.log_file = os.path.join(
            self.output_dir,
            f'fps_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        self.stats_file = os.path.join(
            self.output_dir,
            f'fps_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        
        # Write CSV header
        with open(self.stats_file, 'w') as f:
            f.write('timestamp,image_mono_fps,image_stereo_fps,imu_fps,odometry_fps,path_fps,pose_fps,pointcloud_fps,'
                   'odometry_latency_ms,pose_latency_ms,active_topics\n')
        
        # Create timer for periodic status updates
        self.timer = self.create_timer(self.print_interval, self.update_stats)
        
        self.get_logger().info('='*70)
        self.get_logger().info('FPS PROFILER STARTED')
        self.get_logger().info('='*70)
        self.get_logger().info(f'Logging to: {self.output_dir}')
        self.get_logger().info('Monitoring VINS topics for FPS and latency...')
        self.get_logger().info('='*70)

    def image_callback(self, msg):
        self._update_topic_stats('image_mono', msg.header.stamp)

    def image_stereo_callback(self, msg):
        self._update_topic_stats('image_stereo', msg.header.stamp)

    def imu_callback(self, msg):
        self._update_topic_stats('imu', msg.header.stamp)

    def odom_callback(self, msg):
        latency = self._calculate_latency(msg.header.stamp)
        self._update_topic_stats('odometry', msg.header.stamp, latency)

    def path_callback(self, msg):
        self._update_topic_stats('path', msg.header.stamp)

    def pose_callback(self, msg):
        latency = self._calculate_latency(msg.header.stamp)
        self._update_topic_stats('pose', msg.header.stamp, latency)

    def points_callback(self, msg):
        self._update_topic_stats('point_cloud', msg.header.stamp)

    def _update_topic_stats(self, topic_name, timestamp, latency_ms=None):
        """Update statistics for a given topic."""
        current_time = time.time()
        self.topic_stats[topic_name]['timestamps'].append(current_time)
        self.topic_stats[topic_name]['count'] += 1
        self.topic_stats[topic_name]['active'] = True
        
        # Store latency if provided
        if latency_ms is not None and latency_ms >= 0:
            self.topic_stats[topic_name]['latencies'].append(latency_ms)

    def _calculate_latency(self, msg_timestamp):
        """Calculate latency between message timestamp and current time in milliseconds."""
        try:
            ros_time_sec = msg_timestamp.sec + msg_timestamp.nanosec / 1e9
            current_time = time.time()
            latency_ms = (current_time - ros_time_sec) * 1000
            return max(0, latency_ms)  # Avoid negative values
        except:
            return -1

    def _calculate_fps(self, topic_name):
        """Calculate FPS for a given topic."""
        timestamps = self.topic_stats[topic_name]['timestamps']
        
        if len(timestamps) < 2:
            return 0.0
        
        time_diff = timestamps[-1] - timestamps[0]
        if time_diff <= 0:
            return 0.0
        
        fps = (len(timestamps) - 1) / time_diff
        return fps

    def _get_avg_latency(self, topic_name):
        """Get average latency for a topic."""
        latencies = self.topic_stats[topic_name].get('latencies', deque())
        if not latencies:
            return 0.0
        return sum(latencies) / len(latencies)

    def update_stats(self):
        """Update and display FPS statistics."""
        current_time = time.time()
        
        # Update FPS for all topics
        for topic_name in self.topic_stats:
            self.topic_stats[topic_name]['fps'] = self._calculate_fps(topic_name)
        
        # Only print if at least one topic is active
        active_topics = [t for t in self.topic_stats if self.topic_stats[t]['active']]
        
        if not active_topics:
            return
        
        # Print to console
        elapsed = current_time - self.start_time
        self.get_logger().info(f"\n{'='*70}")
        self.get_logger().info(f"FPS STATISTICS [Elapsed: {elapsed:.1f}s]")
        self.get_logger().info(f"{'='*70}")
        
        # Input sources
        self.get_logger().info("📥 INPUT SOURCES:")
        if self.topic_stats['image_mono']['active']:
            self.get_logger().info(
                f"  • Image (Mono):     {self.topic_stats['image_mono']['fps']:6.2f} FPS "
                f"(total: {self.topic_stats['image_mono']['count']} frames)"
            )
        if self.topic_stats['image_stereo']['active']:
            self.get_logger().info(
                f"  • Image (Stereo):   {self.topic_stats['image_stereo']['fps']:6.2f} FPS "
                f"(total: {self.topic_stats['image_stereo']['count']} frames)"
            )
        if self.topic_stats['imu']['active']:
            self.get_logger().info(
                f"  • IMU:              {self.topic_stats['imu']['fps']:6.2f} FPS "
                f"(total: {self.topic_stats['imu']['count']} samples)"
            )
        
        # Processing outputs
        self.get_logger().info("⚙️  PROCESSING OUTPUTS:")
        if self.topic_stats['odometry']['active']:
            avg_latency = self._get_avg_latency('odometry')
            self.get_logger().info(
                f"  • Odometry:         {self.topic_stats['odometry']['fps']:6.2f} FPS "
                f"(latency: {avg_latency:6.2f}ms, total: {self.topic_stats['odometry']['count']})"
            )
        if self.topic_stats['pose']['active']:
            avg_latency = self._get_avg_latency('pose')
            self.get_logger().info(
                f"  • Camera Pose:      {self.topic_stats['pose']['fps']:6.2f} FPS "
                f"(latency: {avg_latency:6.2f}ms, total: {self.topic_stats['pose']['count']})"
            )
        if self.topic_stats['path']['active']:
            self.get_logger().info(
                f"  • Path:             {self.topic_stats['path']['fps']:6.2f} FPS "
                f"(total: {self.topic_stats['path']['count']})"
            )
        if self.topic_stats['point_cloud']['active']:
            self.get_logger().info(
                f"  • Point Cloud:      {self.topic_stats['point_cloud']['fps']:6.2f} FPS "
                f"(total: {self.topic_stats['point_cloud']['count']})"
            )
        
        # Bottleneck analysis
        self.get_logger().info("🔍 BOTTLENECK ANALYSIS:")
        input_fps = [self.topic_stats[t]['fps'] for t in ['image_mono', 'image_stereo', 'imu'] 
                     if self.topic_stats[t]['active']]
        output_fps = [self.topic_stats[t]['fps'] for t in ['odometry', 'pose'] 
                      if self.topic_stats[t]['active']]
        
        if input_fps and output_fps:
            avg_input = sum(input_fps) / len(input_fps)
            avg_output = sum(output_fps) / len(output_fps)
            throughput_ratio = avg_output / avg_input if avg_input > 0 else 0
            self.get_logger().info(f"  • Avg Input FPS:    {avg_input:.2f}")
            self.get_logger().info(f"  • Avg Output FPS:   {avg_output:.2f}")
            self.get_logger().info(f"  • Throughput Ratio: {throughput_ratio:.2%}")
            
            if throughput_ratio < 0.5:
                self.get_logger().warn("  ⚠️  WARNING: Low throughput ratio detected!")
            elif throughput_ratio < 0.8:
                self.get_logger().warn("  ⚠️  INFO: Moderate processing lag detected")
        
        self.get_logger().info(f"{'='*70}\n")
        
        # Log to file
        self._log_to_file()

    def _log_to_file(self):
        """Log statistics to CSV and JSON files."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # CSV logging
        csv_line = f"{current_time},"
        csv_line += f"{self.topic_stats['image_mono']['fps']:.2f},"
        csv_line += f"{self.topic_stats['image_stereo']['fps']:.2f},"
        csv_line += f"{self.topic_stats['imu']['fps']:.2f},"
        csv_line += f"{self.topic_stats['odometry']['fps']:.2f},"
        csv_line += f"{self.topic_stats['path']['fps']:.2f},"
        csv_line += f"{self.topic_stats['pose']['fps']:.2f},"
        csv_line += f"{self.topic_stats['point_cloud']['fps']:.2f},"
        csv_line += f"{self._get_avg_latency('odometry'):.2f},"
        csv_line += f"{self._get_avg_latency('pose'):.2f},"
        csv_line += f"{len([t for t in self.topic_stats if self.topic_stats[t]['active']])}"
        csv_line += "\n"
        
        with open(self.stats_file, 'a') as f:
            f.write(csv_line)
        
        # JSON logging (for more detailed analysis)
        json_data = {
            'timestamp': current_time,
            'elapsed_seconds': time.time() - self.start_time,
            'fps_stats': {}
        }
        
        for topic_name, stats in self.topic_stats.items():
            json_data['fps_stats'][topic_name] = {
                'fps': stats['fps'],
                'count': stats['count'],
                'active': stats['active'],
                'avg_latency_ms': self._get_avg_latency(topic_name)
            }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(json_data) + '\n')


def main(args=None):
    rclpy.init(args=args)
    profiler = FPSProfiler()
    
    try:
        rclpy.spin(profiler)
    except KeyboardInterrupt:
        profiler.get_logger().info('\n' + '='*70)
        profiler.get_logger().info('FPS PROFILER SHUTDOWN')
        profiler.get_logger().info('='*70)
        
        # Print final summary
        profiler.get_logger().info("\n📊 FINAL SUMMARY:")
        profiler.get_logger().info(f"Total runtime: {time.time() - profiler.start_time:.1f}s")
        profiler.get_logger().info(f"Results saved to: {profiler.output_dir}")
        profiler.get_logger().info(f"  • CSV: {os.path.basename(profiler.stats_file)}")
        profiler.get_logger().info(f"  • JSON: {os.path.basename(profiler.log_file)}")
        profiler.get_logger().info('='*70)
    finally:
        profiler.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
