#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import os
from datetime import datetime


def extract_vins_odometry_from_bag(bag_path, output_file):
    """Extract VINS odometry data from ROS2 bag to CSV"""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from nav_msgs.msg import Odometry

        print(f"📦 Extracting VINS odometry data from {bag_path}...")

        storage_options = rosbag2_py.StorageOptions(
            uri=bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        # Get topic information
        topic_types = reader.get_all_topics_and_types()
        print("📋 Available topics in bag:")
        for topic_metadata in topic_types:
            print(f"   {topic_metadata.name} ({topic_metadata.type})")

        # Look for VINS odometry topics
        vins_topics = [
            '/vins_estimator/odometry',
            '/vins_estimator/imu_propagate',
            '/vins/odometry',
            '/odometry/imu',
            '/vins_estimator/path',
            '/estimator/odometry'
        ]

        # Find the actual VINS topic in the bag
        available_topics = [topic.name for topic in topic_types]
        vins_topic = None

        for topic in vins_topics:
            if topic in available_topics:
                vins_topic = topic
                break

        if vins_topic is None:
            print("❌ No VINS odometry topic found. Available topics:")
            for topic in available_topics:
                if 'odom' in topic.lower() or 'vins' in topic.lower():
                    print(f"   {topic}")
            return None

        print(f"✅ Using VINS topic: {vins_topic}")

        vins_data = []
        message_count = 0

        # Read messages from the bag
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()

            if topic == vins_topic:
                try:
                    msg = deserialize_message(data, Odometry)

                    # Extract timestamp (convert nanoseconds to seconds)
                    msg_timestamp = timestamp * 1e-9

                    # Extract position
                    x = msg.pose.pose.position.x
                    y = msg.pose.pose.position.y
                    z = msg.pose.pose.position.z

                    # Extract orientation (quaternion)
                    qx = msg.pose.pose.orientation.x
                    qy = msg.pose.pose.orientation.y
                    qz = msg.pose.pose.orientation.z
                    qw = msg.pose.pose.orientation.w

                    # Extract linear velocity
                    vx = msg.twist.twist.linear.x
                    vy = msg.twist.twist.linear.y
                    vz = msg.twist.twist.linear.z

                    # Extract angular velocity
                    wx = msg.twist.twist.angular.x
                    wy = msg.twist.twist.angular.y
                    wz = msg.twist.twist.angular.z

                    # Convert quaternion to Euler angles (optional)
                    # Roll (x-axis rotation)
                    sinr_cosp = 2 * (qw * qx + qy * qz)
                    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
                    roll = np.arctan2(sinr_cosp, cosr_cosp)

                    # Pitch (y-axis rotation)
                    sinp = 2 * (qw * qy - qz * qx)
                    if abs(sinp) >= 1:
                        # use 90 degrees if out of range
                        pitch = np.copysign(np.pi / 2, sinp)
                    else:
                        pitch = np.arcsin(sinp)

                    # Yaw (z-axis rotation)
                    siny_cosp = 2 * (qw * qz + qx * qy)
                    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                    yaw = np.arctan2(siny_cosp, cosy_cosp)

                    vins_data.append({
                        'timestamp': msg_timestamp,
                        'x': x,
                        'y': y,
                        'z': z,
                        'qx': qx,
                        'qy': qy,
                        'qz': qz,
                        'qw': qw,
                        'roll': roll,
                        'pitch': pitch,
                        'yaw': yaw,
                        'vx': vx,
                        'vy': vy,
                        'vz': vz,
                        'wx': wx,
                        'wy': wy,
                        'wz': wz
                    })

                    message_count += 1

                    if message_count % 100 == 0:
                        print(
                            f"   Processed {message_count} odometry messages...")

                except Exception as e:
                    print(f"⚠️  Error processing message {message_count}: {e}")
                    continue

        # Reader automatically closes when going out of scope

        if len(vins_data) == 0:
            print("❌ No valid VINS odometry data found")
            return None

        # Create DataFrame and save to CSV
        df = pd.DataFrame(vins_data)

        # Sort by timestamp to ensure chronological order
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save to CSV
        df.to_csv(output_file, index=False)

        # Print summary statistics
        duration = df['timestamp'].max() - df['timestamp'].min()
        frequency = len(df) / duration if duration > 0 else 0

        print(f"✅ VINS odometry data saved: {len(df)} points to {output_file}")
        print(f"📊 Summary:")
        print(
            f"   Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"   Frequency: {frequency:.1f} Hz")
        print(f"   Position range:")
        print(f"     X: {df['x'].min():.2f} to {df['x'].max():.2f} m")
        print(f"     Y: {df['y'].min():.2f} to {df['y'].max():.2f} m")
        print(f"     Z: {df['z'].min():.2f} to {df['z'].max():.2f} m")
        print(f"   Total distance: {calculate_trajectory_distance(df):.2f} m")

        return df

    except ImportError:
        print("❌ rosbag2_py not available. Please install ROS2 rosbag2 packages:")
        print("   sudo apt install ros-humble-rosbag2-py")
        return None
    except Exception as e:
        print(f"❌ Error extracting VINS odometry data: {e}")
        return None


def calculate_trajectory_distance(df):
    """Calculate total distance traveled in trajectory"""
    if len(df) < 2:
        return 0.0

    distances = np.sqrt(
        np.diff(df['x'])**2 +
        np.diff(df['y'])**2 +
        np.diff(df['z'])**2
    )
    return np.sum(distances)


def extract_gps_data_from_bag(bag_path, output_file):
    """Extract GPS data from ROS2 bag to CSV (for convenience)"""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import NavSatFix

        print(f"📡 Extracting GPS data from {bag_path}...")

        storage_options = rosbag2_py.StorageOptions(
            uri=bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        gps_data = []
        message_count = 0

        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()

            # Look for GPS topics
            if topic in ['/fix', '/gps/fix', '/navsat/fix', '/ublox_gps/fix']:
                try:
                    msg = deserialize_message(data, NavSatFix)

                    # Only save valid GPS fixes
                    if msg.status.status >= 0:
                        gps_data.append({
                            'timestamp': timestamp * 1e-9,
                            'latitude': msg.latitude,
                            'longitude': msg.longitude,
                            'altitude': msg.altitude,
                            'status': msg.status.status,
                            'service': msg.status.service
                        })
                        message_count += 1

                        if message_count % 50 == 0:
                            print(
                                f"   Processed {message_count} GPS messages...")

                except Exception as e:
                    print(f"⚠️  Error processing GPS message: {e}")
                    continue

        # Reader automatically closes when going out of scope

        if len(gps_data) == 0:
            print("⚠️  No valid GPS data found")
            return None

        # Create DataFrame and save to CSV
        df = pd.DataFrame(gps_data)
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        df.to_csv(output_file, index=False)

        duration = df['timestamp'].max() - df['timestamp'].min()
        frequency = len(df) / duration if duration > 0 else 0

        print(f"✅ GPS data saved: {len(df)} points to {output_file}")
        print(f"📊 GPS Summary:")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Frequency: {frequency:.1f} Hz")
        print(
            f"   Latitude range: {df['latitude'].min():.6f} to {df['latitude'].max():.6f}")
        print(
            f"   Longitude range: {df['longitude'].min():.6f} to {df['longitude'].max():.6f}")
        print(
            f"   Altitude range: {df['altitude'].min():.2f} to {df['altitude'].max():.2f} m")

        return df

    except Exception as e:
        print(f"❌ Error extracting GPS data: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Extract VINS odometry and GPS data from ROS2 bags to CSV files')
    parser.add_argument('bag_path',
                        help='Path to ROS2 bag file (.db3)')
    parser.add_argument('--output-dir', default='./extracted_data',
                        help='Output directory for CSV files (default: ./extracted_data)')
    parser.add_argument('--vins-only', action='store_true',
                        help='Extract only VINS odometry data')
    parser.add_argument('--gps-only', action='store_true',
                        help='Extract only GPS data')
    parser.add_argument('--prefix', default='',
                        help='Prefix for output filenames')

    args = parser.parse_args()

    print("🚁 VINS-GPS Data Extractor for ROS2 Bags")
    print("="*45)
    print(f"📂 Input bag: {args.bag_path}")
    print(f"📁 Output directory: {args.output_dir}")

    # Check if bag file exists
    if not os.path.exists(args.bag_path):
        print(f"❌ Bag file not found: {args.bag_path}")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate output filenames
    bag_name = os.path.splitext(os.path.basename(args.bag_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.prefix:
        vins_output = os.path.join(
            args.output_dir, f"{args.prefix}_{bag_name}_vins_odometry.csv")
        gps_output = os.path.join(
            args.output_dir, f"{args.prefix}_{bag_name}_gps.csv")
    else:
        vins_output = os.path.join(
            args.output_dir, f"{bag_name}_vins_odometry.csv")
        gps_output = os.path.join(args.output_dir, f"{bag_name}_gps.csv")

    success_count = 0

    # Extract VINS odometry data
    if not args.gps_only:
        print(f"\n🧭 Extracting VINS odometry data...")
        vins_df = extract_vins_odometry_from_bag(args.bag_path, vins_output)
        if vins_df is not None:
            success_count += 1

    # Extract GPS data
    if not args.vins_only:
        print(f"\n📡 Extracting GPS data...")
        gps_df = extract_gps_data_from_bag(args.bag_path, gps_output)
        if gps_df is not None:
            success_count += 1

    # Summary
    print(f"\n✅ Extraction completed!")
    print(f"📈 Successfully extracted {success_count} dataset(s)")
    print(f"📁 Files saved to: {args.output_dir}")

    if not args.gps_only and os.path.exists(vins_output):
        print(f"   🧭 VINS: {vins_output}")
    if not args.vins_only and os.path.exists(gps_output):
        print(f"   📡 GPS: {gps_output}")

    print(f"\n🎯 Usage with offline analysis:")
    if not args.gps_only and os.path.exists(vins_output):
        print(f"   python offline_vins_gps_analysis.py \\")
        print(f"     --vins {vins_output} \\")
        if not args.vins_only and os.path.exists(gps_output):
            print(f"     --gps {gps_output} \\")
        else:
            print(f"     --gps {args.bag_path} \\")
        print(f"     --output ./analysis_{bag_name}")


if __name__ == '__main__':
    main()
