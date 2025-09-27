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
            '/Odometry',  # This bag uses /Odometry
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
            print("❌ No VINS odometry topic found. Available odometry topics:")
            odom_topics = [
                t.name for t in topic_types if 'Odometry' in t.type or 'odometry' in t.name.lower()]
            for topic in odom_topics:
                print(f"   {topic}")
            return None

        print(f"✅ Using VINS topic: {vins_topic}")

        vins_data = []
        message_count = 0

        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()

            if topic == vins_topic:
                try:
                    msg = deserialize_message(data, Odometry)

                    # Extract position
                    position = msg.pose.pose.position

                    # Extract orientation (quaternion)
                    orientation = msg.pose.pose.orientation

                    # Extract linear velocity
                    linear_vel = msg.twist.twist.linear

                    # Extract angular velocity
                    angular_vel = msg.twist.twist.angular

                    vins_data.append({
                        'timestamp': timestamp * 1e-9,  # Convert to seconds
                        'x': position.x,
                        'y': position.y,
                        'z': position.z,
                        'qx': orientation.x,
                        'qy': orientation.y,
                        'qz': orientation.z,
                        'qw': orientation.w,
                        'vx': linear_vel.x,
                        'vy': linear_vel.y,
                        'vz': linear_vel.z,
                        'wx': angular_vel.x,
                        'wy': angular_vel.y,
                        'wz': angular_vel.z
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

        # Calculate trajectory statistics
        total_distance = calculate_trajectory_distance(df)
        duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
        avg_speed = total_distance / duration if duration > 0 else 0

        print(f"✅ VINS extraction completed:")
        print(f"   📊 Messages: {len(df):,}")
        print(f"   ⏱️  Duration: {duration:.1f} seconds")
        print(f"   📏 Distance: {total_distance:.1f} meters")
        print(f"   🚀 Avg Speed: {avg_speed:.2f} m/s")

        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"✅ VINS data saved to: {output_file}")

        return df

    except ImportError:
        print("❌ rosbag2_py not available. Please install ros2 rosbag2 packages.")
        return None
    except Exception as e:
        print(f"❌ Error extracting VINS data: {e}")
        return None


def calculate_trajectory_distance(df):
    """Calculate total trajectory distance"""
    if len(df) < 2:
        return 0.0

    distances = np.sqrt(
        np.diff(df['x'])**2 +
        np.diff(df['y'])**2 +
        np.diff(df['z'])**2
    )
    return np.sum(distances)


def extract_filtered_gps_from_bag(bag_path, output_file, gps_source='auto'):
    """Extract filtered GPS data from ROS2 bag using better sources than raw /fix"""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from nav_msgs.msg import Path
        from sensor_msgs.msg import NavSatFix

        print(f"📡 Extracting GPS data from {bag_path}...")
        print(f"🎯 GPS source: {gps_source}")

        storage_options = rosbag2_py.StorageOptions(
            uri=bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        # Get available topics
        topic_types = reader.get_all_topics_and_types()
        available_topics = [topic.name for topic in topic_types]

        # Define GPS source preferences (best to worst)
        gps_sources = {
            # PPK corrected GPS (most accurate)
            'ppk': '/globalEstimator/ppk_path',
            'filtered': '/globalEstimator/gps_path',  # Filtered GPS path
            'global': '/globalEstimator/global_path',  # Global fused estimate
            'raw': '/fix'                            # Raw GPS (least stable)
        }

        # Auto-select best available GPS source
        if gps_source == 'auto':
            selected_topic = None
            selected_source = 'raw'  # fallback

            for source_name, topic_name in gps_sources.items():
                if topic_name in available_topics:
                    selected_topic = topic_name
                    selected_source = source_name
                    break

            if selected_topic is None:
                print("❌ No GPS topics found")
                return None

            print(
                f"✅ Auto-selected GPS source: {selected_source} ({selected_topic})")
        else:
            if gps_source in gps_sources:
                selected_topic = gps_sources[gps_source]
                selected_source = gps_source
            else:
                selected_topic = gps_source  # Custom topic name
                selected_source = 'custom'

            if selected_topic not in available_topics:
                print(f"❌ GPS topic not found: {selected_topic}")
                print("Available GPS-related topics:")
                for topic in available_topics:
                    if any(keyword in topic.lower() for keyword in ['gps', 'fix', 'nav', 'global']):
                        print(f"   {topic}")
                return None

        print(f"🎯 Using GPS topic: {selected_topic}")

        gps_data = []
        message_count = 0

        # Determine message type based on topic
        use_path_msgs = 'path' in selected_topic.lower()

        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()

            if topic == selected_topic:
                try:
                    if use_path_msgs:
                        # Extract from Path message (filtered GPS paths)
                        msg = deserialize_message(data, Path)

                        if len(msg.poses) > 0:
                            # Use the latest pose in the path
                            latest_pose = msg.poses[-1]

                            # Convert local coordinates to GPS-like format
                            # Note: This assumes the path is in local coordinates
                            # For proper GPS coordinates, we'd need the reference point
                            gps_data.append({
                                'timestamp': timestamp * 1e-9,
                                'latitude': latest_pose.pose.position.y,   # Approximation
                                'longitude': latest_pose.pose.position.x,  # Approximation
                                'altitude': latest_pose.pose.position.z,
                                'x_local': latest_pose.pose.position.x,
                                'y_local': latest_pose.pose.position.y,
                                'z_local': latest_pose.pose.position.z,
                                'source': selected_source
                            })
                    else:
                        # Extract from NavSatFix message (raw GPS)
                        msg = deserialize_message(data, NavSatFix)

                        if msg.status.status >= 0:  # Valid GPS fix
                            gps_data.append({
                                'timestamp': timestamp * 1e-9,
                                'latitude': msg.latitude,
                                'longitude': msg.longitude,
                                'altitude': msg.altitude,
                                'status': msg.status.status,
                                'service': msg.status.service,
                                'source': selected_source
                            })

                    message_count += 1

                    if message_count % 50 == 0:
                        print(f"   Processed {message_count} GPS messages...")

                except Exception as e:
                    print(f"⚠️  Error processing GPS message: {e}")
                    continue

        # Reader automatically closes when going out of scope

        if len(gps_data) == 0:
            print("❌ No valid GPS data found")
            return None

        # Create DataFrame and save to CSV
        df = pd.DataFrame(gps_data)
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Calculate trajectory statistics
        if use_path_msgs and 'x_local' in df.columns:
            # Calculate distance using local coordinates
            if len(df) > 1:
                distances = np.sqrt(
                    np.diff(df['x_local'])**2 +
                    np.diff(df['y_local'])**2 +
                    np.diff(df['z_local'])**2
                )
                total_distance = np.sum(distances)
            else:
                total_distance = 0.0
        else:
            total_distance = 0.0  # Cannot calculate distance from lat/lon easily

        duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
        avg_speed = total_distance / duration if duration > 0 and total_distance > 0 else 0

        print(f"✅ GPS extraction completed:")
        print(f"   📊 Messages: {len(df):,}")
        print(f"   ⏱️  Duration: {duration:.1f} seconds")
        print(f"   🎯 Source: {selected_source} ({selected_topic})")
        if total_distance > 0:
            print(f"   📏 Distance: {total_distance:.1f} meters")
            print(f"   🚀 Avg Speed: {avg_speed:.2f} m/s")

        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"✅ GPS data saved to: {output_file}")

        return df

    except ImportError:
        print("❌ rosbag2_py not available. Please install ros2 rosbag2 packages.")
        return None
    except Exception as e:
        print(f"❌ Error extracting GPS data: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Extract VINS and GPS data from ROS2 bags')
    parser.add_argument('bag_path', help='Path to ROS2 bag file (.db3)')
    parser.add_argument('--output-dir', default='.',
                        help='Output directory for CSV files')
    parser.add_argument('--vins-only', action='store_true',
                        help='Extract only VINS data')
    parser.add_argument('--gps-only', action='store_true',
                        help='Extract only GPS data')
    parser.add_argument('--gps-source', default='auto',
                        choices=['auto', 'ppk', 'filtered', 'global', 'raw'],
                        help='GPS data source (auto=best available, ppk=PPK corrected, filtered=filtered GPS, global=fused estimate, raw=raw GPS)')

    args = parser.parse_args()

    if not os.path.exists(args.bag_path):
        print(f"❌ Bag file not found: {args.bag_path}")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate output filenames
    bag_name = os.path.splitext(os.path.basename(args.bag_path))[0]
    vins_output = os.path.join(
        args.output_dir, f'{bag_name}_vins_odometry.csv')
    gps_output = os.path.join(args.output_dir, f'{bag_name}_gps_data.csv')

    print(f"🎯 Enhanced VINS-GPS Data Extractor")
    print(f"📦 Processing: {args.bag_path}")
    print(f"📁 Output directory: {args.output_dir}")
    print("=" * 50)

    results = {}

    # Extract VINS data
    if not args.gps_only:
        print("\n🧭 Extracting VINS odometry data...")
        vins_df = extract_vins_odometry_from_bag(args.bag_path, vins_output)
        results['vins'] = vins_df is not None

    # Extract GPS data
    if not args.vins_only:
        print("\n📡 Extracting GPS data...")
        gps_df = extract_filtered_gps_from_bag(
            args.bag_path, gps_output, args.gps_source)
        results['gps'] = gps_df is not None

    # Summary
    print("\n" + "=" * 50)
    print("📋 Extraction Summary:")
    if 'vins' in results:
        status = "✅ SUCCESS" if results['vins'] else "❌ FAILED"
        print(f"   🧭 VINS: {status}")
        if results['vins']:
            print(f"      📄 File: {vins_output}")
    if 'gps' in results:
        status = "✅ SUCCESS" if results['gps'] else "❌ FAILED"
        print(f"   📡 GPS: {status}")
        if results['gps']:
            print(f"      📄 File: {gps_output}")
            print(f"      🎯 Source: {args.gps_source}")

    if all(results.values()):
        print("\n🎉 Data extraction completed successfully!")
        print("💡 Use offline_vins_gps_analysis.py to analyze the extracted data")
    else:
        print("\n⚠️  Some extractions failed. Check the error messages above.")


if __name__ == '__main__':
    main()
