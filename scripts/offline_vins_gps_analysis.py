#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import os
from scipy import interpolate
import math


def gps_to_local(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """Convert GPS coordinates to local ENU coordinates"""
    R_earth = 6378137.0  # Earth radius in meters

    dlat = lat - ref_lat
    dlon = lon - ref_lon
    dalt = alt - ref_alt

    # Convert to meters (ENU coordinates)
    x = dlon * R_earth * \
        math.cos(math.radians(ref_lat)) * math.pi / 180.0  # East
    y = dlat * R_earth * math.pi / 180.0  # North
    z = dalt  # Up

    return x, y, z


def load_vins_data(file_path):
    """Load VINS trajectory data from CSV"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded VINS data: {len(df)} points from {file_path}")
        return df
    except Exception as e:
        print(f"❌ Error loading VINS data: {e}")
        return None


def extract_gps_from_bag(bag_path, output_file):
    """Extract GPS data from ROS bag to CSV"""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import NavSatFix

        print(f"📦 Extracting GPS data from {bag_path}...")

        storage_options = rosbag2_py.StorageOptions(
            uri=bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        gps_data = []

        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            if topic == '/fix':
                msg = deserialize_message(data, NavSatFix)
                if msg.status.status >= 0:  # Valid GPS fix
                    gps_data.append({
                        'timestamp': timestamp * 1e-9,  # Convert to seconds
                        'latitude': msg.latitude,
                        'longitude': msg.longitude,
                        'altitude': msg.altitude
                    })

        # Save to CSV
        df = pd.DataFrame(gps_data)
        df.to_csv(output_file, index=False)
        print(f"✅ GPS data saved: {len(df)} points to {output_file}")
        return df

    except ImportError:
        print("❌ rosbag2_py not available. Please provide GPS data as CSV file.")
        return None
    except Exception as e:
        print(f"❌ Error extracting GPS data: {e}")
        return None


def load_or_extract_gps(gps_source):
    """Load GPS data from CSV or extract from bag"""
    if gps_source.endswith('.csv'):
        try:
            df = pd.read_csv(gps_source)
            print(f"✅ Loaded GPS data: {len(df)} points from {gps_source}")
            return df
        except Exception as e:
            print(f"❌ Error loading GPS CSV: {e}")
            return None
    else:
        # Assume it's a bag file
        gps_csv = gps_source.replace('.db3', '_gps.csv')
        return extract_gps_from_bag(gps_source, gps_csv)


def align_trajectories_method(vins_positions, gps_positions):
    """Align VINS trajectory with GPS using optimal rotation and translation"""
    # Convert to numpy arrays for easier manipulation
    vins_pos = np.array(vins_positions)
    gps_pos = np.array(gps_positions)

    # Only align X-Y coordinates, preserve Z as-is
    vins_xy = vins_pos[:, :2]  # Only X, Y
    gps_xy = gps_pos[:, :2]    # Only X, Y

    # Center both trajectories at origin (X-Y only)
    vins_xy_centered = vins_xy - np.mean(vins_xy, axis=0)
    gps_xy_centered = gps_xy - np.mean(gps_xy, axis=0)

    # Try different rotations to find best alignment
    best_error = float('inf')
    best_rotation = 0

    print("🔍 Finding optimal trajectory alignment (X-Y only)...")

    # Test rotations from 0 to 360 degrees in 1-degree steps
    for angle_deg in range(0, 360, 1):
        angle_rad = np.radians(angle_deg)

        # 2D rotation matrix (only for X-Y plane)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix_2d = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])

        # Apply rotation to VINS X-Y trajectory only
        vins_xy_rotated = vins_xy_centered @ rotation_matrix_2d.T

        # Calculate alignment error (sum of squared distances in X-Y only)
        error = np.sum((vins_xy_rotated - gps_xy_centered)**2)

        if error < best_error:
            best_error = error
            best_rotation = angle_deg

    print(f"✅ Best alignment: {best_rotation}° rotation (X-Y plane only)")

    # Apply best rotation to X-Y coordinates only
    best_angle_rad = np.radians(best_rotation)
    cos_a, sin_a = np.cos(best_angle_rad), np.sin(best_angle_rad)
    best_rotation_matrix_2d = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ])

    # Apply rotation and re-center for X-Y
    vins_xy_aligned = (vins_xy - np.mean(vins_xy, axis=0)
                       ) @ best_rotation_matrix_2d.T
    vins_xy_aligned += np.mean(gps_xy, axis=0)

    # Combine aligned X-Y with original Z coordinates
    vins_aligned = np.column_stack([
        vins_xy_aligned[:, 0],  # Aligned X
        vins_xy_aligned[:, 1],  # Aligned Y
        vins_pos[:, 2]          # Original Z (unchanged)
    ])

    return vins_aligned, best_rotation


def synchronize_trajectories(vins_df, gps_df, max_time_diff=0.5, align_trajectories=True):
    """Synchronize VINS and GPS trajectories by timestamp"""
    print(f"🔄 Synchronizing trajectories (max time diff: {max_time_diff}s)...")

    # Convert GPS to local coordinates using first GPS point as reference
    ref_lat = gps_df.iloc[0]['latitude']
    ref_lon = gps_df.iloc[0]['longitude']
    ref_alt = gps_df.iloc[0]['altitude']

    print(
        f"📍 GPS reference: lat={ref_lat:.6f}, lon={ref_lon:.6f}, alt={ref_alt:.2f}")

    # Convert all GPS points to local coordinates
    gps_local = []
    for _, row in gps_df.iterrows():
        x, y, z = gps_to_local(row['latitude'], row['longitude'], row['altitude'],
                               ref_lat, ref_lon, ref_alt)
        gps_local.append({
            'timestamp': row['timestamp'],
            'x': x, 'y': y, 'z': z
        })

    gps_local_df = pd.DataFrame(gps_local)

    # Synchronize by finding closest timestamps
    synchronized = []
    for _, vins_row in vins_df.iterrows():
        vins_time = vins_row['timestamp']

        # Find closest GPS point
        time_diffs = abs(gps_local_df['timestamp'] - vins_time)
        min_idx = time_diffs.idxmin()
        min_time_diff = time_diffs.iloc[min_idx]

        if min_time_diff <= max_time_diff:
            gps_row = gps_local_df.iloc[min_idx]
            synchronized.append({
                'timestamp': vins_time,
                'vins_x': vins_row['x'],
                'vins_y': vins_row['y'],
                'vins_z': vins_row['z'],
                'gps_x': gps_row['x'],
                'gps_y': gps_row['y'],
                'gps_z': gps_row['z'],
                'time_diff': min_time_diff
            })

    sync_df = pd.DataFrame(synchronized)

    if len(sync_df) == 0:
        print("❌ No synchronized points found")
        return sync_df

    print(f"✅ Synchronized {len(sync_df)} point pairs")

    # Align trajectories to fix orientation mismatch
    if align_trajectories and len(sync_df) > 0:
        vins_positions = sync_df[['vins_x', 'vins_y', 'vins_z']].values
        gps_positions = sync_df[['gps_x', 'gps_y', 'gps_z']].values

        aligned_positions, rotation_angle = align_trajectories_method(
            vins_positions, gps_positions)

        # Update VINS positions with aligned values
        sync_df['vins_x'] = aligned_positions[:, 0]
        sync_df['vins_y'] = aligned_positions[:, 1]
        sync_df['vins_z'] = aligned_positions[:, 2]

        print(f"🎯 Trajectories aligned with {rotation_angle}° rotation")
    else:
        print("⚠️  Trajectory alignment skipped")

    return sync_df


def calculate_trajectory_errors(sync_df):
    """Calculate comprehensive error metrics"""
    print("📊 Calculating error metrics...")

    # Position errors
    sync_df['error_x'] = sync_df['vins_x'] - sync_df['gps_x']
    sync_df['error_y'] = sync_df['vins_y'] - sync_df['gps_y']
    sync_df['error_z'] = sync_df['vins_z'] - sync_df['gps_z']

    # 2D and 3D errors
    sync_df['error_2d'] = np.sqrt(
        sync_df['error_x']**2 + sync_df['error_y']**2)
    sync_df['error_3d'] = np.sqrt(
        sync_df['error_x']**2 + sync_df['error_y']**2 + sync_df['error_z']**2)

    # Calculate metrics
    metrics = {
        'n_points': len(sync_df),
        'duration': sync_df['timestamp'].max() - sync_df['timestamp'].min(),

        # Mean errors
        'mean_error_x': sync_df['error_x'].mean(),
        'mean_error_y': sync_df['error_y'].mean(),
        'mean_error_z': sync_df['error_z'].mean(),
        'mean_error_2d': sync_df['error_2d'].mean(),
        'mean_error_3d': sync_df['error_3d'].mean(),

        # Standard deviations
        'std_error_x': sync_df['error_x'].std(),
        'std_error_y': sync_df['error_y'].std(),
        'std_error_z': sync_df['error_z'].std(),
        'std_error_2d': sync_df['error_2d'].std(),
        'std_error_3d': sync_df['error_3d'].std(),

        # RMSE
        'rmse_x': np.sqrt((sync_df['error_x']**2).mean()),
        'rmse_y': np.sqrt((sync_df['error_y']**2).mean()),
        'rmse_z': np.sqrt((sync_df['error_z']**2).mean()),
        'rmse_2d': np.sqrt((sync_df['error_2d']**2).mean()),
        'rmse_3d': np.sqrt((sync_df['error_3d']**2).mean()),

        # Maximum errors
        'max_error_2d': sync_df['error_2d'].max(),
        'max_error_3d': sync_df['error_3d'].max(),

        # Percentiles
        'p50_error_2d': sync_df['error_2d'].quantile(0.5),
        'p95_error_2d': sync_df['error_2d'].quantile(0.95),
        'p99_error_2d': sync_df['error_2d'].quantile(0.99),
    }

    return sync_df, metrics


def create_comprehensive_plots(sync_df, metrics, output_dir):
    """Create comprehensive visualization plots"""
    print("🎨 Creating visualization plots...")

    plt.style.use('default')
    fig = plt.figure(figsize=(24, 20))  # Even larger figure size

    # 1. Trajectory comparison (top-down view)
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(sync_df['vins_x'], sync_df['vins_y'],
             'b-', linewidth=2, label='VINS', alpha=0.8)
    ax1.plot(sync_df['gps_x'], sync_df['gps_y'], 'r-',
             linewidth=2, label='GPS', alpha=0.8)
    ax1.plot(sync_df['vins_x'].iloc[0], sync_df['vins_y'].iloc[0],
             'go', markersize=8, label='Start')
    ax1.plot(sync_df['vins_x'].iloc[-1],
             sync_df['vins_y'].iloc[-1], 'ro', markersize=8, label='End')
    ax1.set_xlabel('X Position (m)', fontsize=10)
    ax1.set_ylabel('Y Position (m)', fontsize=10)
    ax1.set_title('Flight Trajectory Comparison', fontsize=11, pad=5)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.tick_params(labelsize=9)

    # 2. 2D Error over time
    ax2 = plt.subplot(3, 3, 2)
    time_rel = sync_df['timestamp'] - sync_df['timestamp'].iloc[0]
    ax2.plot(time_rel, sync_df['error_2d'], 'r-', linewidth=1.5)
    ax2.axhline(y=metrics['mean_error_2d'], color='g', linestyle='--',
                label=f'Mean: {metrics["mean_error_2d"]:.2f}m')
    ax2.axhline(y=metrics['rmse_2d'], color='orange',
                linestyle='--', label=f'RMSE: {metrics["rmse_2d"]:.2f}m')
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.set_ylabel('2D Error (m)', fontsize=10)
    ax2.set_title('2D Position Error Over Time', fontsize=11, pad=5)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=9)

    # 3. Error components
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(time_rel, sync_df['error_x'], 'r-', label='X Error', alpha=0.7)
    ax3.plot(time_rel, sync_df['error_y'], 'g-', label='Y Error', alpha=0.7)
    ax3.plot(time_rel, sync_df['error_z'], 'b-', label='Z Error', alpha=0.7)
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('Error (m)', fontsize=10)
    ax3.set_title('Position Error Components', fontsize=11, pad=5)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=9)

    # 4. Error distribution histogram
    ax4 = plt.subplot(3, 3, 4)
    ax4.hist(sync_df['error_2d'], bins=30, alpha=0.7,
             color='red', edgecolor='black')
    ax4.axvline(x=metrics['mean_error_2d'], color='green',
                linestyle='--', label=f'Mean: {metrics["mean_error_2d"]:.2f}m')
    ax4.axvline(x=metrics['p95_error_2d'], color='orange',
                linestyle='--', label=f'95th %: {metrics["p95_error_2d"]:.2f}m')
    ax4.set_xlabel('2D Error (m)', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('2D Error Distribution', fontsize=11, pad=20)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=9)

    # 5. Cumulative error distribution
    ax5 = plt.subplot(3, 3, 5)
    sorted_errors = np.sort(sync_df['error_2d'])
    percentiles = np.arange(1, len(sorted_errors) + 1) / \
        len(sorted_errors) * 100
    ax5.plot(sorted_errors, percentiles, 'b-', linewidth=2)
    ax5.axvline(x=metrics['p50_error_2d'], color='green',
                linestyle='--', label=f'50th %: {metrics["p50_error_2d"]:.2f}m')
    ax5.axvline(x=metrics['p95_error_2d'], color='orange',
                linestyle='--', label=f'95th %: {metrics["p95_error_2d"]:.2f}m')
    ax5.set_xlabel('2D Error (m)', fontsize=10)
    ax5.set_ylabel('Cumulative %', fontsize=10)
    ax5.set_title('Cumulative Error Distribution', fontsize=11, pad=20)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(labelsize=9)

    # 6. X-Y Error scatter
    ax6 = plt.subplot(3, 3, 6)
    scatter = ax6.scatter(
        sync_df['error_x'], sync_df['error_y'], c=time_rel, cmap='viridis', alpha=0.6)
    ax6.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax6.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax6.set_xlabel('X Error (m)', fontsize=10)
    ax6.set_ylabel('Y Error (m)', fontsize=10)
    ax6.set_title('X-Y Error Scatter (by time)', fontsize=11, pad=20)
    ax6.axis('equal')
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(labelsize=9)
    plt.colorbar(scatter, ax=ax6, label='Time (s)', shrink=0.8)

    # 7. 3D Error over time
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(time_rel, sync_df['error_3d'], 'purple', linewidth=1.5)
    ax7.axhline(y=metrics['mean_error_3d'], color='g', linestyle='--',
                label=f'Mean: {metrics["mean_error_3d"]:.2f}m')
    ax7.axhline(y=metrics['rmse_3d'], color='orange',
                linestyle='--', label=f'RMSE: {metrics["rmse_3d"]:.2f}m')
    ax7.set_xlabel('Time (s)', fontsize=10)
    ax7.set_ylabel('3D Error (m)', fontsize=10)
    ax7.set_title('3D Position Error Over Time', fontsize=11, pad=20)
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    ax7.tick_params(labelsize=9)

    # 8. Altitude comparison
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(time_rel, sync_df['vins_z'], 'b-', label='VINS Z', linewidth=2)
    ax8.plot(time_rel, sync_df['gps_z'], 'r-', label='GPS Z', linewidth=2)
    ax8.set_xlabel('Time (s)', fontsize=10)
    ax8.set_ylabel('Altitude (m)', fontsize=10)
    ax8.set_title('Altitude Comparison', fontsize=11, pad=20)
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)
    ax8.tick_params(labelsize=9)

    # 9. Statistics text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    stats_text = f"""
VINS-GPS Error Analysis Summary
{'='*35}

Dataset Information:
• Points analyzed: {metrics['n_points']:,}
• Duration: {metrics['duration']:.1f} seconds
• Avg frequency: {metrics['n_points']/metrics['duration']:.1f} Hz

Position Errors (meters):
• Mean 2D error: {metrics['mean_error_2d']:.3f} ± {metrics['std_error_2d']:.3f}
• Mean 3D error: {metrics['mean_error_3d']:.3f} ± {metrics['std_error_3d']:.3f}

Root Mean Square Errors:
• RMSE 2D: {metrics['rmse_2d']:.3f} m
• RMSE 3D: {metrics['rmse_3d']:.3f} m
• RMSE X: {metrics['rmse_x']:.3f} m
• RMSE Y: {metrics['rmse_y']:.3f} m
• RMSE Z: {metrics['rmse_z']:.3f} m

Error Distribution:
• 50th percentile: {metrics['p50_error_2d']:.3f} m
• 95th percentile: {metrics['p95_error_2d']:.3f} m
• 99th percentile: {metrics['p99_error_2d']:.3f} m
• Maximum 2D: {metrics['max_error_2d']:.3f} m
• Maximum 3D: {metrics['max_error_3d']:.3f} m
    """.strip()

    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    # Use only tight_layout with generous padding

    # Save the plot
    plot_file = os.path.join(output_dir, 'vins_gps_analysis_comprehensive.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', pad_inches=0.3)
    print(f"✅ Comprehensive plot saved: {plot_file}")

    plt.tight_layout(pad=10, w_pad=2.5, h_pad=5)
    plt.show()


def save_results(sync_df, metrics, output_dir):
    """Save analysis results to files"""
    print(f"💾 Saving results to {output_dir}...")

    os.makedirs(output_dir, exist_ok=True)

    # Save synchronized data
    sync_file = os.path.join(output_dir, 'synchronized_trajectories.csv')
    sync_df.to_csv(sync_file, index=False)
    print(f"✅ Synchronized data saved: {sync_file}")

    # Save metrics
    metrics_file = os.path.join(output_dir, 'error_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("VINS-GPS Trajectory Error Analysis\n")
        f.write("="*40 + "\n\n")

        f.write(f"Dataset Information:\n")
        f.write(f"  Points analyzed: {metrics['n_points']:,}\n")
        f.write(f"  Duration: {metrics['duration']:.1f} seconds\n")
        f.write(
            f"  Average frequency: {metrics['n_points']/metrics['duration']:.1f} Hz\n\n")

        f.write(f"Position Errors (meters):\n")
        f.write(
            f"  Mean X error: {metrics['mean_error_x']:.6f} ± {metrics['std_error_x']:.6f}\n")
        f.write(
            f"  Mean Y error: {metrics['mean_error_y']:.6f} ± {metrics['std_error_y']:.6f}\n")
        f.write(
            f"  Mean Z error: {metrics['mean_error_z']:.6f} ± {metrics['std_error_z']:.6f}\n")
        f.write(
            f"  Mean 2D error: {metrics['mean_error_2d']:.6f} ± {metrics['std_error_2d']:.6f}\n")
        f.write(
            f"  Mean 3D error: {metrics['mean_error_3d']:.6f} ± {metrics['std_error_3d']:.6f}\n\n")

        f.write(f"Root Mean Square Errors (RMSE):\n")
        f.write(f"  RMSE X: {metrics['rmse_x']:.6f} m\n")
        f.write(f"  RMSE Y: {metrics['rmse_y']:.6f} m\n")
        f.write(f"  RMSE Z: {metrics['rmse_z']:.6f} m\n")
        f.write(f"  RMSE 2D: {metrics['rmse_2d']:.6f} m\n")
        f.write(f"  RMSE 3D: {metrics['rmse_3d']:.6f} m\n\n")

        f.write(f"Error Distribution:\n")
        f.write(
            f"  50th percentile (median): {metrics['p50_error_2d']:.6f} m\n")
        f.write(f"  95th percentile: {metrics['p95_error_2d']:.6f} m\n")
        f.write(f"  99th percentile: {metrics['p99_error_2d']:.6f} m\n")
        f.write(f"  Maximum 2D error: {metrics['max_error_2d']:.6f} m\n")
        f.write(f"  Maximum 3D error: {metrics['max_error_3d']:.6f} m\n")

    print(f"✅ Metrics saved: {metrics_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Offline VINS-GPS trajectory error analysis')
    parser.add_argument('--vins', required=True,
                        help='VINS trajectory CSV file or pattern')
    parser.add_argument('--gps', required=True,
                        help='GPS CSV file or ROS bag file')
    parser.add_argument(
        '--output', default='./analysis_output', help='Output directory')
    parser.add_argument('--max-time-diff', type=float, default=0.5,
                        help='Maximum time difference for synchronization (seconds)')
    parser.add_argument('--no-alignment', action='store_true',
                        help='Disable automatic trajectory alignment')

    args = parser.parse_args()

    print("🚁 VINS-GPS Trajectory Error Analysis")
    print("="*40)

    if args.no_alignment:
        print("⚠️  Trajectory alignment disabled")
    else:
        print("🎯 Automatic trajectory alignment enabled")

    # Load VINS data
    if '*' in args.vins:
        vins_files = glob.glob(args.vins)
        if not vins_files:
            print(f"❌ No VINS files found matching: {args.vins}")
            return
        vins_file = max(vins_files)  # Use most recent
        print(f"📂 Using most recent VINS file: {vins_file}")
    else:
        vins_file = args.vins

    vins_df = load_vins_data(vins_file)
    if vins_df is None:
        return

    # Load GPS data
    gps_df = load_or_extract_gps(args.gps)
    if gps_df is None:
        return

    # Synchronize trajectories
    sync_df = synchronize_trajectories(
        vins_df, gps_df, args.max_time_diff, not args.no_alignment)
    if len(sync_df) == 0:
        print("❌ No synchronized points found. Check time alignment.")
        return

    # Calculate errors
    sync_df, metrics = calculate_trajectory_errors(sync_df)

    # Print summary
    print("\n📊 Analysis Summary:")
    print(f"   Synchronized points: {metrics['n_points']:,}")
    print(f"   RMSE 2D: {metrics['rmse_2d']:.3f} m")
    print(f"   RMSE 3D: {metrics['rmse_3d']:.3f} m")
    print(f"   Max 2D error: {metrics['max_error_2d']:.3f} m")
    print(f"   95th percentile: {metrics['p95_error_2d']:.3f} m")

    # Save results
    save_results(sync_df, metrics, args.output)

    # Create plots
    create_comprehensive_plots(sync_df, metrics, args.output)

    print(f"\n✅ Analysis complete! Results saved to: {args.output}")


if __name__ == '__main__':
    main()
