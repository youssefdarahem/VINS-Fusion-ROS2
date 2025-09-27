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

            # Check if this is enhanced GPS data with local coordinates
            if 'x_local' in df.columns and 'y_local' in df.columns:
                print(f"📍 Detected enhanced GPS data with local coordinates")
                if 'source' in df.columns:
                    print(
                        f"🎯 GPS source: {df['source'].iloc[0] if len(df) > 0 else 'unknown'}")

            return df
        except Exception as e:
            print(f"❌ Error loading GPS CSV: {e}")
            return None
    else:
        # Assume it's a bag file - suggest using enhanced extractor
        print("💡 For better GPS accuracy, consider using:")
        print("   python3 extract_vins_gps_enhanced.py your_bag.db3 --gps-source ppk")
        print("   This will use filtered GPS data instead of raw /fix topic")
        print("")
        print("🔄 Falling back to raw GPS extraction...")
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


def synchronize_trajectories(vins_df, gps_df, max_time_diff=0.5, align_trajectories=True, align_altitude=True):
    """Synchronize VINS and GPS trajectories by timestamp"""
    print(f"🔄 Synchronizing trajectories (max time diff: {max_time_diff}s)...")

    # Check if GPS data already has local coordinates (enhanced extraction)
    if 'x_local' in gps_df.columns and 'y_local' in gps_df.columns:
        print("📍 Using enhanced GPS data with local coordinates")
        gps_local = []
        for _, row in gps_df.iterrows():
            gps_local.append({
                'timestamp': row['timestamp'],
                'x': row['x_local'],
                'y': row['y_local'],
                'z': row['z_local'] if 'z_local' in row else row.get('altitude', 0)
            })
        gps_local_df = pd.DataFrame(gps_local)

    else:
        # Convert GPS to local coordinates using first GPS point as reference
        print("📍 Converting GPS lat/lon to local coordinates")
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

    # Align altitude (Z-axis) using mean offset
    if align_altitude and len(sync_df) > 0:
        mean_vins_z = sync_df['vins_z'].mean()
        mean_gps_z = sync_df['gps_z'].mean()
        altitude_offset = mean_gps_z - mean_vins_z

        # Apply altitude offset to VINS Z coordinates
        sync_df['vins_z'] = sync_df['vins_z'] + altitude_offset

        print(f"📏 Altitude aligned with {altitude_offset:.2f}m offset")
    else:
        print("⚠️  Altitude alignment skipped")

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
    """Create comprehensive visualization plots as separate figures"""
    print("🎨 Creating comprehensive visualization plots...")

    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Main Trajectory Comparison
    plt.style.use('default')
    fig1, ax1 = plt.subplots(figsize=(12, 10))

    ax1.plot(sync_df['vins_x'], sync_df['vins_y'],
             'b-', linewidth=2, label='VINS', alpha=0.8)
    ax1.plot(sync_df['gps_x'], sync_df['gps_y'], 'r-',
             linewidth=2, label='GPS', alpha=0.8)
    ax1.plot(sync_df['vins_x'].iloc[0], sync_df['vins_y'].iloc[0],
             'go', markersize=8, label='Start')
    ax1.plot(sync_df['vins_x'].iloc[-1],
             sync_df['vins_y'].iloc[-1], 'ro', markersize=8, label='End')
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title('Flight Trajectory Comparison', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Add trajectory statistics
    vins_distance = calculate_trajectory_distance(sync_df, 'vins')
    gps_distance = calculate_trajectory_distance(sync_df, 'gps')
    stats_text = f"VINS distance: {vins_distance:.1f} m\nGPS distance: {gps_distance:.1f} m\nMean 2D error: {metrics['mean_error_2d']:.3f} m\nRMSE 2D: {metrics['rmse_2d']:.3f} m"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plot_file1 = os.path.join(output_dir, 'main_trajectory_comparison.png')
    plt.savefig(plot_file1, dpi=300, bbox_inches='tight')
    print(f"✅ Main trajectory plot saved: {plot_file1}")
    plt.close()

    # Figure 2: Error Analysis Over Time
    fig2, ((ax2, ax3), (ax4, ax5)) = plt.subplots(2, 2, figsize=(16, 12))
    time_rel = sync_df['timestamp'] - sync_df['timestamp'].iloc[0]

    # 2D Error over time
    ax2.plot(time_rel, sync_df['error_2d'], 'r-', linewidth=2)
    ax2.axhline(y=metrics['mean_error_2d'], color='g', linestyle='--',
                label=f'Mean: {metrics["mean_error_2d"]:.3f}m')
    ax2.axhline(y=metrics['rmse_2d'], color='orange',
                linestyle='--', label=f'RMSE: {metrics["rmse_2d"]:.3f}m')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('2D Error (m)', fontsize=11)
    ax2.set_title('2D Position Error Over Time', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Error components
    ax3.plot(time_rel, sync_df['error_x'], 'r-',
             label='X Error', alpha=0.8, linewidth=1.5)
    ax3.plot(time_rel, sync_df['error_y'], 'g-',
             label='Y Error', alpha=0.8, linewidth=1.5)
    ax3.plot(time_rel, sync_df['error_z'], 'b-',
             label='Z Error', alpha=0.8, linewidth=1.5)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Error (m)', fontsize=11)
    ax3.set_title('Position Error Components', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 3D Error over time
    ax4.plot(time_rel, sync_df['error_3d'], 'purple', linewidth=2)
    ax4.axhline(y=metrics['mean_error_3d'], color='g', linestyle='--',
                label=f'Mean: {metrics["mean_error_3d"]:.3f}m')
    ax4.axhline(y=metrics['rmse_3d'], color='orange',
                linestyle='--', label=f'RMSE: {metrics["rmse_3d"]:.3f}m')
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('3D Error (m)', fontsize=11)
    ax4.set_title('3D Position Error Over Time', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # X-Y Error scatter
    scatter = ax5.scatter(
        sync_df['error_x'], sync_df['error_y'], c=time_rel, cmap='viridis', alpha=0.6)
    ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax5.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax5.set_xlabel('X Error (m)', fontsize=11)
    ax5.set_ylabel('Y Error (m)', fontsize=11)
    ax5.set_title('X-Y Error Scatter (by time)', fontsize=12)
    ax5.axis('equal')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Time (s)', shrink=0.8)

    plt.tight_layout()
    plot_file2 = os.path.join(output_dir, 'error_analysis_over_time.png')
    plt.savefig(plot_file2, dpi=300, bbox_inches='tight')
    print(f"✅ Error analysis plot saved: {plot_file2}")
    plt.close()

    # Figure 3: Error Distributions
    fig3, ((ax6, ax7), (ax8, ax9)) = plt.subplots(2, 2, figsize=(16, 12))

    # Error distribution histogram
    ax6.hist(sync_df['error_2d'], bins=30, alpha=0.7,
             color='red', edgecolor='black')
    ax6.axvline(x=metrics['mean_error_2d'], color='green',
                linestyle='--', label=f'Mean: {metrics["mean_error_2d"]:.3f}m')
    ax6.axvline(x=metrics['p95_error_2d'], color='orange',
                linestyle='--', label=f'95th %: {metrics["p95_error_2d"]:.3f}m')
    ax6.set_xlabel('2D Error (m)', fontsize=11)
    ax6.set_ylabel('Frequency', fontsize=11)
    ax6.set_title('2D Error Distribution', fontsize=12)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    # Cumulative error distribution
    sorted_errors = np.sort(sync_df['error_2d'])
    percentiles = np.arange(1, len(sorted_errors) + 1) / \
        len(sorted_errors) * 100
    ax7.plot(sorted_errors, percentiles, 'b-', linewidth=2)
    ax7.axvline(x=metrics['p50_error_2d'], color='green',
                linestyle='--', label=f'50th %: {metrics["p50_error_2d"]:.3f}m')
    ax7.axvline(x=metrics['p95_error_2d'], color='orange',
                linestyle='--', label=f'95th %: {metrics["p95_error_2d"]:.3f}m')
    ax7.set_xlabel('2D Error (m)', fontsize=11)
    ax7.set_ylabel('Cumulative %', fontsize=11)
    ax7.set_title('Cumulative Error Distribution', fontsize=12)
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)

    # Error component distributions
    bins = np.linspace(
        0, max(sync_df['error_3d'].max(), sync_df['error_2d'].max()), 30)
    ax8.hist(sync_df['error_x'], bins=bins,
             alpha=0.6, label='X Error', color='red')
    ax8.hist(sync_df['error_y'], bins=bins, alpha=0.6,
             label='Y Error', color='green')
    ax8.hist(sync_df['error_z'], bins=bins,
             alpha=0.6, label='Z Error', color='blue')
    ax8.set_xlabel('Position Error (m)', fontsize=11)
    ax8.set_ylabel('Frequency', fontsize=11)
    ax8.set_title('Error Component Distributions', fontsize=12)
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)

    # Error statistics bar chart
    components = ['X', 'Y', 'Z', '2D', '3D']
    rmse_values = [metrics['rmse_x'], metrics['rmse_y'], metrics['rmse_z'],
                   metrics['rmse_2d'], metrics['rmse_3d']]
    mean_values = [abs(metrics['mean_error_x']), abs(metrics['mean_error_y']),
                   abs(metrics['mean_error_z']), metrics['mean_error_2d'], metrics['mean_error_3d']]

    x_pos = np.arange(len(components))
    width = 0.35

    ax9.bar(x_pos - width/2, rmse_values, width,
            label='RMSE', alpha=0.8, color='orange')
    ax9.bar(x_pos + width/2, mean_values, width,
            label='Mean', alpha=0.8, color='skyblue')
    ax9.set_xlabel('Error Component', fontsize=11)
    ax9.set_ylabel('Error (m)', fontsize=11)
    ax9.set_title('Error Statistics Summary', fontsize=12)
    ax9.set_xticks(x_pos)
    ax9.set_xticklabels(components)
    ax9.legend(fontsize=10)
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file3 = os.path.join(output_dir, 'error_distributions.png')
    plt.savefig(plot_file3, dpi=300, bbox_inches='tight')
    print(f"✅ Error distributions plot saved: {plot_file3}")
    plt.close()

    # Figure 4: Altitude Analysis
    fig4, ax10 = plt.subplots(figsize=(14, 8))
    ax10.plot(time_rel, sync_df['vins_z'], 'b-',
              label='VINS Altitude', linewidth=2)
    ax10.plot(time_rel, sync_df['gps_z'], 'r-',
              label='GPS Altitude', linewidth=2)
    ax10.set_xlabel('Time (s)', fontsize=12)
    ax10.set_ylabel('Altitude (m)', fontsize=12)
    ax10.set_title('Altitude Comparison Over Time', fontsize=14)
    ax10.legend(fontsize=11)
    ax10.grid(True, alpha=0.3)

    # Add altitude statistics
    alt_stats = f"VINS range: {sync_df['vins_z'].max() - sync_df['vins_z'].min():.1f} m\nGPS range: {sync_df['gps_z'].max() - sync_df['gps_z'].min():.1f} m\nMean Z error: {metrics['mean_error_z']:.3f} m\nRMSE Z: {metrics['rmse_z']:.3f} m"
    ax10.text(0.02, 0.98, alt_stats, transform=ax10.transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plot_file4 = os.path.join(output_dir, 'altitude_comparison.png')
    plt.savefig(plot_file4, dpi=300, bbox_inches='tight')
    print(f"✅ Altitude comparison plot saved: {plot_file4}")
    plt.close()

    # Figure 5: Statistics Summary
    fig5, ax11 = plt.subplots(figsize=(12, 10))
    ax11.axis('off')

    stats_text = f"""
VINS-GPS Error Analysis Summary
{'='*50}

Dataset Information:
• Points analyzed: {metrics['n_points']:,}
• Duration: {metrics['duration']:.1f} seconds ({metrics['duration']/60:.1f} minutes)
• Average frequency: {metrics['n_points']/metrics['duration']:.1f} Hz

Trajectory Distances:
• VINS total distance: {vins_distance:.2f} m
• GPS total distance: {gps_distance:.2f} m
• Distance difference: {abs(vins_distance - gps_distance):.2f} m

Position Errors (meters):
• Mean 2D error: {metrics['mean_error_2d']:.6f} ± {metrics['std_error_2d']:.6f}
• Mean 3D error: {metrics['mean_error_3d']:.6f} ± {metrics['std_error_3d']:.6f}
• Mean X error: {metrics['mean_error_x']:+.6f} ± {metrics['std_error_x']:.6f}
• Mean Y error: {metrics['mean_error_y']:+.6f} ± {metrics['std_error_y']:.6f}
• Mean Z error: {metrics['mean_error_z']:+.6f} ± {metrics['std_error_z']:.6f}

Root Mean Square Errors (RMSE):
• RMSE 2D: {metrics['rmse_2d']:.6f} m
• RMSE 3D: {metrics['rmse_3d']:.6f} m
• RMSE X: {metrics['rmse_x']:.6f} m
• RMSE Y: {metrics['rmse_y']:.6f} m
• RMSE Z: {metrics['rmse_z']:.6f} m

Error Distribution:
• 50th percentile (median): {metrics['p50_error_2d']:.6f} m
• 95th percentile: {metrics['p95_error_2d']:.6f} m
• 99th percentile: {metrics['p99_error_2d']:.6f} m
• Maximum 2D error: {metrics['max_error_2d']:.6f} m
• Maximum 3D error: {metrics['max_error_3d']:.6f} m

Performance Assessment:
• Points with <1m error: {sum(sync_df['error_2d'] < 1.0):,} ({sum(sync_df['error_2d'] < 1.0)/len(sync_df)*100:.1f}%)
• Points with <2m error: {sum(sync_df['error_2d'] < 2.0):,} ({sum(sync_df['error_2d'] < 2.0)/len(sync_df)*100:.1f}%)
• Points with >5m error: {sum(sync_df['error_2d'] > 5.0):,} ({sum(sync_df['error_2d'] > 5.0)/len(sync_df)*100:.1f}%)
    """.strip()

    ax11.text(0.05, 0.95, stats_text, transform=ax11.transAxes, fontsize=11,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plot_file5 = os.path.join(output_dir, 'statistics_summary.png')
    plt.savefig(plot_file5, dpi=300, bbox_inches='tight')
    print(f"✅ Statistics summary plot saved: {plot_file5}")
    plt.close()

    print(f"📁 All 5 comprehensive plots saved to: {output_dir}")
    print(f"   1. main_trajectory_comparison.png")
    print(f"   2. error_analysis_over_time.png")
    print(f"   3. error_distributions.png")
    print(f"   4. altitude_comparison.png")
    print(f"   5. statistics_summary.png")


def create_trajectory_figure(sync_df, metrics, output_dir):
    """Create focused trajectory comparison figure"""
    print("🎨 Creating trajectory comparison figure...")

    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # X-Y trajectory plot
    ax1.plot(sync_df['vins_x'], sync_df['vins_y'],
             'b-', linewidth=2, label='VINS', alpha=0.8)
    ax1.plot(sync_df['gps_x'], sync_df['gps_y'], 'r-',
             linewidth=2, label='GPS', alpha=0.8)
    ax1.plot(sync_df['vins_x'].iloc[0], sync_df['vins_y'].iloc[0],
             'go', markersize=10, label='Start')
    ax1.plot(sync_df['vins_x'].iloc[-1],
             sync_df['vins_y'].iloc[-1], 'ro', markersize=10, label='End')
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title('Aligned Trajectory Comparison (Top View)', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Add trajectory statistics as text box
    stats_text = f"Trajectory Statistics:\nTotal distance VINS: {calculate_trajectory_distance(sync_df, 'vins'):.1f} m\nTotal distance GPS: {calculate_trajectory_distance(sync_df, 'gps'):.1f} m\nMean 2D error: {metrics['mean_error_2d']:.3f} m\nRMSE 2D: {metrics['rmse_2d']:.3f} m"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # 3D trajectory plot (side view)
    time_rel = sync_df['timestamp'] - sync_df['timestamp'].iloc[0]
    ax2.plot(time_rel, sync_df['vins_z'], 'b-',
             linewidth=2, label='VINS Altitude', alpha=0.8)
    ax2.plot(time_rel, sync_df['gps_z'], 'r-',
             linewidth=2, label='GPS Altitude', alpha=0.8)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Altitude (m)', fontsize=12)
    ax2.set_title('Altitude Comparison Over Time', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Add altitude statistics
    alt_stats = f"Altitude Statistics:\nVINS range: {sync_df['vins_z'].max() - sync_df['vins_z'].min():.1f} m\nGPS range: {sync_df['gps_z'].max() - sync_df['gps_z'].min():.1f} m\nMean Z error: {metrics['mean_error_z']:.3f} m\nRMSE Z: {metrics['rmse_z']:.3f} m"
    ax2.text(0.02, 0.98, alt_stats, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(output_dir, 'trajectory_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Trajectory comparison plot saved: {plot_file}")
    plt.close()


def create_altitude_figure(sync_df, metrics, output_dir):
    """Create focused altitude analysis figure"""
    print("🎨 Creating altitude analysis figure...")

    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    time_rel = sync_df['timestamp'] - sync_df['timestamp'].iloc[0]

    # Altitude comparison over time
    ax1.plot(time_rel, sync_df['vins_z'], 'b-',
             linewidth=2, label='VINS', alpha=0.8)
    ax1.plot(time_rel, sync_df['gps_z'], 'r-',
             linewidth=2, label='GPS', alpha=0.8)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Altitude (m)', fontsize=11)
    ax1.set_title('Altitude Comparison Over Time', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Altitude error over time
    ax2.plot(time_rel, sync_df['error_z'], 'purple', linewidth=2)
    ax2.axhline(y=metrics['mean_error_z'], color='g', linestyle='--',
                label=f'Mean: {metrics["mean_error_z"]:.3f}m')
    ax2.axhline(y=metrics['rmse_z'], color='orange', linestyle='--',
                label=f'RMSE: {metrics["rmse_z"]:.3f}m')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Z Error (m)', fontsize=11)
    ax2.set_title('Altitude Error Over Time', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Altitude correlation scatter
    ax3.scatter(sync_df['gps_z'], sync_df['vins_z'],
                alpha=0.6, c=time_rel, cmap='viridis')
    min_z = min(sync_df['gps_z'].min(), sync_df['vins_z'].min())
    max_z = max(sync_df['gps_z'].max(), sync_df['vins_z'].max())
    ax3.plot([min_z, max_z], [min_z, max_z], 'r--',
             alpha=0.8, label='Perfect correlation')
    ax3.set_xlabel('GPS Altitude (m)', fontsize=11)
    ax3.set_ylabel('VINS Altitude (m)', fontsize=11)
    ax3.set_title('Altitude Correlation', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Altitude error histogram
    ax4.hist(sync_df['error_z'], bins=30, alpha=0.7,
             color='purple', edgecolor='black')
    ax4.axvline(x=metrics['mean_error_z'], color='green', linestyle='--',
                label=f'Mean: {metrics["mean_error_z"]:.3f}m')
    ax4.axvline(x=np.percentile(sync_df['error_z'], 95), color='orange', linestyle='--',
                label=f'95th %: {np.percentile(sync_df["error_z"], 95):.3f}m')
    ax4.set_xlabel('Z Error (m)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Altitude Error Distribution', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(output_dir, 'altitude_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Altitude analysis plot saved: {plot_file}")
    plt.close()


def create_error_analysis_figure(sync_df, metrics, output_dir):
    """Create focused error analysis figure"""
    print("🎨 Creating error analysis figure...")

    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    time_rel = sync_df['timestamp'] - sync_df['timestamp'].iloc[0]

    # Error components over time
    ax1.plot(time_rel, sync_df['error_x'], 'r-',
             label='X Error', linewidth=2, alpha=0.8)
    ax1.plot(time_rel, sync_df['error_y'], 'g-',
             label='Y Error', linewidth=2, alpha=0.8)
    ax1.plot(time_rel, sync_df['error_z'], 'b-',
             label='Z Error', linewidth=2, alpha=0.8)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Position Error (m)', fontsize=11)
    ax1.set_title('Position Error Components Over Time', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2D and 3D error comparison
    ax2.plot(time_rel, sync_df['error_2d'], 'red',
             linewidth=2, label='2D Error', alpha=0.8)
    ax2.plot(time_rel, sync_df['error_3d'], 'purple',
             linewidth=2, label='3D Error', alpha=0.8)
    ax2.axhline(y=metrics['mean_error_2d'], color='red', linestyle='--', alpha=0.7,
                label=f'Mean 2D: {metrics["mean_error_2d"]:.3f}m')
    ax2.axhline(y=metrics['mean_error_3d'], color='purple', linestyle='--', alpha=0.7,
                label=f'Mean 3D: {metrics["mean_error_3d"]:.3f}m')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Position Error (m)', fontsize=11)
    ax2.set_title('2D vs 3D Error Over Time', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Error distribution comparison
    bins = np.linspace(
        0, max(sync_df['error_3d'].max(), sync_df['error_2d'].max()), 30)
    ax3.hist(sync_df['error_x'], bins=bins,
             alpha=0.6, label='X Error', color='red')
    ax3.hist(sync_df['error_y'], bins=bins, alpha=0.6,
             label='Y Error', color='green')
    ax3.hist(sync_df['error_z'], bins=bins,
             alpha=0.6, label='Z Error', color='blue')
    ax3.set_xlabel('Position Error (m)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Error Component Distributions', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Error statistics bar chart
    components = ['X', 'Y', 'Z', '2D', '3D']
    rmse_values = [metrics['rmse_x'], metrics['rmse_y'], metrics['rmse_z'],
                   metrics['rmse_2d'], metrics['rmse_3d']]
    mean_values = [abs(metrics['mean_error_x']), abs(metrics['mean_error_y']),
                   abs(metrics['mean_error_z']), metrics['mean_error_2d'], metrics['mean_error_3d']]

    x_pos = np.arange(len(components))
    width = 0.35

    ax4.bar(x_pos - width/2, rmse_values, width,
            label='RMSE', alpha=0.8, color='orange')
    ax4.bar(x_pos + width/2, mean_values, width,
            label='Mean', alpha=0.8, color='skyblue')
    ax4.set_xlabel('Error Component', fontsize=11)
    ax4.set_ylabel('Error (m)', fontsize=11)
    ax4.set_title('Error Statistics Summary', fontsize=12)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(components)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add text with key statistics
    stats_text = f"Key Statistics:\nRMSE 2D: {metrics['rmse_2d']:.3f} m\nRMSE 3D: {metrics['rmse_3d']:.3f} m\nMax 2D: {metrics['max_error_2d']:.3f} m\n95th %: {metrics['p95_error_2d']:.3f} m"
    ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(output_dir, 'error_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Error analysis plot saved: {plot_file}")
    plt.close()


def calculate_trajectory_distance(sync_df, trajectory_type):
    """Calculate total distance traveled for a trajectory"""
    if trajectory_type == 'vins':
        x_col, y_col = 'vins_x', 'vins_y'
    else:
        x_col, y_col = 'gps_x', 'gps_y'

    distances = np.sqrt(
        np.diff(sync_df[x_col])**2 + np.diff(sync_df[y_col])**2)
    return np.sum(distances)


def save_enhanced_results(sync_df, metrics, output_dir):
    """Save enhanced analysis results with detailed statistics"""
    print(f"💾 Saving enhanced results to {output_dir}...")

    os.makedirs(output_dir, exist_ok=True)

    # Save synchronized data
    sync_file = os.path.join(output_dir, 'synchronized_trajectories.csv')
    sync_df.to_csv(sync_file, index=False)
    print(f"✅ Synchronized data saved: {sync_file}")

    # Calculate additional metrics
    vins_distance = calculate_trajectory_distance(sync_df, 'vins')
    gps_distance = calculate_trajectory_distance(sync_df, 'gps')
    distance_error = abs(vins_distance - gps_distance)

    # Enhanced metrics file
    metrics_file = os.path.join(output_dir, 'detailed_error_analysis.txt')
    with open(metrics_file, 'w') as f:
        f.write("VINS-GPS Trajectory Error Analysis - Detailed Report\n")
        f.write("="*60 + "\n\n")

        f.write("DATASET INFORMATION\n")
        f.write("-"*30 + "\n")
        f.write(
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Points analyzed: {metrics['n_points']:,}\n")
        f.write(
            f"Duration: {metrics['duration']:.1f} seconds ({metrics['duration']/60:.1f} minutes)\n")
        f.write(
            f"Average frequency: {metrics['n_points']/metrics['duration']:.1f} Hz\n\n")

        f.write("TRAJECTORY COMPARISON\n")
        f.write("-"*30 + "\n")
        f.write(f"VINS total distance: {vins_distance:.2f} m\n")
        f.write(f"GPS total distance: {gps_distance:.2f} m\n")
        f.write(
            f"Distance difference: {distance_error:.2f} m ({distance_error/gps_distance*100:.1f}%)\n\n")

        f.write("POSITION ERRORS (meters)\n")
        f.write("-"*30 + "\n")
        f.write(
            f"Mean X error: {metrics['mean_error_x']:+.6f} ± {metrics['std_error_x']:.6f}\n")
        f.write(
            f"Mean Y error: {metrics['mean_error_y']:+.6f} ± {metrics['std_error_y']:.6f}\n")
        f.write(
            f"Mean Z error: {metrics['mean_error_z']:+.6f} ± {metrics['std_error_z']:.6f}\n")
        f.write(
            f"Mean 2D error: {metrics['mean_error_2d']:.6f} ± {metrics['std_error_2d']:.6f}\n")
        f.write(
            f"Mean 3D error: {metrics['mean_error_3d']:.6f} ± {metrics['std_error_3d']:.6f}\n\n")

        f.write("ROOT MEAN SQUARE ERRORS (RMSE)\n")
        f.write("-"*30 + "\n")
        f.write(f"RMSE X: {metrics['rmse_x']:.6f} m\n")
        f.write(f"RMSE Y: {metrics['rmse_y']:.6f} m\n")
        f.write(f"RMSE Z: {metrics['rmse_z']:.6f} m\n")
        f.write(f"RMSE 2D: {metrics['rmse_2d']:.6f} m\n")
        f.write(f"RMSE 3D: {metrics['rmse_3d']:.6f} m\n\n")

        f.write("ERROR DISTRIBUTION STATISTICS\n")
        f.write("-"*30 + "\n")
        f.write(
            f"Minimum 2D error: {metrics['min_error_2d'] if 'min_error_2d' in metrics else sync_df['error_2d'].min():.6f} m\n")
        f.write(f"Maximum 2D error: {metrics['max_error_2d']:.6f} m\n")
        f.write(f"50th percentile (median): {metrics['p50_error_2d']:.6f} m\n")
        f.write(f"95th percentile: {metrics['p95_error_2d']:.6f} m\n")
        f.write(f"99th percentile: {metrics['p99_error_2d']:.6f} m\n\n")

        f.write("PERFORMANCE ASSESSMENT\n")
        f.write("-"*30 + "\n")
        good_points = sum(sync_df['error_2d'] < 1.0)
        acceptable_points = sum(sync_df['error_2d'] < 2.0)
        f.write(
            f"Points with <1m error: {good_points:,} ({good_points/len(sync_df)*100:.1f}%)\n")
        f.write(
            f"Points with <2m error: {acceptable_points:,} ({acceptable_points/len(sync_df)*100:.1f}%)\n")
        f.write(
            f"Points with >5m error: {sum(sync_df['error_2d'] > 5.0):,} ({sum(sync_df['error_2d'] > 5.0)/len(sync_df)*100:.1f}%)\n\n")

        f.write("GENERATED FILES\n")
        f.write("-"*30 + "\n")
        f.write("1. vins_gps_analysis_comprehensive.png - Complete analysis overview\n")
        f.write("2. trajectory_comparison.png - Aligned trajectory paths\n")
        f.write("3. altitude_analysis.png - Altitude-focused analysis\n")
        f.write("4. error_analysis.png - Detailed error breakdown\n")
        f.write("5. synchronized_trajectories.csv - Raw synchronized data\n")
        f.write("6. detailed_error_analysis.txt - This comprehensive report\n")

    print(f"✅ Enhanced metrics saved: {metrics_file}")


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
    parser.add_argument('--no-altitude-alignment', action='store_true',
                        help='Disable automatic altitude alignment')

    args = parser.parse_args()

    print("🚁 VINS-GPS Trajectory Error Analysis")
    print("="*40)

    if args.no_alignment:
        print("⚠️  Trajectory alignment disabled")
    else:
        print("🎯 Automatic trajectory alignment enabled")

    if args.no_altitude_alignment:
        print("⚠️  Altitude alignment disabled")
    else:
        print("📏 Automatic altitude alignment enabled")

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
        vins_df, gps_df, args.max_time_diff, not args.no_alignment, not args.no_altitude_alignment)
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

    # Save enhanced results with detailed report
    save_enhanced_results(sync_df, metrics, args.output)

    # Create all plots separately
    print("\n🎨 Generating analysis figures...")

    # 1. Comprehensive overview plots (now separate figures)
    print("📊 Creating comprehensive analysis plots...")
    create_comprehensive_plots(sync_df, metrics, args.output)

    # 2. Focused trajectory comparison
    print("📊 Creating trajectory comparison plot...")
    create_trajectory_figure(sync_df, metrics, args.output)

    # 3. Altitude-focused analysis
    print("📊 Creating altitude analysis plot...")
    create_altitude_figure(sync_df, metrics, args.output)

    # 4. Error analysis breakdown
    print("📊 Creating error analysis plot...")
    create_error_analysis_figure(sync_df, metrics, args.output)

    print(f"\n✅ Complete analysis finished!")
    print(f"📁 All results saved to: {args.output}")
    print(f"📈 Generated 9 analysis figures:")
    print(f"   1. main_trajectory_comparison.png - Main trajectory view")
    print(f"   2. error_analysis_over_time.png - Error evolution")
    print(f"   3. error_distributions.png - Statistical distributions")
    print(f"   4. altitude_comparison.png - Altitude analysis")
    print(f"   5. statistics_summary.png - Complete statistics")
    print(f"   6. trajectory_comparison.png - Focused trajectory paths")
    print(f"   7. altitude_analysis.png - Detailed altitude analysis")
    print(f"   8. error_analysis.png - Comprehensive error breakdown")
    print(f"📋 Enhanced report: detailed_error_analysis.txt")


if __name__ == '__main__':
    main()
