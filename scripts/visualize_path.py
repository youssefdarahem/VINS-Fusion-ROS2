#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import glob


def plot_flight_path(csv_file, show_orientation=False):
    """
    Visualize the flight path from VINS trajectory data

    Args:
        csv_file: Path to the CSV file with trajectory data
        show_orientation: Whether to show orientation arrows along the path
    """
    # Read the CSV file
    try:
        data = pd.read_csv(csv_file)
        print(f"Loaded {len(data)} trajectory points from {csv_file}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Extract x, y coordinates
    x = data['x'].values
    y = data['y'].values

    # Create the plot
    plt.figure(figsize=(12, 10))

    # Plot the flight path
    plt.plot(x, y, 'b-', linewidth=2, alpha=0.7, label='Flight Path')

    # Mark start and end points
    plt.plot(x[0], y[0], 'go', markersize=10, label='Start')
    plt.plot(x[-1], y[-1], 'ro', markersize=10, label='End')

    # Add orientation arrows if requested
    if show_orientation and len(data) > 10:
        # Sample every N points to avoid clutter
        skip = max(1, len(data) // 20)
        for i in range(0, len(data) - 1, skip):
            if i + skip < len(data):
                dx = x[i + skip] - x[i]
                dy = y[i + skip] - y[i]
                # Normalize and scale the arrow
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    dx_norm = dx / length * 0.5  # Scale factor for arrow size
                    dy_norm = dy / length * 0.5
                    plt.arrow(x[i], y[i], dx_norm, dy_norm,
                              head_width=0.2, head_length=0.1,
                              fc='red', ec='red', alpha=0.6)

    # Customize the plot
    plt.xlabel('X Position (meters)', fontsize=12)
    plt.ylabel('Y Position (meters)', fontsize=12)
    plt.title('Flight Path Visualization (Top-Down View)',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')  # Equal aspect ratio for accurate representation

    # Add statistics
    distance = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    duration = data['timestamp'].iloc[-1] - data['timestamp'].iloc[0]

    stats_text = f"Total Distance: {distance:.2f}m\nDuration: {duration:.1f}s\nAvg Speed: {distance/duration:.2f}m/s"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save the plot
    output_file = csv_file.replace('.csv', '_path_visualization.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")

    # Show the plot
    plt.show()


def plot_altitude_profile(csv_file):
    """
    Plot the altitude profile over time
    """
    try:
        data = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    plt.figure(figsize=(12, 6))

    # Calculate relative time
    time_rel = data['timestamp'] - data['timestamp'].iloc[0]

    plt.plot(time_rel, data['z'], 'b-', linewidth=2)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Z Position (meters)', fontsize=12)
    plt.title('Altitude Profile', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Add statistics
    z_min, z_max = data['z'].min(), data['z'].max()
    z_range = z_max - z_min
    stats_text = f"Min Z: {z_min:.2f}m\nMax Z: {z_max:.2f}m\nRange: {z_range:.2f}m"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()

    # Save the plot
    output_file = csv_file.replace('.csv', '_altitude_profile.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Altitude profile saved to: {output_file}")

    plt.show()


def find_latest_files(directory="/tmp/vins_output"):
    """
    Find the latest VINS output files
    """
    if not os.path.exists(directory):
        print(f"Output directory {directory} does not exist")
        return []

    # Look for CSV files
    pattern = os.path.join(directory, "*.csv")
    files = glob.glob(pattern)

    if not files:
        print(f"No CSV files found in {directory}")
        return []

    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    return files


def main():
    parser = argparse.ArgumentParser(
        description='Visualize VINS flight trajectory')
    parser.add_argument('--file', '-f', type=str, help='Path to CSV file')
    parser.add_argument('--orientation', '-o', action='store_true',
                        help='Show orientation arrows along the path')
    parser.add_argument('--altitude', '-a', action='store_true',
                        help='Also plot altitude profile')
    parser.add_argument('--latest', '-l', action='store_true',
                        help='Use the latest files from /tmp/vins_output')

    args = parser.parse_args()

    if args.latest or not args.file:
        # Find latest files
        files = find_latest_files()
        if not files:
            print(
                "No files found. Please run VINS logger first or specify a file with --file")
            return

        print(f"Found {len(files)} files:")
        for i, f in enumerate(files[:5]):  # Show first 5
            print(f"  {i+1}: {f}")

        # Use the most recent file
        csv_file = files[0]
        print(f"\nUsing latest file: {csv_file}")
    else:
        csv_file = args.file

    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return

    # Plot the flight path
    plot_flight_path(csv_file, args.orientation)

    # Plot altitude profile if requested
    if args.altitude:
        plot_altitude_profile(csv_file)


if __name__ == '__main__':
    main()
