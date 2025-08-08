#!/usr/bin/env python3

"""
Quick test script to demonstrate trajectory alignment functionality
"""

import numpy as np
import matplotlib.pyplot as plt
import math


def create_test_trajectories():
    """Create simulated VINS and GPS trajectories with known rotation"""
    # Create a figure-8 pattern for GPS trajectory
    t = np.linspace(0, 4*np.pi, 100)
    gps_x = 10 * np.sin(t)
    gps_y = 5 * np.sin(2*t)

    # Create VINS trajectory with 180° rotation and some noise
    rotation_angle = 180  # degrees
    angle_rad = np.radians(rotation_angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    # Apply rotation
    vins_x = cos_a * gps_x - sin_a * gps_y
    vins_y = sin_a * gps_x + cos_a * gps_y

    # Add some noise and offset to make it realistic
    vins_x += np.random.normal(0, 0.5, len(vins_x)) + 2
    vins_y += np.random.normal(0, 0.5, len(vins_y)) - 1

    return gps_x, gps_y, vins_x, vins_y, rotation_angle


def align_trajectories_demo(vins_positions, gps_positions):
    """Demonstrate trajectory alignment algorithm"""
    # Convert to numpy arrays
    vins_pos = np.array(vins_positions)
    gps_pos = np.array(gps_positions)

    # Center both trajectories
    vins_centered = vins_pos - np.mean(vins_pos, axis=0)
    gps_centered = gps_pos - np.mean(gps_pos, axis=0)

    print("🔍 Testing trajectory alignment...")

    best_error = float('inf')
    best_rotation = 0
    errors_by_angle = []

    # Test rotations from 0 to 360 degrees
    angles = range(0, 360, 5)
    for angle_deg in angles:
        angle_rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # 2D rotation matrix (we only need X-Y)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])

        # Apply rotation to VINS trajectory
        vins_rotated = vins_centered[:, :2] @ rotation_matrix.T

        # Calculate alignment error
        error = np.sum((vins_rotated - gps_centered[:, :2])**2)
        errors_by_angle.append(error)

        if error < best_error:
            best_error = error
            best_rotation = angle_deg

    print(f"✅ Best alignment found: {best_rotation}° rotation")
    print(f"   Error reduction: {errors_by_angle[0]:.1f} → {best_error:.1f}")

    # Apply best rotation
    best_angle_rad = np.radians(best_rotation)
    cos_a, sin_a = np.cos(best_angle_rad), np.sin(best_angle_rad)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ])

    vins_aligned = (
        vins_pos[:, :2] - np.mean(vins_pos[:, :2], axis=0)) @ rotation_matrix.T
    vins_aligned += np.mean(gps_pos[:, :2], axis=0)

    return vins_aligned, best_rotation, angles, errors_by_angle


def plot_alignment_demo():
    """Create demonstration plots"""
    # Generate test data
    gps_x, gps_y, vins_x, vins_y, true_rotation = create_test_trajectories()

    # Prepare 3D positions (Z=0 for 2D case)
    gps_positions = np.column_stack([gps_x, gps_y, np.zeros(len(gps_x))])
    vins_positions = np.column_stack([vins_x, vins_y, np.zeros(len(vins_x))])

    # Perform alignment
    aligned_positions, detected_rotation, angles, errors = align_trajectories_demo(
        vins_positions, gps_positions
    )

    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Original trajectories (misaligned)
    ax1.plot(gps_x, gps_y, 'r-', linewidth=2,
             label='GPS Ground Truth', alpha=0.8)
    ax1.plot(vins_x, vins_y, 'b-', linewidth=2,
             label='VINS (Original)', alpha=0.8)
    ax1.plot(gps_x[0], gps_y[0], 'go', markersize=8, label='Start')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title(f'Before Alignment\n(True rotation: {true_rotation}°)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Plot 2: Aligned trajectories
    ax2.plot(gps_x, gps_y, 'r-', linewidth=2,
             label='GPS Ground Truth', alpha=0.8)
    ax2.plot(aligned_positions[:, 0], aligned_positions[:, 1], 'g-',
             linewidth=2, label='VINS (Aligned)', alpha=0.8)
    ax2.plot(gps_x[0], gps_y[0], 'go', markersize=8, label='Start')
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title(
        f'After Alignment\n(Detected rotation: {detected_rotation}°)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # Plot 3: Error vs rotation angle
    ax3.plot(angles, errors, 'b-', linewidth=2)
    ax3.axvline(x=detected_rotation, color='r', linestyle='--',
                label=f'Best angle: {detected_rotation}°')
    ax3.axvline(x=true_rotation, color='g', linestyle='--',
                label=f'True angle: {true_rotation}°')
    ax3.set_xlabel('Rotation Angle (degrees)')
    ax3.set_ylabel('Alignment Error')
    ax3.set_title('Error vs Rotation Angle')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Error comparison
    original_error = np.sqrt(
        np.mean((vins_x - gps_x)**2 + (vins_y - gps_y)**2))
    aligned_error = np.sqrt(np.mean(
        (aligned_positions[:, 0] - gps_x)**2 +
        (aligned_positions[:, 1] - gps_y)**2
    ))

    categories = ['Before\nAlignment', 'After\nAlignment']
    errors_rms = [original_error, aligned_error]
    colors = ['red', 'green']

    bars = ax4.bar(categories, errors_rms, color=colors, alpha=0.7)
    ax4.set_ylabel('RMS Error (m)')
    ax4.set_title('Alignment Effectiveness')
    ax4.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, error in zip(bars, errors_rms):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{error:.2f}m', ha='center', va='bottom')

    plt.tight_layout()
    plt.suptitle('VINS-GPS Trajectory Alignment Demo', fontsize=16, y=1.02)

    print(f"\n📊 Results Summary:")
    print(f"   True rotation: {true_rotation}°")
    print(f"   Detected rotation: {detected_rotation}°")
    print(f"   Detection error: {abs(detected_rotation - true_rotation)}°")
    print(f"   RMS error before: {original_error:.3f}m")
    print(f"   RMS error after: {aligned_error:.3f}m")
    print(f"   Error reduction: {(1 - aligned_error/original_error)*100:.1f}%")

    plt.show()


if __name__ == '__main__':
    print("🚁 VINS-GPS Trajectory Alignment Demo")
    print("="*40)
    print("This demo shows how trajectory alignment fixes the")
    print("180° rotation problem common in VIO systems.\n")

    plot_alignment_demo()

    print("\n✅ Demo complete!")
    print("This is why automatic alignment is crucial for")
    print("accurate VINS-GPS error analysis.")
