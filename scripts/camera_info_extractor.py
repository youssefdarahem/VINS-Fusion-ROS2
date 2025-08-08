#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage
import json
import numpy as np
import cv2
from cv_bridge import CvBridge
import os
from datetime import datetime


class CameraInfoExtractor(Node):
    def __init__(self):
        super().__init__('camera_info_extractor')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Camera info storage
        self.camera_info = None
        self.info_received = False
        self.image_count = 0
        self.sample_images_saved = 0
        self.max_sample_images = 5

        # Create output directory
        self.output_dir = "camera_extraction_output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Subscribers
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_callback, 10)
        self.image_sub = self.create_subscription(
            CompressedImage, '/camera/image_color/compressed', self.image_callback, 10)
        self.mono_image_sub = self.create_subscription(
            CompressedImage, '/camera/image_mono_compressed/compressed', self.mono_image_callback, 10)

        self.get_logger().info("Camera Info Extractor started")
        self.get_logger().info(f"Output directory: {self.output_dir}")

    def camera_info_callback(self, msg):
        """Extract and save camera calibration info"""
        if not self.info_received:
            self.camera_info = msg
            self.info_received = True
            print(msg)

            # Extract camera parameters
            camera_data = {
                'header': {
                    'frame_id': msg.header.frame_id,
                    'timestamp': {
                        'sec': msg.header.stamp.sec,
                        'nanosec': msg.header.stamp.nanosec
                    }
                },
                'image_dimensions': {
                    'width': msg.width,
                    'height': msg.height
                },
                'camera_matrix': {
                    'data': msg.k.tolist(),
                    'rows': 3,
                    'cols': 3,
                    'fx': msg.k[0],
                    'fy': msg.k[4],
                    'cx': msg.k[2],
                    'cy': msg.k[5]
                },
                'distortion_model': msg.distortion_model,
                'distortion_coefficients': msg.d.tolist(),
                'rectification_matrix': msg.r.tolist(),
                'projection_matrix': {
                    'data': msg.p.tolist(),
                    'rows': 3,
                    'cols': 4
                },
                'binning': {
                    'x': msg.binning_x,
                    'y': msg.binning_y
                },
                'roi': {
                    'x_offset': msg.roi.x_offset,
                    'y_offset': msg.roi.y_offset,
                    'height': msg.roi.height,
                    'width': msg.roi.width,
                    'do_rectify': msg.roi.do_rectify
                }
            }

            # Save to JSON file
            json_file = os.path.join(self.output_dir, 'camera_info.json')
            with open(json_file, 'w') as f:
                json.dump(camera_data, f, indent=2)

            # Save human-readable summary
            summary_file = os.path.join(self.output_dir, 'camera_summary.txt')
            with open(summary_file, 'w') as f:
                f.write("Camera Calibration Summary\n")
                f.write("==========================\n\n")
                f.write(f"Frame ID: {msg.header.frame_id}\n")
                f.write(f"Image Size: {msg.width} x {msg.height}\n")
                f.write(f"Distortion Model: {msg.distortion_model}\n\n")

                f.write("Intrinsic Parameters:\n")
                f.write(f"  fx = {msg.k[0]:.4f}\n")
                f.write(f"  fy = {msg.k[4]:.4f}\n")
                f.write(f"  cx = {msg.k[2]:.4f}\n")
                f.write(f"  cy = {msg.k[5]:.4f}\n\n")

                f.write("Camera Matrix K:\n")
                k_matrix = np.array(msg.k).reshape(3, 3)
                for row in k_matrix:
                    f.write(
                        f"  [{row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f}]\n")

                if len(msg.d) > 0:
                    f.write(f"\nDistortion Coefficients: {msg.d}\n")

                f.write(f"\nProjection Matrix P:\n")
                p_matrix = np.array(msg.p).reshape(3, 4)
                for row in p_matrix:
                    f.write(
                        f"  [{row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f} {row[3]:10.4f}]\n")

            self.get_logger().info(f"✅ Camera info saved to {json_file}")
            self.get_logger().info(f"📋 Camera summary saved to {summary_file}")

            # Print key parameters
            self.get_logger().info(
                f"📸 Camera: {msg.width}x{msg.height}, fx={msg.k[0]:.1f}, fy={msg.k[4]:.1f}")

    def image_callback(self, msg):
        """Save sample color images"""
        if self.sample_images_saved < self.max_sample_images:
            try:
                # Convert compressed image to OpenCV format
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if cv_image is not None:
                    timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    filename = f"sample_color_{self.sample_images_saved:03d}_{timestamp:.3f}.jpg"
                    filepath = os.path.join(self.output_dir, filename)

                    cv2.imwrite(filepath, cv_image)
                    self.sample_images_saved += 1

                    self.get_logger().info(
                        f"🖼️  Saved color image: {filename}")

                    if self.sample_images_saved == 1:
                        # Save image info for the first image
                        info_file = os.path.join(
                            self.output_dir, 'image_info.txt')
                        with open(info_file, 'w') as f:
                            f.write("Sample Image Information\n")
                            f.write("========================\n\n")
                            f.write(f"Image shape: {cv_image.shape}\n")
                            f.write(
                                f"Channels: {cv_image.shape[2] if len(cv_image.shape) > 2 else 1}\n")
                            f.write(f"Data type: {cv_image.dtype}\n")
                            f.write(f"Encoding: BGR (OpenCV default)\n")
                            f.write(f"Timestamp: {timestamp:.6f}\n")
                            f.write(f"Frame ID: {msg.header.frame_id}\n")

            except Exception as e:
                self.get_logger().error(f"Error saving color image: {e}")

        self.image_count += 1

    def mono_image_callback(self, msg):
        """Save sample mono images"""
        if self.sample_images_saved < self.max_sample_images:
            try:
                # Convert compressed image to OpenCV format
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

                if cv_image is not None:
                    timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    filename = f"sample_mono_{self.sample_images_saved:03d}_{timestamp:.3f}.jpg"
                    filepath = os.path.join(self.output_dir, filename)

                    cv2.imwrite(filepath, cv_image)

                    self.get_logger().info(f"🖼️  Saved mono image: {filename}")

            except Exception as e:
                self.get_logger().error(f"Error saving mono image: {e}")

    def print_final_summary(self):
        """Print final extraction summary"""
        summary_file = os.path.join(self.output_dir, 'extraction_summary.txt')
        with open(summary_file, 'w') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Camera Extraction Summary\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"========================\n\n")
            f.write(
                f"Camera info received: {'✅ Yes' if self.info_received else '❌ No'}\n")
            f.write(f"Total images processed: {self.image_count}\n")
            f.write(f"Sample images saved: {self.sample_images_saved}\n\n")

            if self.camera_info:
                f.write(f"Camera specifications:\n")
                f.write(
                    f"  Resolution: {self.camera_info.width}x{self.camera_info.height}\n")
                f.write(
                    f"  Focal length: fx={self.camera_info.k[0]:.2f}, fy={self.camera_info.k[4]:.2f}\n")
                f.write(
                    f"  Principal point: cx={self.camera_info.k[2]:.2f}, cy={self.camera_info.k[5]:.2f}\n")
                f.write(
                    f"  Distortion model: {self.camera_info.distortion_model}\n")

        print(f"\n📋 Final summary saved to: {summary_file}")


def main():
    rclpy.init()

    try:
        extractor = CameraInfoExtractor()

        print("🎯 Camera Info Extractor")
        print("========================")
        print("Extracting camera calibration and sample images...")
        print("This will extract:")
        print("- Camera calibration parameters (JSON + human-readable)")
        print("- Sample color and mono images")
        print("- Image specifications")
        print("\nPress Ctrl+C to stop and generate final summary...")
        print("")

        # Run for a limited time or until interrupted
        rclpy.spin(extractor)

    except KeyboardInterrupt:
        print("\n⏹️  Extraction stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'extractor' in locals():
            extractor.print_final_summary()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
