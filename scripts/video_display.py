#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import argparse
import sys


class VideoDisplayNode(Node):
    def __init__(self, topic_name):
        super().__init__('video_display_node')

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Create subscriber for the image topic
        self.subscription = self.create_subscription(
            Image,
            topic_name,
            self.image_callback,
            10  # QoS depth
        )

        self.get_logger().info(f'Subscribing to topic: {topic_name}')
        self.get_logger().info('Press "q" in the image window to quit')

        # Variables for FPS calculation
        self.frame_count = 0
        self.last_time = self.get_clock().now()

    def image_callback(self, msg):
        try:
            # Convert ROS2 Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Resize for better visibility
            cv_image = cv2.resize(cv_image, (800, 600))
            # Calculate and display FPS
            self.frame_count += 1
            current_time = self.get_clock().now()
            time_diff = (current_time - self.last_time).nanoseconds / 1e9

            if time_diff >= 1.0:  # Update FPS every second
                fps = self.frame_count / time_diff
                self.get_logger().info(f'FPS: {fps:.2f}')
                self.frame_count = 0
                self.last_time = current_time

            # Add FPS text to image
            fps_text = f"FPS: {self.frame_count/max(time_diff, 0.001):.1f}"
            cv2.putText(cv_image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            # Add image info
            info_text = f"Size: {cv_image.shape[1]}x{cv_image.shape[0]}"
            cv2.putText(cv_image, info_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            # Display the image
            cv2.imshow('ROS2 Video Feed', cv_image)

            # Check for 'q' key press to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('Quit key pressed. Shutting down...')
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Display video from ROS2 topic')
    parser.add_argument('--topic', '-t',
                        default='/camera/image_color',
                        help='Image topic name (default: /camera/image_color)')
    parser.add_argument('--list-topics', '-l',
                        action='store_true',
                        help='List available image topics')

    args = parser.parse_args()

    # Initialize ROS2
    rclpy.init()

    if args.list_topics:
        # List available topics
        import subprocess
        try:
            result = subprocess.run(['ros2', 'topic', 'list', '-t'],
                                    capture_output=True, text=True)
            print("Available topics:")
            lines = result.stdout.strip().split('\n')
            image_topics = [line.split()[0]
                            for line in lines if 'sensor_msgs/msg/Image' in line]

            if image_topics:
                print("Image topics found:")
                for topic in image_topics:
                    print(f"  {topic}")
            else:
                print("No image topics found.")

        except Exception as e:
            print(f"Error listing topics: {e}")

        rclpy.shutdown()
        return

    try:
        # Create and run the node
        node = VideoDisplayNode(args.topic)

        print(f"Displaying video from topic: {args.topic}")
        print("Press 'q' in the image window to quit")

        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        try:
            node.destroy_node()
        except:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()
