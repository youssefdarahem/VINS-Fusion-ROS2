#!/usr/bin/env python3

import sys
import json
import cv2
import numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage, Image, Imu, CameraInfo
from rosidl_runtime_py.utilities import get_message
from cv_bridge import CvBridge


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: python3 hybvio_convert.py <bag_path> <image_topic> <imu_topic> [camera_info_topic]")
        print("\nExamples:")
        print("  # With CameraInfo (for intrinsics):")
        print("  python3 hybvio_convert.py bag/ /camera/image_mono /imu/data /camera/camera_info")
        print("\n  # Without CameraInfo (uses default intrinsics):")
        print("  python3 hybvio_convert.py bag/ /camera/image_mono /imu/data")
        sys.exit(1)

    bag_path = sys.argv[1]
    image_topic = sys.argv[2]
    imu_topic = sys.argv[3]
    camera_info_topic = sys.argv[4] if len(sys.argv) > 4 else None

    bridge = CvBridge()

    bridge = CvBridge()

    # === Step 1: Read CameraInfo (optional) ===
    camera_info = None
    if camera_info_topic:
        print(f"📷 Reading camera info from {camera_info_topic}...")
        storage_options = StorageOptions(uri=bag_path, storage_id="sqlite3")
        converter_options = ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr"
        )

        reader = SequentialReader()
        reader.open(storage_options, converter_options)

        # Get topic types
        topic_types = reader.get_all_topics_and_types()
        topic_type_map = {t.name: t.type for t in topic_types}

        while reader.has_next():
            topic, data, t = reader.read_next()
            if topic == camera_info_topic:
                msg_type = get_message(topic_type_map[topic])
                camera_info = deserialize_message(data, msg_type)
                break

        if camera_info is None:
            print(
                f"⚠️  Warning: No CameraInfo found on topic {camera_info_topic}, using defaults")
        else:
            print(
                f"✅ Camera info loaded: {camera_info.width}x{camera_info.height}")
        del reader
    else:
        print("📷 No CameraInfo topic provided, will use default intrinsics")

    # === Step 2: Process full bag for IMU + images ===
    print(f"📊 Processing IMU and images...")
    storage_options = StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # Get topic types
    topic_types = reader.get_all_topics_and_types()
    topic_type_map = {t.name: t.type for t in topic_types}

    # Detect image format
    image_msg_type = topic_type_map.get(image_topic, "")
    is_compressed = "CompressedImage" in image_msg_type
    print(f"  🖼️  Image format: {'Compressed' if is_compressed else 'Raw'}")

    jsonl_entries = []
    frame_counter = 0
    video_writer = None
    first_frame = True
    imu_count = 0
    img_count = 0

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()

        # Skip topics we don't care about
        if topic not in [image_topic, imu_topic]:
            continue

        t_sec = timestamp_ns / 1e9

        if topic == imu_topic:
            msg_type = get_message(topic_type_map[topic])
            msg = deserialize_message(data, msg_type)

            # Accelerometer
            jsonl_entries.append({
                "sensor": {"type": "accelerometer", "values": [
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z
                ]},
                "time": t_sec
            })
            # Gyroscope
            jsonl_entries.append({
                "sensor": {"type": "gyroscope", "values": [
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z
                ]},
                "time": t_sec
            })
            imu_count += 1
            if imu_count % 1000 == 0:
                print(f"  📡 Processed {imu_count} IMU messages...")

        elif topic == image_topic:
            msg_type = get_message(topic_type_map[topic])
            msg = deserialize_message(data, msg_type)

            # Handle both compressed and raw images
            if is_compressed:
                np_arr = np.frombuffer(msg.data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                # Raw Image message
                img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            if img is None:
                continue

            if first_frame:
                height, width = img.shape[:2]
                # Estimate FPS from camera info or default to 20
                fps = 20.0
                video_writer = cv2.VideoWriter(
                    "data.mp4",
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (width, height)
                )
                first_frame = False
                print(f"  🎥 Video output: {width}x{height} @ {fps} fps")

            video_writer.write(img)

            jsonl_entries.append({
                "frames": [{"cameraInd": 0, "time": t_sec}],
                "number": frame_counter,
                "time": t_sec
            })
            frame_counter += 1
            img_count += 1
            if img_count % 100 == 0:
                print(f"  🖼️  Processed {img_count} images...")

    if video_writer:
        video_writer.release()

    print(f"\n📝 Sorting {len(jsonl_entries)} entries by timestamp...")
    # Sort by time (critical for HybVIO)
    jsonl_entries.sort(key=lambda x: x["time"])

    print(f"💾 Writing data.jsonl...")
    with open("data.jsonl", "w") as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry) + "\n")

    # === Step 3: Write parameters.txt ===
    print(f"⚙️  Writing parameters.txt...")

    # Use CameraInfo if available, otherwise use defaults
    if camera_info:
        K = camera_info.k  # [fx, 0, cx, 0, fy, cy, ...]
        D = list(camera_info.d) if camera_info.d else []
        img_width = camera_info.width
        img_height = camera_info.height
    else:
        # Default intrinsics (adjust these for your camera!)
        img_width = 1440
        img_height = 1080
        fx = fy = 800.0  # Typical for wide-angle
        cx = img_width / 2.0
        cy = img_height / 2.0
        K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        D = [0.0, 0.0, 0.0, 0.0, 0.0]
        print(
            f"  ⚠️  Using default intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    # Use first 3 distortion coeffs (OpenCV model); pad if needed
    if len(D) < 3:
        D += [0.0] * (3 - len(D))
    distortion_coeffs = D[:3]

    # Default IMU-to-camera transform (IDENTITY — YOU MUST REPLACE THIS!)
    # Format: column-major 4x4 matrix as flat list
    imu_to_cam = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    ]

    with open("parameters.txt", "w") as f:
        f.write(f"focalLengthX {K[0]}; focalLengthY {K[4]};\n")
        f.write(f"principalPointX {K[2]}; principalPointY {K[5]};\n")
        f.write(f"distortionCoeffs {','.join(map(str, distortion_coeffs))};\n")
        f.write("fisheyeCamera false;\n")
        f.write(f"imuToCameraMatrix {','.join(map(str, imu_to_cam))};\n")

    print("\n✅ Conversion complete!")
    print(f"   📊 Processed: {imu_count} IMU messages, {img_count} images")
    print(f"   📁 Created files:")
    print(f"      - data.jsonl ({len(jsonl_entries)} entries)")
    print(f"      - data.mp4")
    print(f"      - parameters.txt")
    print("\n⚠️  WARNING: imuToCameraMatrix is IDENTITY. Replace with real calibration from:")
    print(f"      body_T_cam0 matrix in your VINS config file!")


if __name__ == "__main__":
    main()
