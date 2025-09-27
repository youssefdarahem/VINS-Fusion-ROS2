#!/usr/bin/env python3

import os
import glob
import subprocess
import argparse
from datetime import datetime


def process_bag_files(bag_pattern, output_dir, extract_script):
    """Process multiple bag files matching the pattern"""

    # Find all bag files matching the pattern
    bag_files = glob.glob(bag_pattern)

    if not bag_files:
        print(f"❌ No bag files found matching pattern: {bag_pattern}")
        return

    print(f"🔍 Found {len(bag_files)} bag file(s):")
    for i, bag_file in enumerate(bag_files, 1):
        print(f"   {i}. {bag_file}")

    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)

    successful_extractions = 0
    failed_extractions = 0

    for i, bag_file in enumerate(bag_files, 1):
        print(f"\n{'='*60}")
        print(
            f"📦 Processing bag {i}/{len(bag_files)}: {os.path.basename(bag_file)}")
        print(f"{'='*60}")

        # Create subdirectory for this bag
        bag_name = os.path.splitext(os.path.basename(bag_file))[0]
        bag_output_dir = os.path.join(output_dir, bag_name)

        try:
            # Run the extraction script
            cmd = [
                'python3', extract_script,
                bag_file,
                '--output-dir', bag_output_dir,
                '--prefix', f"bag_{i:03d}"
            ]

            print(f"🔧 Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print(f"✅ Successfully processed: {bag_file}")
                successful_extractions += 1
            else:
                print(f"❌ Failed to process: {bag_file}")
                print(f"Error: {result.stderr}")
                failed_extractions += 1

        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout processing: {bag_file}")
            failed_extractions += 1
        except Exception as e:
            print(f"❌ Error processing {bag_file}: {e}")
            failed_extractions += 1

    # Final summary
    print(f"\n{'='*60}")
    print(f"📊 BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful extractions: {successful_extractions}")
    print(f"❌ Failed extractions: {failed_extractions}")
    print(f"📁 Output directory: {output_dir}")

    if successful_extractions > 0:
        print(f"\n🎯 To analyze extracted data:")
        print(f"   # For individual bag analysis:")
        for bag_file in bag_files[:3]:  # Show first 3 examples
            bag_name = os.path.splitext(os.path.basename(bag_file))[0]
            vins_file = os.path.join(
                output_dir, bag_name, f"*_vins_odometry.csv")
            gps_file = os.path.join(output_dir, bag_name, f"*_gps.csv")
            print(
                f"   python offline_vins_gps_analysis.py --vins {vins_file} --gps {gps_file}")

        if len(bag_files) > 3:
            print(f"   # ... and {len(bag_files) - 3} more")


def main():
    parser = argparse.ArgumentParser(
        description='Batch extract VINS odometry and GPS data from multiple ROS2 bags')
    parser.add_argument('bag_pattern',
                        help='Glob pattern for bag files (e.g., "*.db3" or "/path/to/bags/*.db3")')
    parser.add_argument('--output-dir', default='./batch_extracted_data',
                        help='Output directory for all extracted CSV files')
    parser.add_argument('--extract-script', default='./extract_vins_odometry.py',
                        help='Path to the extraction script')

    args = parser.parse_args()

    print("🚁 Batch VINS-GPS Data Extractor")
    print("="*40)
    print(f"🔍 Pattern: {args.bag_pattern}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🔧 Script: {args.extract_script}")

    # Check if extraction script exists
    if not os.path.exists(args.extract_script):
        print(f"❌ Extraction script not found: {args.extract_script}")
        return

    # Process all bag files
    process_bag_files(args.bag_pattern, args.output_dir, args.extract_script)


if __name__ == '__main__':
    main()
