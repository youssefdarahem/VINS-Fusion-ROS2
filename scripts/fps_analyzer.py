#!/usr/bin/env python3

"""
FPS Profiling Analysis Tool
Analyzes and visualizes FPS profiling results from fps_profiler.py
"""

import json
import csv
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import statistics


class FPSAnalyzer:
    """Analyze FPS profiling data."""
    
    def __init__(self, csv_file=None, json_file=None):
        self.csv_file = csv_file
        self.json_file = json_file
        self.csv_data = None
        self.json_data = None
        
        if csv_file:
            self.load_csv(csv_file)
        if json_file:
            self.load_json(json_file)
    
    def load_csv(self, filepath):
        """Load and parse CSV file."""
        self.csv_data = []
        try:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric fields
                    for key in row:
                        if key != 'timestamp':
                            try:
                                row[key] = float(row[key])
                            except ValueError:
                                pass
                    self.csv_data.append(row)
            print(f"✅ Loaded {len(self.csv_data)} records from {filepath}")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
    
    def load_json(self, filepath):
        """Load and parse JSON file."""
        self.json_data = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    self.json_data.append(json.loads(line))
            print(f"✅ Loaded {len(self.json_data)} records from {filepath}")
        except Exception as e:
            print(f"❌ Error loading JSON: {e}")
    
    def print_csv_summary(self):
        """Print summary statistics from CSV data."""
        if not self.csv_data or len(self.csv_data) < 2:
            print("⚠️  Not enough CSV data for analysis")
            return
        
        topics = ['image_mono_fps', 'image_stereo_fps', 'imu_fps', 'odometry_fps', 
                  'path_fps', 'pose_fps', 'pointcloud_fps']
        latency_topics = ['odometry_latency_ms', 'pose_latency_ms']
        
        print("\n" + "="*80)
        print("📊 FPS STATISTICS SUMMARY")
        print("="*80)
        
        # Filter out zero/inactive values
        for topic in topics:
            values = [row[topic] for row in self.csv_data if row.get(topic, 0) > 0.1]
            if values:
                print(f"\n{topic.replace('_', ' ').title()}:")
                print(f"  Mean:   {statistics.mean(values):8.2f} FPS")
                print(f"  Median: {statistics.median(values):8.2f} FPS")
                print(f"  Min:    {min(values):8.2f} FPS")
                print(f"  Max:    {max(values):8.2f} FPS")
                if len(values) > 1:
                    print(f"  StdDev: {statistics.stdev(values):8.2f} FPS")
        
        print(f"\n{'LATENCY ANALYSIS':^80}")
        for topic in latency_topics:
            values = [row[topic] for row in self.csv_data if row.get(topic, 0) > 0]
            if values:
                print(f"\n{topic.replace('_', ' ').title()}:")
                print(f"  Mean:   {statistics.mean(values):8.2f} ms")
                print(f"  Median: {statistics.median(values):8.2f} ms")
                print(f"  Min:    {min(values):8.2f} ms")
                print(f"  Max:    {max(values):8.2f} ms")
                if len(values) > 1:
                    print(f"  StdDev: {statistics.stdev(values):8.2f} ms")
        
        print("\n" + "="*80)
    
    def print_json_summary(self):
        """Print summary from JSON data."""
        if not self.json_data or len(self.json_data) < 2:
            print("⚠️  Not enough JSON data for analysis")
            return
        
        print("\n" + "="*80)
        print("📈 TIME-SERIES ANALYSIS")
        print("="*80)
        
        # Extract all topic names from first record
        first_record = self.json_data[0]
        topics = list(first_record['fps_stats'].keys())
        
        for topic in topics:
            values = []
            for record in self.json_data:
                fps = record['fps_stats'].get(topic, {}).get('fps', 0)
                if fps > 0.1:
                    values.append(fps)
            
            if values:
                is_active = all(
                    record['fps_stats'].get(topic, {}).get('active', False) 
                    for record in self.json_data[-5:]  # Check last 5 records
                )
                status = "✅ ACTIVE" if is_active else "⚠️  INACTIVE"
                
                print(f"\n{topic.replace('_', ' ').title()} {status}:")
                print(f"  Mean:   {statistics.mean(values):8.2f} FPS")
                print(f"  Median: {statistics.median(values):8.2f} FPS")
                print(f"  Min:    {min(values):8.2f} FPS")
                print(f"  Max:    {max(values):8.2f} FPS")
                
                # Check for stability
                if len(values) > 5:
                    recent = values[-5:]
                    variance = max(recent) - min(recent)
                    stability = "Stable ✅" if variance < 5 else f"Variable ({variance:.1f} FPS)"
                    print(f"  Stability: {stability}")
        
        print("\n" + "="*80)
    
    def detect_bottlenecks(self):
        """Detect potential bottlenecks."""
        if not self.csv_data:
            return
        
        print("\n" + "="*80)
        print("🔍 BOTTLENECK DETECTION")
        print("="*80)
        
        # Get average FPS for inputs and outputs
        input_fps = []
        output_fps = []
        
        for row in self.csv_data:
            input_fps.extend([row.get('image_mono_fps', 0), row.get('imu_fps', 0)])
            output_fps.extend([row.get('odometry_fps', 0), row.get('pose_fps', 0)])
        
        input_fps = [x for x in input_fps if x > 0.1]
        output_fps = [x for x in output_fps if x > 0.1]
        
        if input_fps and output_fps:
            avg_input = statistics.mean(input_fps)
            avg_output = statistics.mean(output_fps)
            ratio = avg_output / avg_input if avg_input > 0 else 0
            
            print(f"\nAverage Input FPS:  {avg_input:.2f}")
            print(f"Average Output FPS: {avg_output:.2f}")
            print(f"Throughput Ratio:   {ratio:.1%}")
            
            if ratio < 0.5:
                print("\n⚠️  CRITICAL: Severe bottleneck detected!")
                print("   • Processing speed is <50% of input speed")
                print("   • Recommendations:")
                print("     - Reduce image resolution")
                print("     - Lower feature tracking quality")
                print("     - Enable GPU mode")
                print("     - Check CPU usage (top, htop)")
            elif ratio < 0.8:
                print("\n⚠️  WARNING: Moderate bottleneck detected")
                print("   • Processing speed is 50-80% of input speed")
                print("   • Consider optimizations")
            else:
                print("\n✅ No significant bottleneck detected")
        
        print("\n" + "="*80)
    
    def export_comparison(self, output_file):
        """Export comparison data for multiple runs."""
        if not self.csv_data:
            print("No CSV data to export")
            return
        
        try:
            with open(output_file, 'w') as f:
                f.write("# FPS Profiling Comparison\n")
                f.write(f"File: {self.csv_file}\n\n")
                
                f.write("| Metric | Mean | Median | Min | Max |\n")
                f.write("|--------|------|--------|-----|-----|\n")
                
                topics = ['image_mono_fps', 'imu_fps', 'odometry_fps', 'pose_fps']
                for topic in topics:
                    values = [row[topic] for row in self.csv_data if row.get(topic, 0) > 0.1]
                    if values:
                        f.write(f"| {topic.replace('_fps', '').title()} | "
                               f"{statistics.mean(values):.2f} | "
                               f"{statistics.median(values):.2f} | "
                               f"{min(values):.2f} | "
                               f"{max(values):.2f} |\n")
            
            print(f"✅ Comparison exported to {output_file}")
        except Exception as e:
            print(f"❌ Error exporting: {e}")


def find_latest_results():
    """Find the latest FPS profiling results."""
    results_dir = Path('fps_profiling_results')
    
    if not results_dir.exists():
        print("No fps_profiling_results directory found")
        return None, None
    
    csv_files = sorted(results_dir.glob('fps_stats_*.csv'), reverse=True)
    json_files = sorted(results_dir.glob('fps_log_*.json'), reverse=True)
    
    csv_file = csv_files[0] if csv_files else None
    json_file = json_files[0] if json_files else None
    
    return csv_file, json_file


def main():
    parser = argparse.ArgumentParser(
        description='Analyze FPS profiling results from fps_profiler.py'
    )
    parser.add_argument('--csv', help='CSV file path (auto-detects if not specified)')
    parser.add_argument('--json', help='JSON file path (auto-detects if not specified)')
    parser.add_argument('--export', help='Export comparison to markdown file')
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    
    args = parser.parse_args()
    
    # Auto-detect if not specified
    csv_file = args.csv
    json_file = args.json
    
    if not csv_file or not json_file:
        detected_csv, detected_json = find_latest_results()
        if not csv_file:
            csv_file = detected_csv
        if not json_file:
            json_file = detected_json
    
    if not csv_file and not json_file:
        print("❌ No profiling results found. Run fps_profiler.py first.")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = FPSAnalyzer(csv_file=csv_file, json_file=json_file)
    
    # Run analyses
    if csv_file:
        analyzer.print_csv_summary()
    
    if json_file:
        analyzer.print_json_summary()
    
    analyzer.detect_bottlenecks()
    
    if args.export:
        analyzer.export_comparison(args.export)


if __name__ == '__main__':
    main()
