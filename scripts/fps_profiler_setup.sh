#!/bin/bash

# FPS Profiler Quick Start
# This script sets up and runs the complete FPS profiling workflow

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "FPS PROFILER - QUICK START SETUP"
echo "=========================================="
echo ""

# Make scripts executable
echo "📦 Making scripts executable..."
chmod +x "$SCRIPT_DIR/fps_profiler.py"
chmod +x "$SCRIPT_DIR/fps_analyzer.py"

# Create results directory
mkdir -p "$PROJECT_DIR/fps_profiling_results"
echo "✅ Created results directory"

echo ""
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. In Terminal 1 - Start VINS node:"
echo "   source install/setup.bash"
echo "   ros2 run vins vins_node config/euroc/euroc_mono_imu_config.yaml"
echo ""
echo "2. In Terminal 2 - Play rosbag dataset:"
echo "   ros2 bag play /path/to/your/dataset --read-ahead-queue-size 5000"
echo ""
echo "3. In Terminal 3 - Run FPS Profiler:"
echo "   python scripts/fps_profiler.py"
echo ""
echo "4. After collection - Analyze results:"
echo "   python scripts/fps_analyzer.py"
echo "   (or specify files: python scripts/fps_analyzer.py --csv path/to/file.csv)"
echo ""
echo "=========================================="
