#!/bin/bash
#
# VGGT to SuGaR Pipeline
# Processes images through VGGT and converts to SuGaR format
#

set -e  # Exit on error

# Default values
INPUT_DIR=""
OUTPUT_DIR=""
DEVICE="cuda"
TRAIN_SPLIT=0.9

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --train_split)
            TRAIN_SPLIT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 -i <input_dir> -o <output_dir> [options]"
            echo ""
            echo "Options:"
            echo "  -i, --input       Input images directory (required)"
            echo "  -o, --output      Output directory (required)"
            echo "  --device          Device to use (default: cuda)"
            echo "  --train_split     Train/test split ratio (default: 0.9)"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [[ -z "$INPUT_DIR" ]] || [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: Input and output directories are required"
    echo "Usage: $0 -i <input_dir> -o <output_dir>"
    exit 1
fi

# Check if input directory exists
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

echo "========================================"
echo "VGGT to SuGaR Pipeline"
echo "========================================"
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Device:           $DEVICE"
echo "Train/test split: $TRAIN_SPLIT"
echo ""

# Create temporary directory for VGGT output
TEMP_DIR="${OUTPUT_DIR}_vggt_temp"
echo "Creating temporary directory: $TEMP_DIR"
mkdir -p "$TEMP_DIR"

# Step 1: Run VGGT batch processor
echo ""
echo "Step 1: Running VGGT batch processor..."
echo "----------------------------------------"
python vggt_batch_processor.py "$INPUT_DIR" --output_dir "$TEMP_DIR" --device "$DEVICE"

if [[ $? -ne 0 ]]; then
    echo "Error: VGGT processing failed"
    exit 1
fi

# Step 2: Convert to SuGaR format
echo ""
echo "Step 2: Converting to SuGaR format..."
echo "----------------------------------------"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Read the transforms.json from VGGT output
TRANSFORMS_JSON="${TEMP_DIR}/transforms.json"
if [[ ! -f "$TRANSFORMS_JSON" ]]; then
    echo "Error: transforms.json not found in VGGT output"
    exit 1
fi

# Python script to convert VGGT format to SuGaR format
python3 << EOF
import json
import shutil
from pathlib import Path
import numpy as np

# Paths
temp_dir = Path("$TEMP_DIR")
output_dir = Path("$OUTPUT_DIR")
train_split = float("$TRAIN_SPLIT")

# Load VGGT transforms
with open(temp_dir / "transforms.json") as f:
    vggt_data = json.load(f)

frames = vggt_data["frames"]
num_frames = len(frames)
num_train = int(num_frames * train_split)

print(f"Total frames: {num_frames}")
print(f"Train frames: {num_train}")
print(f"Test frames: {num_frames - num_train}")

# Copy images to output directory
print("Copying images...")
for frame in frames:
    src = temp_dir / frame["file_path"]
    dst = output_dir / Path(frame["file_path"]).name
    shutil.copy2(src, dst)

# Copy depth maps if they exist
if (temp_dir / "depth").exists():
    print("Copying depth maps...")
    (output_dir / "depth").mkdir(exist_ok=True)
    for depth_file in (temp_dir / "depth").glob("*"):
        shutil.copy2(depth_file, output_dir / "depth" / depth_file.name)

# Get camera parameters from first frame
first_intrinsics = frames[0]["intrinsics"]
camera_params = {
    "fl_x": first_intrinsics["fx"],
    "fl_y": first_intrinsics["fy"],
    "cx": first_intrinsics["cx"],
    "cy": first_intrinsics["cy"],
    "w": 518,  # VGGT output size
    "h": 518,
    "camera_angle_x": 2 * np.arctan(518 / (2 * first_intrinsics["fx"])),
    "camera_angle_y": 2 * np.arctan(518 / (2 * first_intrinsics["fy"]))
}

# Create train frames
train_frames = []
for i in range(num_train):
    frame = frames[i]
    train_frames.append({
        "file_path": f"./{Path(frame['file_path']).name}",
        "transform_matrix": frame["transform_matrix"]
    })

# Create test frames
test_frames = []
for i in range(num_train, num_frames):
    frame = frames[i]
    test_frames.append({
        "file_path": f"./{Path(frame['file_path']).name}",
        "transform_matrix": frame["transform_matrix"]
    })

# Save transforms_train.json
train_data = {
    **camera_params,
    "frames": train_frames
}
with open(output_dir / "transforms_train.json", 'w') as f:
    json.dump(train_data, f, indent=2)

# Save transforms_test.json
test_data = {
    **camera_params,
    "frames": test_frames
}
with open(output_dir / "transforms_test.json", 'w') as f:
    json.dump(test_data, f, indent=2)

print(f"✅ Saved transforms_train.json ({len(train_frames)} frames)")
print(f"✅ Saved transforms_test.json ({len(test_frames)} frames)")

# Save metadata
metadata = {
    "num_frames": num_frames,
    "num_train": num_train,
    "num_test": num_frames - num_train,
    "format": "blender",
    "source": "vggt_batch_processor"
}
with open(output_dir / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Conversion complete!")
EOF

if [[ $? -ne 0 ]]; then
    echo "Error: Conversion to SuGaR format failed"
    exit 1
fi

# Cleanup temporary directory
echo ""
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo ""
echo "========================================"
echo "✅ Pipeline Complete!"
echo "========================================"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Review camera poses:"
echo "     cat $OUTPUT_DIR/transforms_train.json"
echo ""
echo "  2. Run SuGaR training (full pipeline):"
echo "     python train_full_pipeline.py -s $OUTPUT_DIR"
echo ""
echo "  3. Or run individual SuGaR training steps:"
echo "     python train.py -s $OUTPUT_DIR -c <checkpoint_path> -r density"
echo ""
