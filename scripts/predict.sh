#!/bin/bash

# Prediction shell script wrapper for the aerial image segmentation model.
#
# This script provides a convenient CLI interface to:
# - Run inference on single images
# - Set model configuration
# - Select compute device
# - Specify input/output paths

# Default values
CONFIG="configs/inference.yaml"
DEVICE="mps"
INPUT_IMAGE="data/test_image.jpg"
OUTPUT_DIR="predictions"

mkdir -p "$OUTPUT_DIR"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -i|--input)
            INPUT_IMAGE="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Check if input image is provided
if [ -z "$INPUT_IMAGE" ]; then
    echo "Error: Input image path is required"
    echo "Usage: ./scripts/predict.sh --input path/to/image.jpg [--config path/to/config.yaml] [--device cpu|cuda|mps]"
    exit 1
fi

# Run prediction
poetry run python scripts/predict.py \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --input "$INPUT_IMAGE"

echo "Prediction complete. Results saved to $OUTPUT_DIR"