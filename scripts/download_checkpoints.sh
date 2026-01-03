#!/bin/bash
# Download SAM2 checkpoints

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SAM2_DIR="$PROJECT_ROOT/segment-anything-2-real-time"
CHECKPOINTS_DIR="$SAM2_DIR/checkpoints"

echo "Downloading SAM2 checkpoints..."

if [ -d "$CHECKPOINTS_DIR" ]; then
    cd "$CHECKPOINTS_DIR"
    bash download_ckpts.sh
else
    echo "Error: SAM2 not installed. Run setup_sam2.sh first."
    exit 1
fi

echo "Checkpoints downloaded successfully!"
