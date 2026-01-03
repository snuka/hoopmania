#!/bin/bash
# Setup script for SAM2 (Segment Anything Model 2)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SAM2_DIR="$PROJECT_ROOT/segment-anything-2-real-time"

echo "==================================="
echo "SAM2 Setup Script"
echo "==================================="

# Clone repository if not exists
if [ ! -d "$SAM2_DIR" ]; then
    echo "Cloning SAM2 repository..."
    cd "$PROJECT_ROOT"
    git clone https://github.com/Gy920/segment-anything-2-real-time.git
else
    echo "SAM2 repository already exists"
fi

# Install as editable package
echo "Installing SAM2 package..."
cd "$SAM2_DIR"
pip install -e .

# Build Cython extensions
echo "Building Cython extensions..."
python setup.py build_ext --inplace

# Download checkpoints
echo "Downloading checkpoints..."
cd "$SAM2_DIR/checkpoints"
if [ ! -f "sam2.1_hiera_large.pt" ]; then
    bash download_ckpts.sh
else
    echo "Checkpoints already downloaded"
fi

echo ""
echo "==================================="
echo "SAM2 setup complete!"
echo "==================================="
