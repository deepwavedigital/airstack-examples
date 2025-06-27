#!/bin/bash

set -euo pipefail

# ---- CONFIGURATION ----
ENV_FILE=$1
ENV_NAME=""
PYTHON_VERSION=""
PYTHON_TAG=""
CONDA_WHEELS_DIR="/opt/conda-wheels"
WHEEL_FILE=""
RPATH_VALUE='$ORIGIN/../../..'

# ---- STEP 0: Validate inputs ----
if [[ ! -f "$ENV_FILE" ]]; then
    echo "❌ Missing environment.yml file."
    exit 1
fi

# Extract env name
ENV_NAME=$(awk '/^name:/ { print $2 }' "$ENV_FILE")
if [[ -z "$ENV_NAME" ]]; then
    echo "❌ Could not determine environment name from $ENV_FILE"
    exit 1
fi

# Extract Python version (e.g., 3.10)
PYTHON_VERSION=$(awk '/python=/{gsub(/[[:space:]]/, "", $0); split($0, a, "="); print a[2]}' "$ENV_FILE" | head -n1)
if [[ -z "$PYTHON_VERSION" ]]; then
    echo "❌ Could not detect Python version in $ENV_FILE"
    exit 1
fi

# Convert to cpXY tag: e.g., 3.10 → cp310
PYTHON_TAG="cp$(echo "$PYTHON_VERSION" | tr -d '.')"

echo "🔧 Environment name: $ENV_NAME"
echo "🐍 Python version: $PYTHON_VERSION → $PYTHON_TAG"

# Verify conda-wheels directory
if [[ -d "$CONDA_WHEELS_DIR" && -r "$CONDA_WHEELS_DIR" && -x "$CONDA_WHEELS_DIR" ]]; then
    echo "✅ Conda wheels directory found"
else
    echo "❌ This script expects that TensorRT was extracted to $CONDA_WHEELS_DIR but directory was not found"
fi

# Find matching wheel
WHEEL_FILE=$(find "$CONDA_WHEELS_DIR" -type f -name "tensorrt-*-${PYTHON_TAG}-none-*.whl" 2>/dev/null | head -n1)
if [[ -z "$WHEEL_FILE" ]]; then
    echo "❌ No matching TensorRT wheel for Python version ($PYTHON_TAG) found."
    exit 1
fi

echo "📦 Using TensorRT wheel: $WHEEL_FILE"

# ---- STEP 1: Create Conda environment ----
echo "🚧 Creating Conda environment..."
conda env create -f "$ENV_FILE"

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# ---- STEP 2: Install TensorRT wheel ----
echo "📥 Installing TensorRT wheel..."
pip install "$WHEEL_FILE"

# ---- STEP 3: Locate tensorrt.so ----
echo "🔍 Locating tensorrt.so..."
TENSORRT_SO=$(find "$CONDA_PREFIX/lib/python3."*/site-packages/tensorrt -name "tensorrt.so" 2>/dev/null | head -n1)

if [[ -z "$TENSORRT_SO" ]]; then
    echo "❌ tensorrt.so not found after installing wheel."
    exit 1
fi

echo "✅ Found tensorrt.so at: $TENSORRT_SO"

# ---- STEP 4: Patch RPATH ----
echo "🔧 Setting RPATH to $RPATH_VALUE"
if ! command -v patchelf &> /dev/null; then
    echo "❌ patchelf not found. Please install it (e.g., conda install -c conda-forge patchelf)"
    exit 1
fi

patchelf --set-rpath "$RPATH_VALUE" "$TENSORRT_SO"

echo "✅ Patched RPATH"
readelf -d "$TENSORRT_SO" | grep -i rpath || echo "No RPATH set."

echo "🎉 TensorRT environment setup complete."
