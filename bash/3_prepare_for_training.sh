#!/bin/bash
# FILE: prepare_for_training.sh
# Prepare for training
# Usage: ./prepare_for_training.sh
#
# This script prepares for training by setting the mesh resolution and clearing the GPU cache.
# It also checks the GPU name and memory to determine the appropriate mesh resolution.

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================
# PREPARE FOR TRAINING
# ============================================
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_MEM_GB=$((GPU_MEM_MB / 1024))
echo "Name: ${GPU_NAME}"
echo "Memory: ${GPU_MEM_GB} GB (${GPU_MEM_MB} MB)"
echo "CUDA Device: ${CUDA_VISIBLE_DEVICES}"

if [[ "${GPU_NAME}" == *"B200"* ]]; then
    echo -e "${BLUE}Using B200-optimized training parameters:${NC}"
    echo "  - Stage 1 (coarse): 500-1000 iterations"
    echo "  - Stage 2 (mid): Conservative settings"
    echo "  - Stage 3 (fine): Main reconstruction"
    echo "  - Stage 4 (ultra): Optional"
    echo "  - Validation: DISABLED"
    echo "  - Memory clearing: Every 50 iterations"
    
    # Set mesh resolution for B200
    export MESH_RESOLUTION=4096
    export END_STAGE="--end-stage fine"  # Skip ultra for B200 initially
    
elif [[ "${GPU_MEM_GB}" -ge 80 ]]; then
    export MESH_RESOLUTION=4096
    export END_STAGE=""
elif [[ "${GPU_MEM_GB}" -ge 40 ]]; then
    export MESH_RESOLUTION=2048
    export END_STAGE=""
else
    export MESH_RESOLUTION=1024
    export END_STAGE="--end-stage fine"
fi

echo "  Mesh resolution: ${MESH_RESOLUTION}"

# Clear GPU cache before training
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Clearing GPU cache before training..."
    python -c "import torch; torch.cuda.empty_cache(); print('  GPU cache cleared')"
fi