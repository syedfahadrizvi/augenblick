#!/bin/bash
# FILE: optimize_gpu.sh
# Optimize the GPU for the Augenblick pipeline
# Usage: ./optimize_gpu.sh
#
# This script detects the GPU and optimizes it for the Augenblick pipeline.
# It also checks the PyTorch version for B200 compatibility.

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================
# GPU DETECTION & OPTIMIZATION
# ============================================

if [ -n "$CUDA_VISIBLE_DEVICES" ] && nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    GPU_MEM_GB=$((GPU_MEM_MB / 1024))
    echo "Name: ${GPU_NAME}"
    echo "Memory: ${GPU_MEM_GB} GB (${GPU_MEM_MB} MB)"
    echo "CUDA Device: ${CUDA_VISIBLE_DEVICES}"
    
    # Export for Python scripts
    export GPU_NAME="${GPU_NAME}"
    export GPU_MEMORY_MB="${GPU_MEM_MB}"
    
    # GPU-specific optimizations
    if [[ "${GPU_NAME}" == *"B200"* ]]; then
        echo -e "${BLUE}Detected NVIDIA B200 - Applying optimizations${NC}"
        echo "  - Setting memory allocation to expandable segments"
        echo "  - Configuring for conservative training parameters"
        echo "  - Disabling validation to save memory"
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
        export CUDA_LAUNCH_BLOCKING=0
        export NEURALANGELO_SKIP_VALIDATION=1
        export NEURALANGELO_CONSERVATIVE_MODE=1
        
        # Check PyTorch version for B200 compatibility
        python -c "
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f'  CUDA Capability: {cap[0]}.{cap[1]}')
    if cap[0] >= 10:
        print('  ⚠️  B200 (sm_100) may require PyTorch 2.5+ for full support')
"
    elif [[ "${GPU_NAME}" == *"H100"* ]] || [[ "${GPU_NAME}" == *"H200"* ]]; then
        echo "Detected high-end GPU - applying optimizations"
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    elif [[ "${GPU_NAME}" == *"A100"* ]]; then
        echo "Detected A100 - applying optimizations"
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    else
        echo "Standard GPU configuration"
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    fi
else
    echo -e "${YELLOW}Running in CPU mode${NC}"
    GPU_MEM_GB=0
fi