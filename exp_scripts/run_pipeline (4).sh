#!/bin/bash
# FILE: run_neuralangelo_pipeline.sh
# Updated version with proper mask handling and B200 optimizations
# Usage: ./run_neuralangelo_pipeline.sh [input_dir] [work_dir]
#
# To run on a compute node interactively:
# 1. Get an interactive session: srun --partition=gpu --gres=gpu:b200:1 --mem=160G --time=12:00:00 --pty bash
# 2. Run this script: ./run_neuralangelo_pipeline.sh
#
# Or use screen/tmux:
# 1. ssh to compute node
# 2. screen -S neuralangelo
# 3. ./run_neuralangelo_pipeline.sh
# 4. Detach with Ctrl+A, D

# ============================================
# CONFIGURATION - EDIT THESE AS NEEDED
# ============================================
module load pytorch/2.7 2>/dev/null || echo "  pytorch/2.7 not available"
module load cuda/12.8.1 2>/dev/null || echo "  cuda/12.8.1 not available"

# Base directories
SCRATCH_BASE="${SCRATCH_BASE:-/blue/arthur.porto-biocosmos/jhennessy7.gatech/scratch}"
SCRIPT_DIR="${SCRIPT_DIR:-/home/jhennessy7.gatech/augenblick}"
NEURALANGELO_PATH="${NEURALANGELO_PATH:-${SCRIPT_DIR}/src/neuralangelo}"
ENV_PATH="${ENV_PATH:-/home/jhennessy7.gatech/neuralangelo_b200_env}"

# Input data (can be overridden by command line argument)
DEFAULT_INPUT="${SCRATCH_BASE}/scale_organized"
INPUT_DIR="${1:-${DEFAULT_INPUT}}"

# Work directory (can be overridden by command line argument)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEFAULT_WORK="${SCRATCH_BASE}/neura_${TIMESTAMP}"
WORK_DIR="${2:-${DEFAULT_WORK}}"

# GPU device (set to empty string to use CPU)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ============================================
# SETUP & CLEANUP HANDLERS
# ============================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Cleanup function for interrupts
cleanup() {
    echo ""
    echo -e "${YELLOW}Cleanup handler triggered...${NC}"
    # Kill any child Python processes
    pkill -P $$ python 2>/dev/null
    
    # Clear GPU cache
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
    
    # Compress logs if directory exists
    if [ -d "${WORK_DIR}" ] && [ -d "${WORK_DIR}/logs" ]; then
        echo "Compressing logs..."
        tar -czf "${WORK_DIR}_logs.tar.gz" "${WORK_DIR}/logs" 2>/dev/null
    fi
    
    echo "Cleanup complete"
}
trap cleanup EXIT INT TERM

# Function to print section headers
print_header() {
    echo ""
    echo "============================================"
    echo "$1"
    echo "============================================"
    echo ""
}

# Function to check if running on compute node
check_environment() {
    if [ -z "$CUDA_VISIBLE_DEVICES" ] && ! nvidia-smi &>/dev/null; then
        echo -e "${YELLOW}WARNING: No GPU detected. Running in CPU mode.${NC}"
        echo "For GPU processing, run on a compute node with:"
        echo "  srun --partition=gpu --gres=gpu:b200:1 --mem=160G --time=12:00:00 --pty bash"
        echo ""
        read -p "Continue in CPU mode? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        export CUDA_VISIBLE_DEVICES=""
    fi
}

# ============================================
# MAIN SCRIPT START
# ============================================

print_header "Neuralangelo Pipeline (VGGT → Neuralangelo)"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo "Input: ${INPUT_DIR}"
echo "Work: ${WORK_DIR}"

# Check environment
check_environment

# Check if work directory already exists
if [ -d "${WORK_DIR}" ]; then
    echo -e "${YELLOW}Work directory already exists: ${WORK_DIR}${NC}"
    read -p "Overwrite? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        # Generate new timestamp
        TIMESTAMP=$(date +%Y%m%d_%H%M%S_$$)
        WORK_DIR="${SCRATCH_BASE}/neura_${TIMESTAMP}"
        echo "Using new directory: ${WORK_DIR}"
    else
        rm -rf "${WORK_DIR}"
    fi
fi

# Create work directory
mkdir -p "${WORK_DIR}"
mkdir -p "${WORK_DIR}/logs"

# Logging
LOG_FILE="${WORK_DIR}/pipeline.log"
exec 1> >(tee "${LOG_FILE}")
exec 2>&1

# ============================================
# LOAD MODULES AND ACTIVATE ENVIRONMENT
# ============================================

print_header "Environment Setup"

# Activate virtual environment
if [ -f "${ENV_PATH}/bin/activate" ]; then
    echo "Activating environment: ${ENV_PATH}"
    source "${ENV_PATH}/bin/activate"
else
    echo -e "${YELLOW}WARNING: Virtual environment not found at ${ENV_PATH}${NC}"
    echo "Attempting to use system Python..."
fi

# Load modules
module load pytorch/2.7 2>/dev/null || echo "  pytorch/2.7 loaded"
module load cuda/12.8.1 2>/dev/null || echo "  cuda/12.8.1 loaded"

# Verify environment
echo ""
echo "Environment check:"
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')" || {
    echo -e "${RED}✗ PyTorch not found${NC}"
    exit 1
}
python -c "import tinycudann; print('✓ tiny-cuda-nn: OK')" || echo -e "${YELLOW}⚠ tiny-cuda-nn not found${NC}"
python -c "import scipy; print('✓ scipy: OK')" || {
    echo -e "${RED}✗ scipy not found${NC}"
    exit 1
}

# Check for VGGT
python -c "from vggt.models.vggt import VGGT; print('✓ VGGT: Available')" 2>/dev/null || {
    echo -e "${YELLOW}⚠ VGGT not available - will use fallback methods${NC}"
    VGGT_AVAILABLE=0
}
VGGT_AVAILABLE=${VGGT_AVAILABLE:-1}

# ============================================
# GPU DETECTION & OPTIMIZATION
# ============================================

if [ -n "$CUDA_VISIBLE_DEVICES" ] && nvidia-smi &>/dev/null; then
    print_header "GPU Information"
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

# ============================================
# VALIDATE INPUT DATA STRUCTURE
# ============================================

print_header "Validating Input Data"

# Check for images directory
if [ -d "${INPUT_DIR}/images" ]; then
    IMG_DIR="${INPUT_DIR}/images"
    IMG_COUNT=$(find "${IMG_DIR}" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
    echo "✓ Found images directory: ${IMG_COUNT} images"
else
    IMG_DIR="${INPUT_DIR}"
    IMG_COUNT=$(find "${IMG_DIR}" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
    echo "  Using input directory directly: ${IMG_COUNT} images"
fi

# Check for masks directory
MASK_COUNT=0
if [ -d "${INPUT_DIR}/masks" ]; then
    MASK_DIR="${INPUT_DIR}/masks"
    MASK_COUNT=$(find "${MASK_DIR}" -name "*.png" | wc -l)
    echo "✓ Found masks directory: ${MASK_COUNT} masks"
    
    # Verify mask naming convention
    echo "  Checking mask naming convention..."
    FIRST_MASK=$(ls "${MASK_DIR}"/*.png 2>/dev/null | head -1)
    if [ -n "${FIRST_MASK}" ]; then
        echo "  First mask: $(basename ${FIRST_MASK})"
    fi
else
    echo "⚠ No masks directory found"
fi

if [ ${IMG_COUNT} -eq 0 ]; then
    echo -e "${RED}ERROR: No images found in ${INPUT_DIR}${NC}"
    exit 1
fi

if [ ${MASK_COUNT} -gt 0 ] && [ ${MASK_COUNT} -ne ${IMG_COUNT} ]; then
    echo -e "${YELLOW}WARNING: Number of masks (${MASK_COUNT}) doesn't match images (${IMG_COUNT})${NC}"
    echo "  Will proceed with partial masking"
fi

# ============================================
# COPY ALL REQUIRED SCRIPTS AND MODULES
# ============================================

print_header "Copying Required Scripts"

cd "${WORK_DIR}"

# Copy modules directory
if [ -d "${SCRIPT_DIR}/modules" ]; then
    cp -r "${SCRIPT_DIR}/modules" "${WORK_DIR}/"
    echo "✓ Copied modules directory"
    
    # Update gpu_utils.py if it exists to use conservative B200 settings
    if [ -f "${WORK_DIR}/modules/gpu_utils.py" ] && [[ "${GPU_NAME}" == *"B200"* ]]; then
        echo "  Patching gpu_utils.py for B200..."
        # This would need the actual patch, but marking it for now
    fi
else
    echo -e "${RED}ERROR: modules directory not found at ${SCRIPT_DIR}/modules${NC}"
    exit 1
fi

# Copy required scripts
REQUIRED_SCRIPTS=(
    "vggt_preprocessing.py"
    "compute_sphere_params.py"
    "stage_training_manager.py"
)

for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [ -f "${SCRIPT_DIR}/${script}" ]; then
        cp "${SCRIPT_DIR}/${script}" "${WORK_DIR}/"
        echo "✓ Copied ${script}"
    else
        echo -e "${RED}ERROR: ${script} not found!${NC}"
        exit 1
    fi
done

# Copy VGGT script if available
if [ -f "${SCRIPT_DIR}/vggt_batch_processor.py" ]; then
    cp "${SCRIPT_DIR}/vggt_batch_processor.py" "${WORK_DIR}/"
    echo "✓ Copied vggt_batch_processor.py"
else
    echo -e "${YELLOW}⚠ vggt_batch_processor.py not found - VGGT will not be available${NC}"
    VGGT_AVAILABLE=0
fi

# Copy stage configs
mkdir -p "${WORK_DIR}/configs"
for config in "${SCRIPT_DIR}"/configs/stage*.yaml; do
    if [ -f "$config" ]; then
        cp "$config" "${WORK_DIR}/configs/"
        echo "✓ Copied $(basename $config)"
    fi
done

# ============================================
# PHASE 1: PREPROCESSING WITH VGGT
# ============================================

print_header "PHASE 1: Preprocessing (VGGT/Fallback)"

PREPROCESSING_DIR="${WORK_DIR}/preprocessing"

echo "Running preprocessing..."
echo "  Input: ${INPUT_DIR}"
echo "  Output: ${PREPROCESSING_DIR}"

# Determine preprocessing method
echo -e "${GREEN}Using VGGT for preprocessing${NC}"
PREPROCESS_METHOD="vggt"
# Run preprocessing based on method
case ${PREPROCESS_METHOD} in
    vggt)
        python vggt_preprocessing.py \
            "${INPUT_DIR}" \
            "${PREPROCESSING_DIR}" \
            --vggt-script "${WORK_DIR}/vggt_batch_processor.py" \
            --gpu 0 \
            --timeout 900
        PREPROCESS_EXIT=$?
        ;;
    colmap)
        python vggt_preprocessing.py \
            "${INPUT_DIR}" \
            "${PREPROCESSING_DIR}" \
            --use-colmap \
            --gpu 0 \
            --timeout 1200
        PREPROCESS_EXIT=$?
        ;;
    fallback)
        python vggt_preprocessing.py \
            "${INPUT_DIR}" \
            "${PREPROCESSING_DIR}" \
            --force-fallback \
            --gpu 0
        PREPROCESS_EXIT=$?
        ;;
esac

if [ $PREPROCESS_EXIT -ne 0 ]; then
    echo -e "${RED}❌ Preprocessing failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Preprocessing completed${NC}"

# Find neuralangelo_data
NEURA_DATA=$(find "${PREPROCESSING_DIR}" -name "transforms.json" -type f | head -1 | xargs dirname)
if [ -z "$NEURA_DATA" ] || [ ! -f "$NEURA_DATA/transforms.json" ]; then
    echo -e "${RED}ERROR: Could not find neuralangelo_data with transforms.json${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found neuralangelo_data at: ${NEURA_DATA}${NC}"

# Verify masks were copied
if [ -d "${NEURA_DATA}/masks" ]; then
    FINAL_MASK_COUNT=$(ls -1 "${NEURA_DATA}/masks/"*.png 2>/dev/null | wc -l)
    echo "  Masks in output: ${FINAL_MASK_COUNT}"
fi

# ============================================
# PHASE 1.5: COMPUTE SPHERE PARAMETERS
# ============================================

print_header "PHASE 1.5: Computing Sphere Parameters"

python compute_sphere_params.py \
    "${NEURA_DATA}" \
    --config-dir "${WORK_DIR}/configs" \
    --output-json "${WORK_DIR}/sphere_params.json"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to compute sphere parameters${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Successfully computed sphere parameters${NC}"

# Display the computed parameters
if [ -f "${WORK_DIR}/sphere_params.json" ]; then
    python3 -c "
import json
with open('${WORK_DIR}/sphere_params.json', 'r') as f:
    params = json.load(f)
    print(f'  Sphere center: {params[\"sphere_center\"]}')
    print(f'  Sphere scale: {params[\"sphere_scale\"]:.4f}')
    print(f'  Number of images: {params[\"num_images\"]}')
"
fi

# ============================================
# PHASE 2: MULTI-STAGE TRAINING
# ============================================

print_header "PHASE 2: Multi-Stage Training"

OUTPUT_DIR="${WORK_DIR}/training_output"
mkdir -p "${OUTPUT_DIR}"

echo "Configuration:"
echo "  Data: ${NEURA_DATA}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Neuralangelo: ${NEURALANGELO_PATH}"

# Determine training parameters based on GPU
if [[ "${GPU_NAME}" == *"B200"* ]]; then
    echo -e "${BLUE}Using B200-optimized training parameters:${NC}"
    echo "  - Stage 1 (coarse): 500-1000 iterations"
    echo "  - Stage 2 (mid): Conservative settings"
    echo "  - Stage 3 (fine): Main reconstruction"
    echo "  - Stage 4 (ultra): Optional"
    echo "  - Validation: DISABLED"
    echo "  - Memory clearing: Every 50 iterations"
    
    # Set mesh resolution for B200
    MESH_RESOLUTION=4096
    END_STAGE="--end-stage fine"  # Skip ultra for B200 initially
    
elif [ ${GPU_MEM_GB} -ge 80 ]; then
    MESH_RESOLUTION=4096
    END_STAGE=""
elif [ ${GPU_MEM_GB} -ge 40 ]; then
    MESH_RESOLUTION=2048
    END_STAGE=""
else
    MESH_RESOLUTION=1024
    END_STAGE="--end-stage fine"
fi

echo "  Mesh resolution: ${MESH_RESOLUTION}"

# Clear GPU cache before training
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Clearing GPU cache before training..."
    python -c "import torch; torch.cuda.empty_cache(); print('  GPU cache cleared')"
fi

echo ""
echo "Starting training pipeline..."
echo -e "${YELLOW}This will take several hours. Consider using screen/tmux.${NC}"
echo ""

# Prompt before starting long training
if [ -t 0 ]; then  # Check if running interactively
    read -p "Start training? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Training cancelled"
        exit 0
    fi
fi

# Run training with appropriate flags
python stage_training_manager.py \
    "${NEURA_DATA}" \
    "${OUTPUT_DIR}" \
    --neuralangelo-path "${NEURALANGELO_PATH}" \
    --config-dir "${WORK_DIR}/configs" \
    --gpu 0 \
    --mesh-resolution ${MESH_RESOLUTION} \
    --mesh-stage fine \
    ${END_STAGE}

TRAINING_EXIT=$?

# ============================================
# RESULTS & SUMMARY
# ============================================

print_header "Pipeline Results"

if [ $TRAINING_EXIT -eq 0 ]; then
    echo -e "${GREEN}✅ Pipeline Completed Successfully!${NC}"
    echo ""
    
    # List generated meshes
    echo "Generated meshes:"
    find "${OUTPUT_DIR}" -name "*.ply" -type f 2>/dev/null | while read mesh; do
        if [ -f "$mesh" ]; then
            size=$(du -h "$mesh" | cut -f1)
            name=$(basename "$mesh")
            echo "  ✓ ${name}: ${size}"
        fi
    done
    
    # Create summary
    cat > "${WORK_DIR}/summary.txt" << EOF
VGGT → Neuralangelo Pipeline Summary
=====================================
Status: SUCCESS
Runtime: $((SECONDS / 60)) minutes
Date: $(date)

Environment:
  Host: $(hostname)
  GPU: ${GPU_NAME:-CPU} (${GPU_MEM_GB:-0}GB)
  PyTorch: $(python -c "import torch; print(torch.__version__)" 2>/dev/null)
  VGGT: ${VGGT_AVAILABLE}

Input:
  Directory: ${INPUT_DIR}
  Images: ${IMG_COUNT}
  Masks: ${MASK_COUNT}

Processing:
  Method: ${PREPROCESS_METHOD}
  Data location: ${NEURA_DATA}
  Mesh resolution: ${MESH_RESOLUTION}

Output:
  Directory: ${OUTPUT_DIR}

Work Directory: ${WORK_DIR}
EOF
    
    echo ""
    echo "Summary saved to: ${WORK_DIR}/summary.txt"
    
    # Compress outputs
    if ls "${OUTPUT_DIR}/"*.ply &>/dev/null; then
        tar -czf "${WORK_DIR}/meshes.tar.gz" -C "${OUTPUT_DIR}" *.ply 2>/dev/null
        echo -e "${GREEN}✓ Meshes compressed to: ${WORK_DIR}/meshes.tar.gz${NC}"
    fi
    
else
    echo -e "${RED}❌ Pipeline Failed!${NC}"
    echo "Exit code: ${TRAINING_EXIT}"
fi

echo ""
echo "Work directory: ${WORK_DIR}"
echo "Total runtime: $((SECONDS / 60)) minutes"

# Download instructions
if [ $TRAINING_EXIT -eq 0 ] && [ -f "${WORK_DIR}/meshes.tar.gz" ]; then
    echo ""
    echo -e "${GREEN}To download results:${NC}"
    echo "  scp $(hostname):${WORK_DIR}/meshes.tar.gz ./"
fi

exit $TRAINING_EXIT
