#!/bin/bash
# FILE: run_vggt_sugar_pipeline.sh
# VGGT → Sugar Pipeline with B200 optimizations
# Usage: ./run_vggt_sugar_pipeline.sh [input_dir] [work_dir]
#
# To run on a compute node interactively:
# 1. Get an interactive session: srun --partition=gpu --gres=gpu:b200:1 --mem=160G --time=12:00:00 --pty bash
# 2. Run this script: ./run_vggt_sugar_pipeline.sh
#
# Or use screen/tmux:
# 1. ssh to compute node
# 2. screen -S sugar
# 3. ./run_vggt_sugar_pipeline.sh
# 4. Detach with Ctrl+A, D

# ============================================
# CONFIGURATION - EDIT THESE AS NEEDED
# ============================================
module load pytorch/2.7 2>/dev/null || echo "  pytorch/2.7 not available"
module load cuda/12.8.1 2>/dev/null || echo "  cuda/12.8.1 not available"

# Base directories
SCRATCH_BASE="${SCRATCH_BASE:-/blue/arthur.porto-biocosmos/jhennessy7.gatech/scratch}"
SCRIPT_DIR="${SCRIPT_DIR:-/home/jhennessy7.gatech/augenblick}"
SUGAR_PATH="${SUGAR_PATH:-${SCRIPT_DIR}/src/sugar}"
ENV_PATH="${ENV_PATH:-/home/jhennessy7.gatech/sugar_env}"

# Input data (can be overridden by command line argument)
DEFAULT_INPUT="${SCRATCH_BASE}/scale_organized"
INPUT_DIR="${1:-${DEFAULT_INPUT}}"

# Work directory (can be overridden by command line argument)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEFAULT_WORK="${SCRATCH_BASE}/sugar_${TIMESTAMP}"
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

print_header "VGGT → Sugar Pipeline"
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
        WORK_DIR="${SCRATCH_BASE}/sugar_${TIMESTAMP}"
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

# Check for Sugar dependencies
python -c "import diff_gaussian_rasterization; print('✓ diff-gaussian-rasterization: OK')" || {
    echo -e "${YELLOW}⚠ diff-gaussian-rasterization not found${NC}"
}
python -c "import simple_knn; print('✓ simple-knn: OK')" || {
    echo -e "${YELLOW}⚠ simple-knn not found${NC}"
}

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
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
        export CUDA_LAUNCH_BLOCKING=0

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
else
    echo -e "${RED}ERROR: modules directory not found at ${SCRIPT_DIR}/modules${NC}"
    exit 1
fi

# Copy required scripts
REQUIRED_SCRIPTS=(
    "scripts/vggt_preprocessing.py"
    "vggt_to_sugar.py"
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
if [ -f "${SCRIPT_DIR}/scripts/vggt_batch_processor.py" ]; then
    cp "${SCRIPT_DIR}/scripts/vggt_batch_processor.py" "${WORK_DIR}/"
    echo "✓ Copied vggt_batch_processor.py"
else
    echo -e "${YELLOW}⚠ vggt_batch_processor.py not found - VGGT will not be available${NC}"
    VGGT_AVAILABLE=0
fi

# ============================================
# PHASE 1: PREPROCESSING WITH VGGT
# ============================================

print_header "PHASE 1: Preprocessing (VGGT)"

PREPROCESSING_DIR="${WORK_DIR}/preprocessing"

echo "Running VGGT preprocessing..."
echo "  Input: ${INPUT_DIR}"
echo "  Output: ${PREPROCESSING_DIR}"

# Run VGGT preprocessing
echo -e "${GREEN}Using VGGT for preprocessing${NC}"
python vggt_preprocessing.py \
    "${INPUT_DIR}" \
    "${PREPROCESSING_DIR}" \
    --vggt-script "${WORK_DIR}/vggt_batch_processor.py" \
    --gpu 0 \
    --timeout 900

PREPROCESS_EXIT=$?

if [ $PREPROCESS_EXIT -ne 0 ]; then
    echo -e "${RED}❌ Preprocessing failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Preprocessing completed${NC}"

# Find VGGT output directory
VGGT_OUTPUT=$(find "${PREPROCESSING_DIR}" -type d -name "output" | head -1)
if [ -z "$VGGT_OUTPUT" ] || [ ! -d "$VGGT_OUTPUT" ]; then
    echo -e "${RED}ERROR: Could not find VGGT output directory${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found VGGT output at: ${VGGT_OUTPUT}${NC}"

# ============================================
# PHASE 2: CONVERT VGGT TO SUGAR FORMAT
# ============================================

print_header "PHASE 2: Converting VGGT to Sugar Format"

SUGAR_DATA="${WORK_DIR}/sugar_data"

echo "Converting VGGT output to Sugar format..."
echo "  VGGT output: ${VGGT_OUTPUT}"
echo "  Sugar data: ${SUGAR_DATA}"

python vggt_to_sugar.py \
    "${VGGT_OUTPUT}" \
    "${SUGAR_DATA}" \
    --copy-images \
    --copy-masks

CONVERT_EXIT=$?

if [ $CONVERT_EXIT -ne 0 ]; then
    echo -e "${RED}❌ Conversion to Sugar format failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Conversion completed${NC}"

# Verify Sugar data structure
if [ ! -f "${SUGAR_DATA}/sparse/0/cameras.bin" ] || [ ! -f "${SUGAR_DATA}/sparse/0/images.bin" ] || [ ! -f "${SUGAR_DATA}/sparse/0/points3D.bin" ]; then
    echo -e "${RED}ERROR: Sugar data structure is incomplete${NC}"
    exit 1
fi

# Check for masks
if [ -d "${SUGAR_DATA}/masks" ]; then
    SUGAR_MASK_COUNT=$(ls -1 "${SUGAR_DATA}/masks/"*.png 2>/dev/null | wc -l)
    echo "  Masks in Sugar data: ${SUGAR_MASK_COUNT}"
fi

# ============================================
# PHASE 3: SUGAR TRAINING
# ============================================

print_header "PHASE 3: Sugar Training (Gaussian Splatting + Mesh Extraction)"

OUTPUT_DIR="${WORK_DIR}/sugar_output"
mkdir -p "${OUTPUT_DIR}"

echo "Configuration:"
echo "  Data: ${SUGAR_DATA}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Sugar Path: ${SUGAR_PATH}"

# Determine training parameters based on GPU
if [[ "${GPU_NAME}" == *"B200"* ]]; then
    echo -e "${BLUE}Using B200-optimized training parameters:${NC}"
    echo "  - Iterations: 7000"
    echo "  - Densification: Conservative"
    echo "  - Mesh resolution: High"

    ITERATIONS=7000
    DENSIFY_GRAD_THRESHOLD=0.0002

elif [ ${GPU_MEM_GB} -ge 80 ]; then
    ITERATIONS=30000
    DENSIFY_GRAD_THRESHOLD=0.0002
elif [ ${GPU_MEM_GB} -ge 40 ]; then
    ITERATIONS=15000
    DENSIFY_GRAD_THRESHOLD=0.0002
else
    ITERATIONS=7000
    DENSIFY_GRAD_THRESHOLD=0.0003
fi

echo "  Iterations: ${ITERATIONS}"

# Clear GPU cache before training
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Clearing GPU cache before training..."
    python -c "import torch; torch.cuda.empty_cache(); print('  GPU cache cleared')"
fi

echo ""
echo "Starting Sugar training pipeline..."
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

# Step 3a: Gaussian Splatting Training
print_header "PHASE 3a: Gaussian Splatting Training"

GS_OUTPUT="${OUTPUT_DIR}/gaussian_splatting"

cd "${SUGAR_PATH}"

python train.py \
    -s "${SUGAR_DATA}" \
    -m "${GS_OUTPUT}" \
    --iterations ${ITERATIONS} \
    --densify_grad_threshold ${DENSIFY_GRAD_THRESHOLD} \
    --eval

GS_EXIT=$?

if [ $GS_EXIT -ne 0 ]; then
    echo -e "${RED}❌ Gaussian Splatting training failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Gaussian Splatting training completed${NC}"

# Step 3b: SuGaR Optimization
print_header "PHASE 3b: SuGaR Optimization"

SUGAR_CHECKPOINT="${GS_OUTPUT}/chkpnt30000.pth"
if [ ! -f "${SUGAR_CHECKPOINT}" ]; then
    # Find latest checkpoint
    SUGAR_CHECKPOINT=$(ls -t "${GS_OUTPUT}"/chkpnt*.pth | head -1)
fi

if [ -z "${SUGAR_CHECKPOINT}" ] || [ ! -f "${SUGAR_CHECKPOINT}" ]; then
    echo -e "${RED}ERROR: Could not find Gaussian Splatting checkpoint${NC}"
    exit 1
fi

echo "Using checkpoint: ${SUGAR_CHECKPOINT}"

SUGAR_MODEL="${OUTPUT_DIR}/sugar_model"

python train_coarse_sugar.py \
    -s "${SUGAR_DATA}" \
    -c "${SUGAR_CHECKPOINT}" \
    -r "density" \
    -o "${SUGAR_MODEL}"

SUGAR_EXIT=$?

if [ $SUGAR_EXIT -ne 0 ]; then
    echo -e "${RED}❌ SuGaR optimization failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ SuGaR optimization completed${NC}"

# Step 3c: Extract Mesh
print_header "PHASE 3c: Extracting Mesh"

MESH_OUTPUT="${OUTPUT_DIR}/mesh"
mkdir -p "${MESH_OUTPUT}"

python extract_mesh.py \
    -s "${SUGAR_DATA}" \
    -c "${SUGAR_CHECKPOINT}" \
    -m "${SUGAR_MODEL}" \
    -o "${MESH_OUTPUT}/sugar_mesh.ply" \
    --mesh_resolution 6

MESH_EXIT=$?

if [ $MESH_EXIT -ne 0 ]; then
    echo -e "${RED}❌ Mesh extraction failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Mesh extraction completed${NC}"

# ============================================
# RESULTS & SUMMARY
# ============================================

print_header "Pipeline Results"

if [ $MESH_EXIT -eq 0 ]; then
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
VGGT → Sugar Pipeline Summary
==============================
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
  VGGT output: ${VGGT_OUTPUT}
  Sugar data: ${SUGAR_DATA}
  Iterations: ${ITERATIONS}

Output:
  Directory: ${OUTPUT_DIR}
  Mesh: ${MESH_OUTPUT}/sugar_mesh.ply

Work Directory: ${WORK_DIR}
EOF

    echo ""
    echo "Summary saved to: ${WORK_DIR}/summary.txt"

    # Compress outputs
    if ls "${OUTPUT_DIR}/"*.ply &>/dev/null || ls "${MESH_OUTPUT}/"*.ply &>/dev/null; then
        tar -czf "${WORK_DIR}/meshes.tar.gz" -C "${OUTPUT_DIR}" . 2>/dev/null
        echo -e "${GREEN}✓ Outputs compressed to: ${WORK_DIR}/meshes.tar.gz${NC}"
    fi

else
    echo -e "${RED}❌ Pipeline Failed!${NC}"
    echo "Exit code: ${MESH_EXIT}"
fi

echo ""
echo "Work directory: ${WORK_DIR}"
echo "Total runtime: $((SECONDS / 60)) minutes"

# Download instructions
if [ $MESH_EXIT -eq 0 ] && [ -f "${WORK_DIR}/meshes.tar.gz" ]; then
    echo ""
    echo -e "${GREEN}To download results:${NC}"
    echo "  scp $(hostname):${WORK_DIR}/meshes.tar.gz ./"
fi

exit $MESH_EXIT
