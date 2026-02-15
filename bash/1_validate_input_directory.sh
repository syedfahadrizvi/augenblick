#!/bin/bash

# FILE: validate_input_directory.sh
# Validate the input directory for the Augenblick pipeline
# Usage: ./validate_input_directory.sh <input_dir>
#
# This script checks if the input directory contains images and masks directories
# and validates the naming convention of the masks.
#
# The input directory is the directory that contains the images and masks directories.
# The images directory is the directory that contains the images.

# ============================================
# VALIDATE INPUT DATA STRUCTURE
# ============================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

INPUT_DIR=$1

# Check for images directory
if [ -d "${INPUT_DIR}/images" ]; then
    IMG_DIR="${INPUT_DIR}/images"
    IMG_COUNT=$(find "${IMG_DIR}" \( -type f -o -type l \) \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
    echo "✓ Found images directory: ${IMG_COUNT} images"
else
    IMG_DIR="${INPUT_DIR}"
    IMG_COUNT=$(find "${IMG_DIR}" \( -type f -o -type l \) \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
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