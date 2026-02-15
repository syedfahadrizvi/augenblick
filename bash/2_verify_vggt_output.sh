#!/bin/bash
# FILE: verify_vggt_output.sh
# Verify the output of the VGGT preprocessing
# Usage: ./verify_vggt_output.sh <preprocessing_dir>
#
# This script verifies the output of the VGGT preprocessing by checking if the
# neuralangelo_data directory contains a transforms.json file and a masks directory.
#
# The preprocessing directory is the directory that contains the VGGT preprocessing output.
# The neuralangelo_data directory is the directory that contains the neuralangelo_data output.

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PREPROCESSING_DIR=$1

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