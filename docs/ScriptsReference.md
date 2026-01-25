# Scripts Reference

This document provides a comprehensive overview of the various VGGT → Neuralangelo pipeline scripts in the Augenblick project.

---

## Table of Contents

1. [Overview](#overview)
2. [Script Summaries](#script-summaries)
3. [Detailed Descriptions](#detailed-descriptions)
4. [Comparison Matrix](#comparison-matrix)
5. [When to Use Which Script](#when-to-use-which-script)

---

## Overview

The Augenblick project contains multiple pipeline scripts that perform variations of the same core workflow:

```
Input Images → VGGT Processing → Camera Pose Estimation → Neuralangelo Training → 3D Mesh
```

These scripts evolved over time with different features, platforms, and levels of complexity.

---

## Script Summaries

| Script | Type | Primary Purpose | Complexity |
|--------|------|-----------------|------------|
| `exp_scripts/run_pipeline.sh` | Bash | Full HPC pipeline with B200 optimizations | High |
| `exp_scripts/run_pipeline (4).sh` | Bash | Identical copy of run_pipeline.sh | High |
| `run_neuo_pipeline.py` | Python | Complete VGG-T → Neuralangelo pipeline | Medium |
| `vggt_to_neuralangelo.py` | Python | VGGT processing only (HuggingFace model) | Medium |
| `launch_neuralangelo.sh` | Bash | Full pipeline with cropping and staging | High |
| `run_vgggt_neuor.py` | Python | Simplified pipeline with Singularity | Low |
| `ggt-neuro.py` | Python | Georgia Tech HPC integrated pipeline | High |
| `ggt-neuro-fixed.py` | Python | Bug-fixed version of ggt-neuro.py | High |

---

## Detailed Descriptions

### 1. `exp_scripts/run_pipeline.sh`

**Purpose**: Complete end-to-end Neuralangelo pipeline optimized for HPC clusters with GPU detection and multi-stage training.

**What it does**:
- Loads HPC modules (pytorch/2.7, cuda/12.8.1)
- Activates a Python virtual environment
- Validates input data (images and masks)
- Copies required scripts to a working directory
- Runs VGGT preprocessing via `vggt_preprocessing.py`
- Computes sphere parameters for scene normalization
- Executes multi-stage training with `stage_training_manager.py`
- Applies GPU-specific optimizations (B200, H100, H200, A100)
- Creates summary reports and compresses outputs

**Arguments**:
```bash
./run_pipeline.sh [input_dir] [work_dir]
```

| Argument | Position | Required | Default | Description |
|----------|----------|----------|---------|-------------|
| `input_dir` | 1st | No | `$SCRATCH_BASE/scale_organized` | Directory containing images/masks |
| `work_dir` | 2nd | No | `$SCRATCH_BASE/neura_TIMESTAMP` | Working directory for outputs |

**Environment Variables**:
| Variable | Default | Description |
|----------|---------|-------------|
| `SCRATCH_BASE` | `/blue/arthur.porto-biocosmos/jhennessy7.gatech/scratch` | Base scratch directory |
| `SCRIPT_DIR` | `/home/jhennessy7.gatech/augenblick` | Source scripts location |
| `NEURALANGELO_PATH` | `$SCRIPT_DIR/src/neuralangelo` | Neuralangelo installation |
| `ENV_PATH` | `/home/jhennessy7.gatech/neuralangelo_b200_env` | Python virtual environment |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device to use |

---

### 2. `exp_scripts/run_pipeline (4).sh`

**Purpose**: This is an **identical copy** of `run_pipeline.sh`.

**Difference**: None - the files are exactly the same (575 lines each). This appears to be a versioned backup.

---

### 3. `run_neuo_pipeline.py`

**Purpose**: Python-based complete 3D reconstruction pipeline from VGG-T feature extraction to Neuralangelo mesh generation.

**What it does**:
1. **Step 1**: Runs VGG-T feature extraction via `src/pipeline/run_pipeline.py`
2. **Step 2**: Converts VGG-T predictions (pickle format) to Neuralangelo format
   - Creates `transforms.json` with camera poses
   - Copies images to Neuralangelo directory structure
3. **Step 3**: Checks/prepares Neuralangelo data format (skips COLMAP if poses exist)
4. **Step 4**: Runs Neuralangelo neural surface reconstruction
   - Creates custom config from template
   - Extracts mesh from checkpoint

**Arguments**:
```bash
python run_neuo_pipeline.py input_dir [options]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `input_dir` | Yes | - | Directory containing images |
| `--output_dir` | No | `input_dir` | Output directory for results |
| `--neuralangelo_steps` | No | `20000` | Number of training iterations |
| `--skip_neuralangelo` | No | `False` | Skip Neuralangelo (only run VGG-T) |
| `--config` | No | Auto-generated | Path to custom Neuralangelo config |
| `--force-rerun` | No | `False` | Force re-run of all steps |

**Output Structure**:
```
output_dir/
├── vggt_output/           # VGG-T predictions (.pkl)
├── neuralangelo_data/     # Converted data
│   ├── images/
│   └── transforms.json
├── neuralangelo_preprocessed/
└── neuralangelo_output/   # Training results and mesh
```

---

### 4. `vggt_to_neuralangelo.py`

**Purpose**: Dedicated VGGT processing script that uses the pretrained HuggingFace model (`facebook/VGGT-1B`) to generate camera poses and depth maps.

**What it does**:
- Loads VGGT-1B model from HuggingFace
- Processes images in batches
- Extracts from VGGT outputs:
  - Depth maps
  - Point maps / world points
  - Camera poses from `pose_enc` (6D rotation + 3D translation)
- Converts 6D rotation representation to 3x3 rotation matrices
- Saves processed images, depth maps, and camera parameters
- Creates `transforms.json` and Neuralangelo config YAML

**Arguments**:
```bash
python vggt_to_neuralangelo.py images_dir --output_dir output [options]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `images` | Yes | - | Input images directory |
| `--output_dir` | Yes | - | Output directory |
| `--stride` | No | `1` | Use every Nth image |
| `--batch_size` | No | `8` | Batch size for VGGT processing |
| `--device` | No | `cuda` | Device (cuda/cpu) |
| `--metadata` | No | - | Path to metadata.json from prep_crop.py |

**Output Structure**:
```
output_dir/
├── images/                # Processed images (frame_XXXXXX.png)
├── masks/                 # Resized masks (if available)
├── depth/                 # Depth maps (.png and .npy)
├── transforms.json        # Camera poses and intrinsics
├── vggt_metadata.json     # Processing metadata
└── neuralangelo_config.yaml
```

---

### 5. `launch_neuralangelo.sh`

**Purpose**: Complete pipeline launcher that can either process new datasets through VGGT or use existing processed data.

**What it does**:
1. **VGGT Pipeline** (if `--images` provided):
   - Crops images using `prep_crop.py`
   - Runs VGGT for camera pose estimation via `vggt_batch_final.py`
   - Scales transforms back to original resolution
2. **Setup Working Directory**:
   - Copies training scripts (`staged_train.py`, stage configs)
   - Creates symlinks to image/mask directories
3. **Config Updates**:
   - Updates YAML configs with correct paths
4. **Training Launch**:
   - Can auto-start training or prompt user

**Arguments**:
```bash
./launch_neuralangelo.sh [options]
```

**For new datasets**:
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dataset NAME` | Yes* | - | Dataset name |
| `--images DIR` | Yes* | - | Raw images directory |
| `--masks DIR` | Yes* | - | Masks directory |
| `--stride N` | No | `2` | Sample every Nth frame |
| `--cameras-per-group N` | No | `46` | Camera group size |

**For existing data**:
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--use-existing DIR` | - | - | Use existing neuralangelo directory |
| `--resume-from DIR` | - | - | Resume from existing directory |

**Training Stages**:
- Stage 1: 512×768 (0-2k iterations)
- Stage 2: 2080×3120 (2k-10k iterations)
- Stage 3: 4160×6240 (10k-20k iterations)

---

### 6. `run_vgggt_neuor.py`

**Purpose**: Simplified/minimal VGG-T → Neuralangelo pipeline designed for use with Singularity containers.

**What it does**:
1. Runs VGG-T feature extraction
2. Converts predictions to Neuralangelo format (basic conversion)
3. Optionally runs Neuralangelo via Singularity container

**Arguments**:
```bash
python run_vgggt_neuor.py input_dir [options]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `input_dir` | Yes | - | Directory containing images |
| `--output_dir` | No | `input_dir` | Output directory |
| `--steps` | No | `300000` | Neuralangelo training steps |
| `--skip_reconstruction` | No | `False` | Skip Neuralangelo reconstruction |

**Note**: This script prints the Singularity command but doesn't execute it by default. It's meant as a template/reference.

---

### 7. `ggt-neuro.py`

**Purpose**: Integrated VGGT → Neuralangelo pipeline specifically designed for Georgia Tech HPC environment.

**What it does**:
- Uses a class-based design (`IntegratedVGGTPipeline`)
- Searches for VGGT checkpoint in multiple common locations
- Manually preprocesses images (resize to 518px max, normalize)
- Runs VGGT inference
- Creates complete Neuralangelo project structure
- Generates SLURM job scripts for HPC submission
- Uses Singularity containers for COLMAP and Neuralangelo

**Arguments**:
```bash
python ggt-neuro.py input_dir [options]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `input_dir` | Yes | - | Directory containing input images |
| `--output_name` | No | Auto-generated | Name for output directory |
| `--vggt_checkpoint` | No | Auto-detect | Path to VGGT checkpoint |
| `--test` | No | `False` | Run on test images |

**Checkpoint Search Paths**:
```
~/augenblick/checkpoints/vggt_large.pth
~/augenblick/checkpoints/vggt.pth
~/augenblick/models/vggt_large.pth
~/augenblick/src/vggt/vggt_large.pth
~/augenblick/src/vggt/checkpoints/vggt_large.pth
~/vggt_large.pth
```

---

### 8. `ggt-neuro-fixed.py`

**Purpose**: Bug-fixed and improved version of `ggt-neuro.py` with better error handling and fallbacks.

**Key Fixes/Improvements over ggt-neuro.py**:
1. **Additional checkpoint path**: Added `~/scratch/vggt_large.pth`
2. **Flexible model loading**: Multiple fallback strategies for VGGT initialization parameters
3. **Dimension handling**: Ensures image dimensions are multiples of patch_size (14)
4. **Inference fallback**: Returns dummy outputs if inference fails (allows pipeline to continue)
5. **Quaternion normalization**: Added normalization before rotation matrix conversion
6. **Better error messages**: More detailed troubleshooting info

**Arguments**: Same as `ggt-neuro.py`

---

## Comparison Matrix

| Feature | run_pipeline.sh | run_neuo_pipeline.py | vggt_to_neuralangelo.py | launch_neuralangelo.sh | run_vgggt_neuor.py | ggt-neuro.py |
|---------|-----------------|---------------------|------------------------|----------------------|-------------------|--------------|
| **Language** | Bash | Python | Python | Bash | Python | Python |
| **VGGT Model** | External script | src/pipeline | HuggingFace | External script | src/pipeline | Local checkpoint |
| **Neuralangelo Training** | Yes | Yes | No | Yes | Template only | SLURM script |
| **Multi-stage Training** | Yes | No | N/A | Yes | No | No |
| **GPU Optimization** | B200/H100/A100 | No | Auto dtype | No | No | No |
| **Mask Support** | Yes | Limited | Yes | Yes | No | No |
| **HPC Integration** | Modules | No | No | Modules | Singularity | SLURM |
| **Image Cropping** | No | No | No | Yes | No | No |
| **Complexity** | High | Medium | Medium | High | Low | High |

---

## When to Use Which Script

### Use `exp_scripts/run_pipeline.sh` when:
- Running on an HPC cluster with module system
- You have a B200, H100, H200, or A100 GPU
- You need multi-stage training with different resolutions
- You want automatic GPU optimizations

### Use `run_neuo_pipeline.py` when:
- You want a Python-only solution
- You need to integrate into other Python code
- You want fine-grained control over each step
- You're okay with simpler training (no multi-stage)

### Use `vggt_to_neuralangelo.py` when:
- You only need VGGT processing (no training)
- You want to use the HuggingFace pretrained model
- You need depth maps in addition to camera poses
- You plan to run Neuralangelo separately

### Use `launch_neuralangelo.sh` when:
- You need to crop images before processing
- You want to process new datasets from scratch
- You need multi-stage training with different resolutions
- You want interactive prompts and flexibility

### Use `run_vgggt_neuor.py` when:
- You need a simple reference implementation
- You're using Singularity containers
- You want to understand the basic pipeline flow

### Use `ggt-neuro-fixed.py` when:
- Running on Georgia Tech HPC (or similar)
- You need robust error handling and fallbacks
- You want to generate SLURM job scripts
- Your VGGT checkpoint might be in various locations

---

## Quick Start Examples

### Processing a new dataset with full pipeline:
```bash
# Using shell script
./exp_scripts/run_pipeline.sh /path/to/images /path/to/output

# Using Python
python run_neuo_pipeline.py /path/to/images --output_dir /path/to/output
```

### VGGT processing only:
```bash
python vggt_to_neuralangelo.py /path/to/images --output_dir /path/to/vggt_output
```

### Using existing processed data:
```bash
./launch_neuralangelo.sh --use-existing /path/to/existing/data
```

### With image cropping and staging:
```bash
./launch_neuralangelo.sh \
    --dataset my_skull \
    --images /path/to/raw/images \
    --masks /path/to/masks \
    --stride 2
```
