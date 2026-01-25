#!/usr/bin/env python3
"""
FILE: vggt_preprocessing.py
Fixed VGGT Preprocessing Script - Fixes glob pattern bug
With unified dataset organization and robust timeout handling
"""

import os
import sys
import json
import shutil
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional

# Add modules to path
sys.path.append(str(Path(__file__).parent))
from modules.dataset import DatasetOrganizer
from modules.config import PipelineConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_neuralangelo_data(base_dir: Path) -> Optional[Path]:
    """Recursively search for neuralangelo_data directory or transforms.json"""
    logger.info(f"Searching for neuralangelo_data in {base_dir}")
    
    # First check if transforms.json exists directly in base_dir
    if (base_dir / "transforms.json").exists():
        logger.info(f"Found transforms.json directly in: {base_dir}")
        return base_dir
    
    # Check common locations
    common_paths = [
        base_dir / "neuralangelo_data",
        base_dir / "neuralangelo",
        base_dir / "output" / "neuralangelo_data",
        base_dir / "output" / "neuralangelo",
        base_dir / "vggt" / "neuralangelo_data",
        base_dir / "output" / "vggt" / "neuralangelo_data",
    ]
    
    for path in common_paths:
        if path.exists() and (path / "transforms.json").exists():
            logger.info(f"Found neuralangelo_data at: {path}")
            return path
    
    # Use rglob to find ANY transforms.json
    try:
        for transforms_file in base_dir.rglob("transforms.json"):
            parent_dir = transforms_file.parent
            logger.info(f"Found transforms.json at: {parent_dir}")
            return parent_dir
    except Exception as e:
        logger.debug(f"Recursive search failed: {e}")
    
    # Last resort: find any directory with neuralangelo in the name
    try:
        for path in base_dir.rglob("*neuralangelo*"):
            if path.is_dir() and (path / "transforms.json").exists():
                logger.info(f"Found neuralangelo directory at: {path}")
                return path
    except Exception as e:
        logger.debug(f"Pattern search failed: {e}")
    
    logger.warning(f"No neuralangelo_data or transforms.json found under {base_dir}")
    return None


def organize_input_data(input_dir: Path, work_dir: Path) -> Tuple[Path, Path, int, int]:
    """Delegate to unified dataset organizer for consistency"""
    logger.info("Organizing input data using unified organizer...")
    
    config = PipelineConfig(input_dir=input_dir, output_dir=work_dir)
    organizer = DatasetOrganizer(config)
    
    if not organizer.organize():
        raise RuntimeError("Dataset organization failed")
    
    images_dir, masks_dir = organizer.get_paths()
    info = organizer.get_info()
    
    return images_dir, masks_dir, info['image_count'], info['mask_count']


def run_vggt_script(images_dir: Path, output_dir: Path, vggt_script: Path, 
                   timeout: int = 600) -> Optional[Path]:
    """Run VGGT script with configurable timeout and progress monitoring"""
    logger.info(f"Running VGGT preprocessing (timeout={timeout}s)...")
    
    # First check if VGGT is available
    try:
        import subprocess
        check_cmd = [sys.executable, "-c", "from vggt.models.vggt import VGGT; print('VGGT OK')"]
        result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            logger.error("VGGT module not found. Please install VGGT first.")
            logger.error(f"Error: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"Failed to check VGGT availability: {e}")
        return None
    
    vggt_output = output_dir / "vggt"
    vggt_output.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    existing_data = find_neuralangelo_data(vggt_output)
    if existing_data:
        logger.info("VGGT output already exists, using cached data")
        return existing_data
    
    # Run VGGT script directly on the images directory (not parent)
    cmd = [
        sys.executable,
        str(vggt_script),
        str(images_dir),  # Pass images directory directly, not parent
        "--output_dir", str(vggt_output)
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Output directory: {vggt_output}")
    
    try:
        # Run with progress monitoring and capture output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            text=True
        )
        
        # Collect output for debugging
        output_lines = []
        import time
        start_time = time.time()
        
        while True:
            # Check if process finished
            retcode = process.poll()
            if retcode is not None:
                # Read any remaining output
                remaining = process.stdout.read()
                if remaining:
                    output_lines.append(remaining)
                break
            
            # Check timeout
            if time.time() - start_time > timeout:
                logger.warning(f"VGGT timeout after {timeout}s")
                process.kill()
                process.wait()
                return None
            
            # Read available output (non-blocking would be better but this works)
            line = process.stdout.readline()
            if line:
                output_lines.append(line.strip())
                # Log important lines
                if any(keyword in line.lower() for keyword in ['error', 'warning', 'failed', 'success', 'complete', 'saved']):
                    logger.info(f"VGGT: {line.strip()}")
            
            time.sleep(0.1)
        
        # Log full output if failed
        if process.returncode != 0:
            logger.error(f"VGGT failed with code {process.returncode}")
            logger.error("VGGT output:")
            for line in output_lines[-50:]:  # Last 50 lines
                logger.error(f"  {line}")
            return None
        
        logger.info("VGGT completed successfully")
        
        # Debug: List what was actually created
        logger.info("VGGT output directory contents:")
        try:
            for item in vggt_output.rglob("*"):
                if item.is_file():
                    logger.info(f"  - {item.relative_to(vggt_output)}")
                    if item.name == "transforms.json":
                        logger.info(f"    Found transforms.json at: {item.parent}")
        except Exception as e:
            logger.warning(f"Could not list output directory: {e}")
        
        # Search for the generated neuralangelo_data
        neuralangelo_data = find_neuralangelo_data(vggt_output)
        
        if neuralangelo_data:
            # Verify depth maps
            depth_dir = neuralangelo_data / "depth_maps"
            if depth_dir.exists():
                depth_files = list(depth_dir.glob("*.npy"))
                logger.info(f"✓ Found {len(depth_files)} depth maps")
            else:
                logger.warning("No depth maps found - depth supervision won't be available")

            # Fix VGGT poses: convert w2c to c2w and fix orientation flips
            logger.info("Fixing VGGT pose convention and orientation flips...")
            transforms_file = neuralangelo_data / "transforms.json"
            if transforms_file.exists():
                try:
                    import subprocess
                    # First apply the pose convention fix
                    fix_script = Path(__file__).parent / "fix_vggt_transforms.py"
                    transforms_fixed = transforms_file.parent / "transforms_fixed.json"
                    cmd = [
                        sys.executable, str(fix_script),
                        str(transforms_file),
                        "--out", str(transforms_fixed)
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    if result.returncode == 0:
                        # Replace original with fixed version
                        import shutil
                        shutil.move(str(transforms_file), str(transforms_file.with_suffix('.json.backup')))
                        shutil.move(str(transforms_fixed), str(transforms_file))
                        logger.info("✓ VGGT poses fixed: converted w2c->c2w and removed orientation flips")
                    else:
                        logger.warning(f"Pose fixing failed: {result.stderr}")
                        logger.warning("Continuing with original VGGT poses")

                    # Then apply additional refinement using feature matching
                    logger.info("Refining VGGT poses using feature matching...")
                    feature_fix_script = Path(__file__).parent / "fix_vggt_poses.py"
                    if feature_fix_script.exists():
                        cmd = [
                            sys.executable, str(feature_fix_script),
                            str(transforms_file),
                            str(images_dir)
                        ]
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                        if result.returncode == 0:
                            logger.info("✓ VGGT poses additionally refined using feature matching")
                        else:
                            logger.warning(f"Feature-based pose refinement failed: {result.stderr}")
                            logger.warning("Continuing with convention-fixed poses")
                    else:
                        logger.info("Feature-based pose refinement script not found, skipping")

                except Exception as e:
                    logger.warning(f"Could not run pose fixing: {e}")
                    logger.warning("Continuing with original VGGT poses")

            # Add mask/depth consistency check
            images_check = neuralangelo_data / "images"
            masks_check = neuralangelo_data / "masks"
            if images_check.exists() and masks_check.exists():
                try:
                    mask_files = set(f.name for f in masks_check.glob("*.png"))
                    image_files = set(f.name for f in images_check.glob("*.png"))
                    if not mask_files.issubset(image_files):
                        logger.warning("Some masks don't have corresponding images")
                    else:
                        logger.info("✓ Masks and images are consistent")
                except Exception as e:
                    logger.warning(f"Could not verify mask/image consistency: {e}")

            return neuralangelo_data
        else:
            logger.error("VGGT didn't create expected neuralangelo_data structure")
            logger.error(f"Check output at: {vggt_output}")
            return None
            
    except Exception as e:
        logger.error(f"VGGT execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_colmap_direct(images_dir: Path, masks_dir: Path, output_dir: Path) -> Optional[Path]:
    """Run COLMAP directly without using module system"""
    logger.info("Running COLMAP (direct)...")
    
    colmap_dir = output_dir / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    
    database_path = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    
    # Check if COLMAP is available
    colmap_paths = [
        "colmap",  # System path
        "/usr/local/bin/colmap",
        "/opt/colmap/bin/colmap",
        str(Path.home() / "colmap/build/src/exe/colmap"),
    ]
    
    colmap_exe = None
    for path in colmap_paths:
        try:
            result = subprocess.run([path, "--help"], capture_output=True, timeout=1)
            if result.returncode == 0:
                colmap_exe = path
                logger.info(f"Found COLMAP at: {path}")
                break
        except:
            continue
    
    if not colmap_exe:
        logger.error("COLMAP not found. Please install COLMAP or add it to PATH")
        return None
    
    try:
        # Feature extraction
        cmd = [
            colmap_exe, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--ImageReader.camera_model", "SIMPLE_PINHOLE",
            "--ImageReader.single_camera", "1"
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info("✓ Feature extraction complete")
        
        # Matching
        cmd = [
            colmap_exe, "exhaustive_matcher",
            "--database_path", str(database_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info("✓ Feature matching complete")
        
        # Reconstruction
        cmd = [
            colmap_exe, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info("✓ Sparse reconstruction complete")
        
        # Convert to text format
        text_dir = colmap_dir / "text"
        text_dir.mkdir(exist_ok=True)
        
        # Find the reconstruction (usually in sparse/0)
        recon_dir = sparse_dir / "0"
        if not recon_dir.exists():
            # Try to find any numbered directory
            for d in sparse_dir.iterdir():
                if d.is_dir() and d.name.isdigit():
                    recon_dir = d
                    break
        
        if recon_dir.exists():
            cmd = [
                colmap_exe, "model_converter",
                "--input_path", str(recon_dir),
                "--output_path", str(text_dir),
                "--output_type", "TXT"
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info("✓ Converted to text format")
            
            # Convert to Neuralangelo format
            neuralangelo_data = output_dir / "neuralangelo_data"
            if convert_colmap_to_neuralangelo(text_dir, images_dir, masks_dir, neuralangelo_data):
                return neuralangelo_data
        else:
            logger.error("COLMAP reconstruction failed - no output found")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"COLMAP failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error running COLMAP: {e}")
    
    return None


def create_fallback_neuralangelo_data(images_dir: Path, masks_dir: Path, output_dir: Path) -> Path:
    """Create a minimal Neuralangelo dataset as last resort"""
    logger.warning("Creating fallback Neuralangelo data with circular camera poses")
    
    neuralangelo_data = output_dir / "neuralangelo_data"
    neuralangelo_data.mkdir(parents=True, exist_ok=True)
    
    # Create directories
    ngp_images = neuralangelo_data / "images"
    ngp_masks = neuralangelo_data / "masks"
    ngp_images.mkdir(exist_ok=True)
    ngp_masks.mkdir(exist_ok=True)
    
    # Copy images and masks
    image_files = sorted(images_dir.glob("*"))
    for i, img_file in enumerate(image_files):
        if img_file.suffix.upper() in ['.JPG', '.JPEG', '.PNG']:
            shutil.copy2(img_file, ngp_images / f"frame_{i:06d}.png")
            
            # Look for corresponding mask
            mask_path = masks_dir / f"{img_file.stem}.png"
            if mask_path.exists():
                shutil.copy2(mask_path, ngp_masks / f"frame_{i:06d}.png")
            else:
                # Create white mask
                from PIL import Image
                img = Image.open(img_file)
                mask = Image.new("L", img.size, 255)
                mask.save(ngp_masks / f"frame_{i:06d}.png")
    
    # Create transforms with circular camera arrangement
    num_images = len(list(ngp_images.glob("*.png")))
    transforms = {
        "camera_model": "OPENCV",
        "fl_x": 621.6,
        "fl_y": 621.6,
        "cx": 320.0,
        "cy": 240.0,
        "w": 640,
        "h": 480,
        "frames": []
    }
    
    import numpy as np
    for i in range(num_images):
        # Arrange cameras in a circle looking at origin
        angle = 2 * np.pi * i / num_images
        radius = 3.0
        height = 0.5
        
        # Camera position
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        y = height
        
        # Create transform matrix (camera to world)
        cam_pos = np.array([x, y, z])
        look_at = np.array([0, 0, 0])
        up = np.array([0, 1, 0])  # Fixed: Standard +Y up convention
        
        # Compute camera axes
        z_axis = (cam_pos - look_at)
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        # Build transformation matrix
        transform = np.eye(4)
        transform[:3, 0] = x_axis
        transform[:3, 1] = y_axis
        transform[:3, 2] = z_axis
        transform[:3, 3] = cam_pos
        
        frame = {
            "file_path": f"images/frame_{i:06d}.png",
            "transform_matrix": transform.tolist(),
            "mask_path": f"masks/frame_{i:06d}.png"
        }
        transforms["frames"].append(frame)
    
    with open(neuralangelo_data / "transforms.json", 'w') as f:
        json.dump(transforms, f, indent=2)
    
    logger.warning(f"Created fallback dataset with {num_images} frames in circular arrangement")
    logger.warning("Camera poses are synthetic - reconstruction quality may be limited")
    
    return neuralangelo_data


def convert_colmap_to_neuralangelo(colmap_text_dir: Path, images_dir: Path, masks_dir: Path, 
                                  output_dir: Path) -> bool:
    """Convert COLMAP output to Neuralangelo format (stub)"""
    logger.info("Converting COLMAP to Neuralangelo format...")
    # This would need proper COLMAP parsing implementation
    return False


def main():
    parser = argparse.ArgumentParser(description="VGGT preprocessing for Neuralangelo")
    parser.add_argument("input_dir", type=Path, help="Input directory with images/masks")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("--vggt-script", type=Path,
                      default=Path.home() / "augenblick/vggt_batch_processor.py",
                      help="Path to VGGT script")
    parser.add_argument("--use-colmap", action="store_true",
                      help="Use COLMAP instead of VGGT (legacy option)")
    parser.add_argument("--use-hybrid", action="store_true", default=True,
                      help="Use hybrid VGGT+COLMAP (default, most reliable)")
    parser.add_argument("--vggt-only", action="store_true",
                      help="Use VGGT only (not recommended)")
    parser.add_argument("--force-fallback", action="store_true",
                      help="Force use of fallback circular cameras")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--timeout", type=int, default=600,
                      help="Timeout for external processes in seconds")
    
    args = parser.parse_args()

    # Handle conflicting flags
    if args.vggt_only:
        args.use_hybrid = False
    if args.use_colmap:
        args.use_hybrid = False

    # Create work directory
    work_dir = args.output_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Organize data using unified organizer
    try:
        images_dir, masks_dir, image_count, mask_count = organize_input_data(args.input_dir, work_dir)
    except Exception as e:
        logger.error(f"Failed to organize dataset: {e}")
        return 1

    if image_count == 0:
        logger.error("No images found!")
        return 1

    # Step 2: Run preprocessing with chosen method
    neuralangelo_data = None

    if args.force_fallback:
        # Forced fallback (testing only)
        neuralangelo_data = create_fallback_neuralangelo_data(images_dir, masks_dir, work_dir)

    elif args.use_hybrid:
        # DEFAULT: Hybrid VGGT+COLMAP (most reliable)
        logger.info("Using HYBRID mode: VGGT depth + COLMAP poses")
        hybrid_script = Path(__file__).parent / "hybrid_vggt_colmap.py"

        if hybrid_script.exists():
            try:
                cmd = [
                    sys.executable,
                    str(hybrid_script),
                    str(images_dir),
                    str(work_dir),
                    "--vggt-script", str(args.vggt_script),
                    "--vggt-timeout", str(args.timeout),
                    "--colmap-timeout", str(args.timeout)
                ]

                logger.info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, capture_output=False)

                # Find the merged output
                neuralangelo_data = find_neuralangelo_data(work_dir)

            except subprocess.CalledProcessError as e:
                logger.error(f"Hybrid pipeline failed: {e}")
                logger.warning("Falling back to VGGT-only mode")
                neuralangelo_data = run_vggt_script(images_dir, work_dir, args.vggt_script, args.timeout)
        else:
            logger.warning(f"Hybrid script not found at {hybrid_script}")
            logger.warning("Falling back to VGGT-only mode")
            neuralangelo_data = run_vggt_script(images_dir, work_dir, args.vggt_script, args.timeout)

    elif args.vggt_only and args.vggt_script.exists():
        # VGGT only mode
        logger.info("Using VGGT-only mode (not recommended)")
        neuralangelo_data = run_vggt_script(images_dir, work_dir, args.vggt_script, args.timeout)

    elif args.use_colmap:
        # COLMAP only mode
        logger.info("Using COLMAP-only mode")
        neuralangelo_data = run_colmap_direct(images_dir, masks_dir, work_dir)

    # Fallback chain if primary method failed
    if not neuralangelo_data:
        logger.warning("Primary method failed, trying fallbacks...")

        # Try COLMAP if we haven't already
        if not args.use_colmap:
            logger.info("Trying COLMAP...")
            neuralangelo_data = run_colmap_direct(images_dir, masks_dir, work_dir)

        # Last resort: synthetic fallback
        if not neuralangelo_data:
            logger.warning("All methods failed, creating fallback dataset")
            neuralangelo_data = create_fallback_neuralangelo_data(images_dir, masks_dir, work_dir)
    
    if neuralangelo_data and neuralangelo_data.exists():
        logger.info("="*60)
        logger.info("✓ Preprocessing completed!")
        logger.info(f"Neuralangelo data created at: {neuralangelo_data}")
        logger.info("="*60)
        
        # Verify output
        transforms_file = neuralangelo_data / "transforms.json"
        if transforms_file.exists():
            with open(transforms_file, 'r') as f:
                transforms = json.load(f)
            logger.info(f"  - {len(transforms.get('frames', []))} frames")
            
            depth_dir = neuralangelo_data / "depth_maps"
            if depth_dir.exists():
                depth_count = len(list(depth_dir.glob("*.npy")))
                logger.info(f"  - {depth_count} depth maps")
            else:
                logger.info("  - No depth maps")
        
        return 0
    else:
        logger.error("Failed to create Neuralangelo data")
        return 1


if __name__ == "__main__":
    sys.exit(main())
