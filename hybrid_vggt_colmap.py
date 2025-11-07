#!/usr/bin/env python3
"""
Hybrid VGGT + COLMAP Pipeline
- Uses COLMAP for robust camera poses (most reliable)
- Uses VGGT for depth maps and masks (neural priors)
- Merges results into single transforms.json for Neuralangelo

This gives best of both worlds: robust poses + depth supervision
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Module loading is handled by the calling environment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_vggt(images_dir: Path, output_dir: Path, vggt_script: Path,
             timeout: int = 900) -> Optional[Path]:
    """
    Run VGGT to get depth maps and masks
    Returns path to VGGT output directory
    """
    logger.info("="*60)
    logger.info("STEP 1: Running VGGT for depth maps and masks")
    logger.info("="*60)

    vggt_output = output_dir / "vggt"
    vggt_output.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    if (vggt_output / "neuralangelo_data").exists():
        logger.info("VGGT output already exists, using cached data")
        return vggt_output / "neuralangelo_data"

    cmd = [
        sys.executable,
        str(vggt_script),
        str(images_dir),
        "--output_dir", str(vggt_output)
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout
        )

        if result.returncode != 0:
            logger.error(f"VGGT failed with code {result.returncode}")
            logger.error(result.stdout[-2000:])  # Last 2000 chars
            return None

        logger.info("✓ VGGT completed successfully")

        # Find neuralangelo_data directory
        vggt_data = vggt_output / "neuralangelo_data"
        if vggt_data.exists():
            return vggt_data

        # Search recursively
        for transforms_file in vggt_output.rglob("transforms.json"):
            return transforms_file.parent

        logger.error("VGGT output structure not found")
        return None

    except subprocess.TimeoutExpired:
        logger.error(f"VGGT timeout after {timeout}s")
        return None
    except Exception as e:
        logger.error(f"VGGT execution failed: {e}")
        return None


def find_colmap_executable() -> Optional[str]:
    """Find COLMAP executable in common locations"""
    candidates = [
        "colmap",
        "/usr/local/bin/colmap",
        "/opt/colmap/bin/colmap",
        str(Path.home() / "colmap/build/src/exe/colmap"),
    ]

    for path in candidates:
        try:
            result = subprocess.run(
                [path, "--help"],
                capture_output=True,
                timeout=2,
                check=False
            )
            if result.returncode == 0:
                logger.info(f"Found COLMAP at: {path}")
                return path
        except:
            continue

    return None


def run_colmap(images_dir: Path, output_dir: Path,
               timeout_per_step: int = 600) -> Optional[Path]:
    """
    Run COLMAP to get camera poses
    Returns path to COLMAP sparse reconstruction directory
    """
    logger.info("="*60)
    logger.info("STEP 2: Running COLMAP for camera poses")
    logger.info("="*60)

    colmap_exe = find_colmap_executable()
    if not colmap_exe:
        logger.error("COLMAP not found. Install COLMAP or add to PATH")
        return None

    colmap_dir = output_dir / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)

    database_path = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    if (sparse_dir / "cameras.txt").exists():
        logger.info("COLMAP reconstruction already exists, using cached data")
        return sparse_dir

    try:
        # Feature extraction
        logger.info("  Extracting features...")
        cmd = [
            colmap_exe, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--ImageReader.camera_model", "SIMPLE_PINHOLE",
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", "1"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=timeout_per_step, check=True)
        logger.info("  ✓ Feature extraction complete")

        # Matching
        logger.info("  Matching features...")
        cmd = [
            colmap_exe, "exhaustive_matcher",
            "--database_path", str(database_path),
            "--SiftMatching.use_gpu", "1"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=timeout_per_step, check=True)
        logger.info("  ✓ Feature matching complete")

        # Reconstruction
        logger.info("  Running bundle adjustment...")
        cmd = [
            colmap_exe, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir.parent)
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=timeout_per_step, check=True)
        logger.info("  ✓ Sparse reconstruction complete")

        # Convert to text
        logger.info("  Converting to text format...")
        text_dir = colmap_dir / "text"
        text_dir.mkdir(exist_ok=True)

        cmd = [
            colmap_exe, "model_converter",
            "--input_path", str(sparse_dir),
            "--output_path", str(text_dir),
            "--output_type", "TXT"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60, check=True)
        logger.info("  ✓ Converted to text format")

        return text_dir

    except subprocess.TimeoutExpired:
        logger.error("COLMAP timeout")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"COLMAP failed: {e}")
        if e.stderr:
            logger.error(e.stderr.decode())
        return None
    except Exception as e:
        logger.error(f"Unexpected COLMAP error: {e}")
        return None


def parse_colmap_cameras(cameras_file: Path) -> Dict:
    """Parse COLMAP cameras.txt"""
    cameras = {}
    with open(cameras_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]

            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras


def parse_colmap_images(images_file: Path) -> List[Dict]:
    """Parse COLMAP images.txt"""
    images = []
    with open(images_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    # COLMAP format: every 2 lines is one image
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        image_id = int(parts[0])
        qw, qx, qy, qz = [float(x) for x in parts[1:5]]
        tx, ty, tz = [float(x) for x in parts[5:8]]
        camera_id = int(parts[8])
        name = parts[9]

        images.append({
            'image_id': image_id,
            'qvec': [qw, qx, qy, qz],
            'tvec': [tx, ty, tz],
            'camera_id': camera_id,
            'name': name
        })

    return sorted(images, key=lambda x: x['name'])


def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix"""
    qvec = np.array(qvec)
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def colmap_to_transform_matrix(qvec, tvec) -> np.ndarray:
    """
    Convert COLMAP w2c pose to NeRF/Blender c2w convention
    COLMAP: w2c with (qw, qx, qy, qz, tx, ty, tz)
    Output: c2w in NeRF convention (OpenGL: +Y up, -Z forward)
    """
    # COLMAP rotation (w2c)
    R_w2c = qvec2rotmat(qvec)
    t_w2c = np.array(tvec)

    # Invert to get c2w
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c

    # Convert from COLMAP convention to NeRF convention
    # COLMAP: +Y down, +Z forward → NeRF: +Y up, -Z forward
    # Flip Y and Z axes
    transform = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ], dtype=np.float64)

    R_c2w = R_c2w @ transform
    t_c2w = transform @ t_c2w

    # Build 4x4 matrix
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = t_c2w

    return c2w


def merge_vggt_colmap(vggt_data_dir: Path, colmap_text_dir: Path,
                      output_dir: Path) -> Path:
    """
    Merge VGGT depth/masks with COLMAP poses
    Returns path to merged neuralangelo_data directory
    """
    logger.info("="*60)
    logger.info("STEP 3: Merging VGGT depth/masks with COLMAP poses")
    logger.info("="*60)

    # Load VGGT transforms.json
    vggt_transforms_file = vggt_data_dir / "transforms.json"
    with open(vggt_transforms_file, 'r') as f:
        vggt_data = json.load(f)

    # Parse COLMAP outputs
    cameras = parse_colmap_cameras(colmap_text_dir / "cameras.txt")
    colmap_images = parse_colmap_images(colmap_text_dir / "images.txt")

    logger.info(f"  VGGT frames: {len(vggt_data['frames'])}")
    logger.info(f"  COLMAP images: {len(colmap_images)}")

    # Create output directory
    merged_dir = output_dir / "neuralangelo_data"
    merged_dir.mkdir(parents=True, exist_ok=True)

    # Copy VGGT depth maps and masks
    for subdir in ['images', 'images_no_scale', 'masks', 'depth_maps']:
        src = vggt_data_dir / subdir
        if src.exists():
            dst = merged_dir / subdir
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            logger.info(f"  ✓ Copied {subdir}")

    # Match COLMAP images to VGGT frames by filename
    colmap_by_name = {img['name']: img for img in colmap_images}

    # Get camera parameters from COLMAP (use first camera)
    camera = list(cameras.values())[0]
    if camera['model'] == 'SIMPLE_PINHOLE':
        f, cx, cy = camera['params']
        fl_x = fl_y = f
    elif camera['model'] == 'PINHOLE':
        fx, fy, cx, cy = camera['params']
        fl_x, fl_y = fx, fy
    else:
        logger.warning(f"Unknown camera model: {camera['model']}, using VGGT intrinsics")
        fl_x = vggt_data.get('fl_x', 621.6)
        fl_y = vggt_data.get('fl_y', 621.6)
        cx = vggt_data.get('cx', camera['width'] / 2)
        cy = vggt_data.get('cy', camera['height'] / 2)

    # Build merged transforms.json
    merged_transforms = {
        "camera_model": "PINHOLE",
        "w": camera['width'],
        "h": camera['height'],
        "fl_x": float(fl_x),
        "fl_y": float(fl_y),
        "cx": float(cx),
        "cy": float(cy),
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "sk_x": 0.0,
        "sk_y": 0.0,
        "aabb_scale": 2,
        "frames": []
    }

    matched = 0
    unmatched = 0

    for vggt_frame in vggt_data['frames']:
        file_path = vggt_frame['file_path']
        filename = Path(file_path).name

        # Try to find matching COLMAP image
        colmap_img = None

        # Direct match
        if filename in colmap_by_name:
            colmap_img = colmap_by_name[filename]
        else:
            # Try without extension
            stem = Path(filename).stem
            for name, img in colmap_by_name.items():
                if Path(name).stem == stem:
                    colmap_img = img
                    break

        if colmap_img:
            # Use COLMAP pose
            c2w = colmap_to_transform_matrix(colmap_img['qvec'], colmap_img['tvec'])
            matched += 1
        else:
            # Fall back to VGGT pose (shouldn't happen often)
            c2w = np.array(vggt_frame['transform_matrix'])
            unmatched += 1
            logger.warning(f"  No COLMAP match for {filename}, using VGGT pose")

        frame = {
            "file_path": file_path,
            "transform_matrix": c2w.tolist()
        }

        # Add mask path if exists
        if 'mask_path' in vggt_frame:
            frame['mask_path'] = vggt_frame['mask_path']

        # Add depth path if exists
        if 'depth_path' in vggt_frame:
            frame['depth_path'] = vggt_frame['depth_path']
            if 'depth_confidence_path' in vggt_frame:
                frame['depth_confidence_path'] = vggt_frame['depth_confidence_path']

        merged_transforms['frames'].append(frame)

    logger.info(f"  Matched {matched}/{len(vggt_data['frames'])} frames to COLMAP poses")
    if unmatched > 0:
        logger.warning(f"  {unmatched} frames fell back to VGGT poses")

    # Compute sphere parameters from COLMAP poses
    camera_positions = []
    for frame in merged_transforms['frames']:
        T = np.array(frame['transform_matrix'])
        camera_positions.append(T[:3, 3])

    camera_positions = np.array(camera_positions)
    center = camera_positions.mean(axis=0)

    # Use origin as center (better for object-centric scenes)
    center = np.array([0.0, 0.0, 0.0])
    distances = np.linalg.norm(camera_positions - center, axis=1)
    radius = distances.max() * 1.3  # 30% padding

    merged_transforms['sphere_center'] = center.tolist()
    merged_transforms['sphere_radius'] = float(radius)
    merged_transforms['sphere_scale'] = float(1.0 / radius)

    logger.info(f"  Sphere center: {center.tolist()}")
    logger.info(f"  Sphere radius: {radius:.4f}")
    logger.info(f"  Sphere scale: {1.0/radius:.4f}")

    # Save merged transforms
    merged_transforms_file = merged_dir / "transforms.json"
    with open(merged_transforms_file, 'w') as f:
        json.dump(merged_transforms, f, indent=2)

    logger.info(f"✓ Merged transforms saved to {merged_transforms_file}")

    return merged_dir


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid VGGT+COLMAP pipeline for Neuralangelo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full hybrid pipeline
  python hybrid_vggt_colmap.py /path/to/images /path/to/output

  # Specify custom VGGT script location
  python hybrid_vggt_colmap.py /path/to/images /path/to/output --vggt-script ~/vggt_batch_processor.py
        """
    )
    parser.add_argument("images_dir", type=Path,
                       help="Input directory with images")
    parser.add_argument("output_dir", type=Path,
                       help="Output directory")
    parser.add_argument("--vggt-script", type=Path,
                       default=Path(__file__).parent / "vggt_batch_processor.py",
                       help="Path to VGGT batch processor script")
    parser.add_argument("--vggt-timeout", type=int, default=900,
                       help="VGGT timeout in seconds")
    parser.add_argument("--colmap-timeout", type=int, default=600,
                       help="COLMAP timeout per step in seconds")
    parser.add_argument("--skip-vggt", action="store_true",
                       help="Skip VGGT (use existing output)")
    parser.add_argument("--skip-colmap", action="store_true",
                       help="Skip COLMAP (use existing output)")

    args = parser.parse_args()

    # Validate inputs
    if not args.images_dir.exists():
        logger.error(f"Images directory not found: {args.images_dir}")
        return 1

    if not args.vggt_script.exists():
        logger.error(f"VGGT script not found: {args.vggt_script}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("HYBRID VGGT + COLMAP PIPELINE")
    logger.info("="*60)
    logger.info(f"Input: {args.images_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("")
    logger.info("Strategy:")
    logger.info("  - VGGT: Depth maps + masks (neural priors)")
    logger.info("  - COLMAP: Camera poses (feature matching)")
    logger.info("  - Result: Best of both worlds!")
    logger.info("")

    # Step 1: Run VGGT
    vggt_data_dir = None
    if not args.skip_vggt:
        vggt_data_dir = run_vggt(
            args.images_dir,
            args.output_dir,
            args.vggt_script,
            args.vggt_timeout
        )
        if not vggt_data_dir:
            logger.error("VGGT failed")
            return 1
    else:
        # Look for existing VGGT output
        vggt_data_dir = args.output_dir / "vggt" / "neuralangelo_data"
        if not vggt_data_dir.exists():
            logger.error(f"VGGT output not found at {vggt_data_dir}")
            return 1
        logger.info(f"Using existing VGGT output: {vggt_data_dir}")

    # Step 2: Run COLMAP
    colmap_text_dir = None
    if not args.skip_colmap:
        colmap_text_dir = run_colmap(
            args.images_dir,
            args.output_dir,
            args.colmap_timeout
        )
        if not colmap_text_dir:
            logger.error("COLMAP failed")
            return 1
    else:
        # Look for existing COLMAP output
        colmap_text_dir = args.output_dir / "colmap" / "text"
        if not colmap_text_dir.exists():
            logger.error(f"COLMAP output not found at {colmap_text_dir}")
            return 1
        logger.info(f"Using existing COLMAP output: {colmap_text_dir}")

    # Step 3: Merge
    merged_dir = merge_vggt_colmap(vggt_data_dir, colmap_text_dir, args.output_dir)

    logger.info("")
    logger.info("="*60)
    logger.info("✅ HYBRID PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"Output: {merged_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Run sanity checks: python sanity_check_poses.py " + str(merged_dir))
    logger.info("  2. Start training with this dataset")

    return 0


if __name__ == "__main__":
    sys.exit(main())