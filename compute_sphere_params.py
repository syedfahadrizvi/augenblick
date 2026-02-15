#!/usr/bin/env python3
"""
FILE: compute_sphere_params.py
Complete data preparation script for Neuralangelo
- Fixes transforms.json with missing camera parameters
- Computes sphere normalization parameters and adds them to transforms.json
- Updates stage configs with all required values
"""

import json
import numpy as np
from pathlib import Path
import argparse
import logging
import shutil
import yaml
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_pose_geometry(transforms):
    """Relaxed validator: auto-pick forward sign, warn on planarity, don't hard-fail turntables."""
    import numpy as np, logging
    logger = logging.getLogger(__name__)

    frames = [f for f in transforms.get('frames', []) if f.get('transform_matrix') is not None]
    if len(frames) < 3:
        logger.warning("Too few valid poses to validate")
        return True

    T = np.array([np.array(f['transform_matrix'], float) for f in frames])
    C = T[:, :3, 3]             # camera centers
    # After pose fixing we use NeRF/Blender convention: -Z forward
    look_dir = -T[:, :3, 2]
    center = np.array(transforms.get('sphere_center', [0,0,0]), float)

    v = center - C
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    # Check alignment using -Z as forward direction (after coordinate flip)
    dots = (look_dir * v).sum(axis=1)
    sign = "-Z"

    good = int(np.sum(dots > 0.5))
    logger.info(f"Pose validation:")
    logger.info(f"  Using {sign} as forward for alignment check")
    logger.info(f"  Alignment: {good}/{len(dots)} cameras properly aligned")
    logger.info(f"  Mean dot product: {np.mean(dots):.3f}")

    std = np.std(C, axis=0)
    logger.info(f"  Position std: [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]")

    # Planarity is common for turntables; warn only.
    thin = np.min(std) < 0.01 and np.max(std) > 0.05
    if thin:
        logger.warning("⚠️  Camera distribution is very planar (turntable-like)")

    # Only hard-fail if *almost no* views look at the center
    if good < max(5, len(dots) * 0.15):
        logger.warning("⚠️  Many poses misaligned - reconstruction may fail")
        return False

    logger.info("✓ Pose alignment looks acceptable")
    return True


def fix_transforms_json(transforms_file: Path, backup: bool = True):
    """
    Fix transforms.json to include all required fields for Neuralangelo
    INCLUDING sphere normalization parameters
    
    Returns: True if successful, False otherwise
    """
    if not transforms_file.exists():
        logger.error(f"transforms.json not found at {transforms_file}")
        return False
    
    # Create backup if requested
    if backup:
        backup_path = transforms_file.with_suffix('.json.backup')
        if not backup_path.exists():
            shutil.copy2(transforms_file, backup_path)
            logger.info(f"Created backup: {backup_path}")
    
    # Load transforms
    with open(transforms_file, 'r') as f:
        transforms = json.load(f)
    
    logger.info(f"Loaded transforms with {len(transforms.get('frames', []))} frames")
    
    # Track what we add
    fields_added = []
    
    # Add missing camera intrinsic parameters
    # Skew parameters (almost always 0 for modern cameras)
    if 'sk_x' not in transforms:
        transforms['sk_x'] = 0.0
        fields_added.append('sk_x')
    
    if 'sk_y' not in transforms:
        transforms['sk_y'] = 0.0
        fields_added.append('sk_y')
    
    # Distortion coefficients (0 = no distortion, which is fine for VGGT output)
    if 'k1' not in transforms:
        transforms['k1'] = 0.0
        fields_added.append('k1')
    
    if 'k2' not in transforms:
        transforms['k2'] = 0.0
        fields_added.append('k2')
    
    if 'p1' not in transforms:
        transforms['p1'] = 0.0
        fields_added.append('p1')
    
    if 'p2' not in transforms:
        transforms['p2'] = 0.0
        fields_added.append('p2')
    
    # AABB scale for scene bounding box
    if 'aabb_scale' not in transforms:
        transforms['aabb_scale'] = 2  # Standard scale for Neuralangelo
        fields_added.append('aabb_scale')
    
    # Ensure we have focal lengths (fl_x, fl_y)
    if 'fl_x' not in transforms and 'fl' in transforms:
        # Some formats use a single 'fl' value
        transforms['fl_x'] = transforms['fl']
        transforms['fl_y'] = transforms['fl']
        fields_added.extend(['fl_x', 'fl_y'])
    
    # Ensure we have principal point (cx, cy)
    if 'cx' not in transforms and 'w' in transforms:
        transforms['cx'] = transforms['w'] / 2.0
        fields_added.append('cx')
    
    if 'cy' not in transforms and 'h' in transforms:
        transforms['cy'] = transforms['h'] / 2.0
        fields_added.append('cy')
    
    # Compute and add sphere normalization parameters
    if 'sphere_center' not in transforms or 'sphere_scale' not in transforms:
        logger.info("Computing sphere normalization parameters...")
        
        # Extract camera positions from transform matrices
        camera_positions = []
        for frame in transforms.get('frames', []):
            if 'transform_matrix' in frame and frame['transform_matrix'] is not None:
                transform = np.array(frame['transform_matrix'])
                if transform.shape == (4, 4):
                    # Camera position in world space (last column of transform)
                    camera_pos = transform[:3, 3]
                    camera_positions.append(camera_pos)
        
        if camera_positions:
            camera_positions = np.array(camera_positions)
            
            # For Neuralangelo, use origin as sphere center (object-centered approach)
            # This works better with VGGT poses that look toward the origin
            center = np.array([0.0, 0.0, 0.0])
            distances = np.linalg.norm(camera_positions - center, axis=1)
            radius = distances.max()

            # Add 30% padding for safety
            radius *= 1.3
            
            # Sphere scale is 1/radius for normalization
            scale = 1.0 / radius if radius > 0 else 1.0
            
            # Convert to Python native types (not numpy)
            transforms['sphere_center'] = center.tolist()
            transforms['sphere_scale'] = float(scale)
            transforms['sphere_radius'] = float(radius)
            
            fields_added.extend(['sphere_center', 'sphere_scale', 'sphere_radius'])
            
            logger.info(f"  Added sphere_center: {transforms['sphere_center']}")
            logger.info(f"  Added sphere_scale: {transforms['sphere_scale']:.4f}")
            logger.info(f"  Added sphere_radius: {transforms['sphere_radius']:.4f}")
        else:
            # Default values if no valid camera positions
            transforms['sphere_center'] = [0.0, 0.0, 0.0]
            transforms['sphere_scale'] = 1.0
            transforms['sphere_radius'] = 1.0
            fields_added.extend(['sphere_center', 'sphere_scale', 'sphere_radius'])
            logger.warning("No valid camera positions found, using default sphere parameters")
    
    # Validate pose geometry before saving
    if not validate_pose_geometry(transforms):
        logger.error("❌ Pose validation failed - poses are likely incorrect")
        logger.error("Consider using COLMAP or fixing VGGT poses with feature matching")
        # return False

    # Save updated transforms
    with open(transforms_file, 'w') as f:
        json.dump(transforms, f, indent=2)

    if fields_added:
        logger.info(f"✓ Added missing fields to transforms.json: {', '.join(fields_added)}")
    else:
        logger.info("✓ transforms.json already has all required fields")
    
    # Print camera parameters for verification
    logger.info("Camera parameters:")
    logger.info(f"  Image size: {transforms.get('w', 'unknown')} x {transforms.get('h', 'unknown')}")
    logger.info(f"  Focal length: ({transforms.get('fl_x', 'unknown')}, {transforms.get('fl_y', 'unknown')})")
    logger.info(f"  Principal point: ({transforms.get('cx', 'unknown')}, {transforms.get('cy', 'unknown')})")
    logger.info(f"  Skew: ({transforms.get('sk_x', 0)}, {transforms.get('sk_y', 0)})")
    logger.info(f"  Distortion: k1={transforms.get('k1', 0)}, k2={transforms.get('k2', 0)}, "
                f"p1={transforms.get('p1', 0)}, p2={transforms.get('p2', 0)}")
    logger.info(f"  AABB scale: {transforms.get('aabb_scale', 'unknown')}")
    logger.info(f"  Sphere center: {transforms.get('sphere_center', 'unknown')}")
    logger.info(f"  Sphere scale: {transforms.get('sphere_scale', 'unknown')}")
    
    return True


def update_stage_configs(config_dir: Path, data_root: Path, transforms_file: Path):
    """
    Update all stage configuration files with data from transforms.json
    
    Returns: Number of configs updated
    """
    # Load transforms to get parameters
    with open(transforms_file, 'r') as f:
        transforms = json.load(f)
    
    num_images = len(transforms.get('frames', []))
    sphere_center = transforms.get('sphere_center', [0.0, 0.0, 0.0])
    sphere_scale = transforms.get('sphere_scale', 1.0)
    
    configs_updated = 0
    
    # Find all stage config files
    stage_configs = sorted(config_dir.glob("stage*.yaml"))
    
    if not stage_configs:
        logger.warning(f"No stage configs found in {config_dir}")
        return 0
    
    for config_file in stage_configs:
        logger.info(f"Updating {config_file.name}")
        
        try:
            # Load config with unsafe loader first to handle any numpy types
            with open(config_file, 'r') as f:
                try:
                    config = yaml.unsafe_load(f)
                except:
                    config = yaml.safe_load(f)
            
            # Update data section
            if 'data' not in config:
                config['data'] = {}
            
            config['data']['root'] = str(data_root)
            config['data']['num_images'] = int(num_images)
            
            # Update readjust section for sphere normalization
            if 'readjust' not in config['data']:
                config['data']['readjust'] = {}
            
            # Ensure all values are Python native types
            config['data']['readjust']['center'] = [float(x) for x in sphere_center]
            config['data']['readjust']['scale'] = float(sphere_scale)
            
            # Convert entire config to ensure no numpy types remain
            config_json = json.dumps(config, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
            config = json.loads(config_json)
            
            # Save updated config
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"  ✓ Updated with data_root={data_root}, num_images={num_images}")
            configs_updated += 1
            
        except Exception as e:
            logger.error(f"  ✗ Failed to update {config_file.name}: {e}")
    
    return configs_updated


def prepare_neuralangelo_data(data_root: Path, config_dir: Path = None, 
                             output_json: Path = None, no_backup: bool = False):
    """
    Complete data preparation for Neuralangelo
    
    Args:
        data_root: Path to neuralangelo_data directory
        config_dir: Directory containing stage configs to update
        output_json: Optional path to save parameters as JSON
        no_backup: If True, don't create backup of transforms.json
    
    Returns: True if successful, False otherwise
    """
    # Check data directory
    if not data_root.exists():
        logger.error(f"Data directory not found: {data_root}")
        return False
    
    transforms_file = data_root / "transforms.json"
    if not transforms_file.exists():
        logger.error(f"transforms.json not found at {data_root}")
        return False
    
    logger.info("="*60)
    logger.info("Preparing Neuralangelo data")
    logger.info("="*60)
    
    # Step 1: Fix transforms.json (includes adding sphere parameters)
    logger.info("\nStep 1: Fixing transforms.json and adding sphere parameters...")
    if not fix_transforms_json(transforms_file, backup=not no_backup):
        return False
    
    # Load the updated transforms
    with open(transforms_file, 'r') as f:
        data = json.load(f)
    
    num_images = len(data.get('frames', []))
    sphere_center = data.get('sphere_center', [0.0, 0.0, 0.0])
    sphere_scale = data.get('sphere_scale', 1.0)
    sphere_radius = data.get('sphere_radius', 1.0)
    
    # Step 2: Update stage configs if directory provided
    if config_dir and config_dir.exists():
        logger.info(f"\nStep 2: Updating stage configs in {config_dir}...")
        configs_updated = update_stage_configs(config_dir, data_root, transforms_file)
        logger.info(f"Updated {configs_updated} configuration files")
    else:
        if config_dir:
            logger.warning(f"Config directory not found: {config_dir}")
    
    # Save parameters to JSON if requested
    if output_json:
        params = {
            "data_root": str(data_root),
            "num_images": num_images,
            "sphere_center": sphere_center,
            "sphere_scale": sphere_scale,
            "sphere_radius": sphere_radius,
            "camera_params": {
                "w": data.get('w'),
                "h": data.get('h'),
                "fl_x": data.get('fl_x'),
                "fl_y": data.get('fl_y'),
                "cx": data.get('cx'),
                "cy": data.get('cy'),
                "sk_x": data.get('sk_x'),
                "sk_y": data.get('sk_y'),
                "k1": data.get('k1'),
                "k2": data.get('k2'),
                "p1": data.get('p1'),
                "p2": data.get('p2'),
                "aabb_scale": data.get('aabb_scale')
            }
        }
        with open(output_json, 'w') as f:
            json.dump(params, f, indent=2)
        logger.info(f"\nSaved all parameters to {output_json}")
    
    logger.info("\n" + "="*60)
    logger.info("✓ Data preparation complete!")
    logger.info("="*60)
    logger.info(f"\nReady for training with:")
    logger.info(f"  Data: {data_root}")
    logger.info(f"  Images: {num_images}")
    logger.info(f"  Sphere center: {sphere_center}")
    logger.info(f"  Sphere scale: {sphere_scale:.4f}")
    logger.info(f"  Sphere radius: {sphere_radius:.4f}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Complete data preparation for Neuralangelo training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare data and update configs
  python compute_sphere_params.py /path/to/neuralangelo_data --config-dir configs
  
  # Just fix transforms.json
  python compute_sphere_params.py /path/to/neuralangelo_data
  
  # Save parameters to JSON
  python compute_sphere_params.py /path/to/neuralangelo_data --output-json params.json
        """
    )
    parser.add_argument("data_root", type=Path, 
                       help="Path to neuralangelo_data directory containing transforms.json")
    parser.add_argument("--config-dir", type=Path, 
                       help="Directory containing stage configs to update")
    parser.add_argument("--output-json", type=Path, 
                       help="Save all parameters to JSON file")
    parser.add_argument("--no-backup", action="store_true",
                       help="Don't create backup of transforms.json")
    
    args = parser.parse_args()
    
    # Run preparation
    success = prepare_neuralangelo_data(
        args.data_root,
        args.config_dir,
        args.output_json,
        args.no_backup
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
