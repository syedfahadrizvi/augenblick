#!/usr/bin/env python3
"""
VGGT Batch Processor - Process all images together for proper multi-view reconstruction
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.geometry import unproject_depth_map_to_point_map
except ImportError:
    print("Error: VGGT not found. Please install VGGT first.")
    exit(1)

try:
    import trimesh
except ImportError:
    trimesh = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def pose_encoding_to_matrix(pose_enc: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Convert VGGT pose encoding to extrinsic and intrinsic matrices"""
    # pose_enc shape can be [9], [N, 9], or [1, N, 9]
    if pose_enc.ndim == 3 and pose_enc.shape[0] == 1:
        pose_enc = pose_enc[0]  # Remove batch dimension: [1, N, 9] -> [N, 9]
    elif pose_enc.ndim == 1:
        pose_enc = pose_enc[np.newaxis, :]  # [9] -> [1, 9]
    
    num_poses = pose_enc.shape[0]
    extrinsics = []
    intrinsics = []
    
    for i in range(num_poses):
        pose = pose_enc[i]  # Get individual pose [9]
        
        # Extract rotation (first 6) and translation (last 3)
        rot_6d = pose[:6]
        translation = pose[6:9]
        
        # Convert 6D rotation to 3x3 matrix
        # Based on "On the Continuity of Rotation Representations in Neural Networks"
        a1 = rot_6d[:3]
        a2 = rot_6d[3:6]
        
        # Normalize a1
        b1 = a1 / (np.linalg.norm(a1) + 1e-8)
        
        # Make a2 orthogonal to b1
        b2 = a2 - np.dot(b1, a2) * b1
        b2 = b2 / (np.linalg.norm(b2) + 1e-8)
        
        # Third axis via cross product
        b3 = np.cross(b1, b2)
        
        # Stack to create rotation matrix
        rotation = np.stack([b1, b2, b3], axis=1)  # 3x3
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        
        extrinsics.append(transform)
        
        # Estimate intrinsics (VGGT doesn't provide explicit intrinsics)
        # Using default values based on 518x518 output
        focal_length = 518 * 1.2  # Reasonable estimate
        intrinsic = np.array([
            [focal_length, 0, 259],  # 518/2 = 259
            [0, focal_length, 259],
            [0, 0, 1]
        ])
        intrinsics.append(intrinsic)
    
    return extrinsics, intrinsics


def process_all_images_together(
    image_paths: List[Path],
    model: VGGT,
    device: str,
    dtype: torch.dtype
) -> Dict:
    """Process all images in a single batch for proper multi-view geometry"""
    
    logger.info(f"Loading and preprocessing {len(image_paths)} images...")
    
    # Load all images at once
    image_paths_str = [str(p) for p in image_paths]
    images = load_and_preprocess_images(image_paths_str).to(device)
    
    logger.info(f"Images tensor shape: {images.shape}")
    
    # Run VGGT inference on all images together
    with torch.no_grad():
        with torch.amp.autocast(device, dtype=dtype):
            predictions = model(images)
    
    # Log what we got
    logger.info("VGGT outputs:")
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    # Extract results
    results = {
        'depth_maps': [],
        'depth_conf': [],
        'world_points': [],
        'extrinsics': [],
        'intrinsics': [],
        'images': images.cpu(),
        'pose_encodings': []
    }
    
    # Process pose encodings
    if 'pose_enc' in predictions:
        pose_enc = predictions['pose_enc'].cpu().numpy()
        logger.info(f"Pose encoding shape: {pose_enc.shape}")
        
        # Convert to matrices
        extrinsics, intrinsics = pose_encoding_to_matrix(pose_enc)
        results['extrinsics'] = extrinsics
        results['intrinsics'] = intrinsics
        results['pose_encodings'] = pose_enc
    
    # Process depth maps
    if 'depth' in predictions:
        depth = predictions['depth'].cpu().numpy()
        logger.info(f"Original depth shape: {depth.shape}")
        
        # Handle shape [1, 69, 518, 518, 1]
        if depth.shape[0] == 1:
            depth = depth[0]  # Remove batch dim -> [69, 518, 518, 1]
        if depth.shape[-1] == 1:
            depth = depth[..., 0]  # Remove last dim -> [69, 518, 518]
        
        # Now depth should be [69, 518, 518]
        for i in range(depth.shape[0]):
            results['depth_maps'].append(depth[i])
    
    # Process depth confidence
    for conf_key in ('depth_conf', 'world_points_conf'):
        if conf_key in predictions:
            conf = predictions[conf_key].cpu().numpy()
            logger.info(f"Original {conf_key} shape: {conf.shape}")
            if conf.shape[0] == 1:
                conf = conf[0]
            if conf.ndim == 4 and conf.shape[-1] == 1:
                conf = conf[..., 0]
            elif conf.ndim == 4 and conf.shape[1] == 1:
                conf = conf[:, 0]
            for i in range(conf.shape[0]):
                results['depth_conf'].append(conf[i])
            break
    
    # Process world points
    if 'world_points' in predictions:
        points = predictions['world_points'].cpu().numpy()
        logger.info(f"Original world_points shape: {points.shape}")
        
        # Handle shape [1, 69, 518, 518, 3]
        if points.shape[0] == 1:
            points = points[0]  # Remove batch dim -> [69, 518, 518, 3]
        
        for i in range(points.shape[0]):
            results['world_points'].append(points[i])
    
    return results


def save_point_cloud_ply(
    results: Dict,
    output_dir: Path,
    conf_threshold: float = 2.0,
    max_points: int = 500000,
) -> Optional[Path]:
    """Generate and save a .ply point cloud from VGGT depth predictions.

    Adapted from masked_reconstruction_vggt.py step2_run_vggt:
    unprojects depth maps into 3D, filters by confidence, attaches RGB
    colours from the input images, and exports via trimesh.
    """
    if trimesh is None:
        logger.error("trimesh is not installed. Install it with: pip install trimesh")
        return None

    logger.info("Generating .ply point cloud...")

    # --- 3D points -----------------------------------------------------------
    # Prefer direct world_points from VGGT; fall back to depth unprojection.
    if results['world_points']:
        points_3d = np.stack(results['world_points'], axis=0)  # (S, H, W, 3)
        logger.info(f"Using world_points directly: {points_3d.shape}")
    elif results['depth_maps'] and results['extrinsics']:
        depth = np.stack(results['depth_maps'], axis=0)  # (S, H, W)
        extrinsic = np.stack(results['extrinsics'], axis=0)  # (S, 4, 4)
        intrinsic = np.stack(results['intrinsics'], axis=0)  # (S, 3, 3)
        points_3d = unproject_depth_map_to_point_map(
            depth, extrinsic[:, :3, :], intrinsic
        )
        logger.info(f"Unprojected depth to 3D points: {points_3d.shape}")
    else:
        logger.error("Cannot generate point cloud: no world_points or depth data")
        return None

    # --- RGB colours ----------------------------------------------------------
    images_tensor = results['images']
    if images_tensor.ndim == 5 and images_tensor.shape[0] == 1:
        images_tensor = images_tensor[0]  # remove outer batch dim
    # (S, C, H, W) -> (S, H, W, C)
    points_rgb = images_tensor.permute(0, 2, 3, 1).numpy()
    points_rgb = np.clip(points_rgb * 255, 0, 255).astype(np.uint8)

    # Resize RGB to match point-map spatial resolution if needed
    pt_h, pt_w = points_3d.shape[1], points_3d.shape[2]
    if points_rgb.shape[1] != pt_h or points_rgb.shape[2] != pt_w:
        resized = np.zeros((points_rgb.shape[0], pt_h, pt_w, 3), dtype=np.uint8)
        for i in range(points_rgb.shape[0]):
            img = Image.fromarray(points_rgb[i])
            img = img.resize((pt_w, pt_h), Image.BILINEAR)
            resized[i] = np.array(img)
        points_rgb = resized

    # --- Confidence mask ------------------------------------------------------
    if results['depth_conf']:
        depth_conf = np.stack(results['depth_conf'], axis=0)  # (S, H, W)
        # Resize conf to match point-map resolution if shapes differ
        if depth_conf.shape[1:] != (pt_h, pt_w):
            from PIL import Image as _PILImg
            resized_conf = np.zeros((depth_conf.shape[0], pt_h, pt_w), dtype=depth_conf.dtype)
            for i in range(depth_conf.shape[0]):
                c = _PILImg.fromarray(depth_conf[i])
                c = c.resize((pt_w, pt_h), _PILImg.BILINEAR)
                resized_conf[i] = np.array(c)
            depth_conf = resized_conf
        conf_mask = depth_conf >= conf_threshold
        logger.info(
            f"Confidence filter ({conf_threshold}): "
            f"{conf_mask.sum()} / {conf_mask.size} points pass"
        )
    else:
        conf_mask = np.ones(points_3d.shape[:3], dtype=bool)
        logger.info("No confidence data available, using all points")

    # --- Flatten & filter -----------------------------------------------------
    points_3d_flat = points_3d[conf_mask]
    points_rgb_flat = points_rgb[conf_mask]

    # Remove NaN / Inf / extreme values
    valid = np.isfinite(points_3d_flat).all(axis=-1)
    points_3d_flat = points_3d_flat[valid]
    points_rgb_flat = points_rgb_flat[valid]
    logger.info(f"Valid points after filtering: {len(points_3d_flat)}")

    # Sub-sample if there are too many points
    if len(points_3d_flat) > max_points:
        indices = np.random.choice(len(points_3d_flat), max_points, replace=False)
        points_3d_flat = points_3d_flat[indices]
        points_rgb_flat = points_rgb_flat[indices]
        logger.info(f"Subsampled to {max_points} points")

    if len(points_3d_flat) == 0:
        logger.error("No valid 3D points to export")
        return None

    # --- Export ----------------------------------------------------------------
    ply_path = output_dir / "points.ply"
    cloud = trimesh.PointCloud(points_3d_flat, colors=points_rgb_flat)
    cloud.export(str(ply_path))

    file_size_mb = ply_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"Saved point cloud: {ply_path} "
        f"({len(points_3d_flat)} points, {file_size_mb:.1f} MB)"
    )
    return ply_path


def save_results(results: Dict, image_paths: List[Path], output_dir: Path):
    """Save all results in Neuralangelo format"""
    # Create directories
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    (output_dir / "depth").mkdir(parents=True, exist_ok=True)
    
    # Save processed images
    logger.info("Saving processed images...")
    images_tensor = results['images']
    
    # Handle image tensor shape
    if images_tensor.shape[0] == 1:
        images_tensor = images_tensor[0]  # Remove batch dim if present
    
    for idx in range(images_tensor.shape[0]):
        img_array = images_tensor[idx].permute(1, 2, 0).numpy()
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        img.save(output_dir / "images" / f"frame_{idx:06d}.png")
        
        # Copy mask if it exists
        orig_path = image_paths[idx]
        mask_path = orig_path.parent.parent / "masks" / f"{orig_path.stem}.png"
        if mask_path.exists():
            mask = Image.open(mask_path).convert("L")
            mask_resized = mask.resize((518, 518), Image.NEAREST)
            mask_resized.save(output_dir / "masks" / f"frame_{idx:06d}.png")
    
    # Save depth maps
    logger.info("Saving depth maps...")
    for idx, depth in enumerate(results['depth_maps']):
        # Save raw depth
        np.save(output_dir / "depth" / f"frame_{idx:06d}_depth.npy", depth)
        
        # Save visualization
        depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        depth_vis = (depth_vis * 255).astype(np.uint8)
        Image.fromarray(depth_vis).save(output_dir / "depth" / f"frame_{idx:06d}_depth.png")
    
    # Save camera parameters
    logger.info("Saving camera parameters...")
    transforms = {
        "camera_model": "PINHOLE",
        "frames": []
    }
    
    for idx in range(len(image_paths)):
        frame_data = {
            "file_path": f"images/frame_{idx:06d}.png",
            "transform_matrix": results['extrinsics'][idx].tolist() if idx < len(results['extrinsics']) else None,
            "intrinsics": {
                "fx": float(results['intrinsics'][idx][0, 0]),
                "fy": float(results['intrinsics'][idx][1, 1]),
                "cx": float(results['intrinsics'][idx][0, 2]),
                "cy": float(results['intrinsics'][idx][1, 2])
            } if idx < len(results['intrinsics']) else None
        }
        transforms["frames"].append(frame_data)
    
    with open(output_dir / "transforms.json", 'w') as f:
        json.dump(transforms, f, indent=2)
    
    # Verify all frames have poses
    null_count = sum(1 for frame in transforms["frames"] if frame["transform_matrix"] is None)
    logger.info(f"Frames with camera poses: {len(transforms['frames']) - null_count}/{len(transforms['frames'])}")
    
    if null_count > 0:
        logger.warning(f"{null_count} frames are missing camera poses!")
    else:
        logger.info("✅ All frames have camera poses!")
    
    # Save metadata
    metadata = {
        "num_frames": len(image_paths),
        "processing_mode": "batch_multiview",
        "pose_encodings_shape": list(results['pose_encodings'].shape) if len(results['pose_encodings']) > 0 else None,
        "all_poses_estimated": null_count == 0
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Process all images together through VGGT")
    parser.add_argument("images", type=Path, help="Input images directory")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--metadata", type=Path, help="Path to metadata.json from prep_crop.py")
    parser.add_argument("--conf_threshold", type=float, default=2.0,
                        help="Depth confidence threshold for point cloud filtering (default: 2.0)")
    parser.add_argument("--max_points", type=int, default=500000,
                        help="Maximum number of points in the .ply output (default: 500000)")
    parser.add_argument("--no_ply", action="store_true",
                        help="Skip .ply point cloud generation")
    
    args = parser.parse_args()
    
    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Get image paths
    image_paths = sorted(args.images.glob("*.png")) + sorted(args.images.glob("*.jpg"))
    logger.info(f"Found {len(image_paths)} images")
    
    if not image_paths:
        logger.error(f"No images found in {args.images}")
        return
    
    # Load model
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    logger.info(f"Loading VGGT model on {args.device} with dtype {dtype}")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(args.device)
    model.eval()
    
    # Process all images together
    results = process_all_images_together(image_paths, model, args.device, dtype)
    
    # Save results
    save_results(results, image_paths, args.output_dir)
    
    # Generate .ply point cloud
    if not args.no_ply:
        ply_path = save_point_cloud_ply(
            results, args.output_dir,
            conf_threshold=args.conf_threshold,
            max_points=args.max_points,
        )
        if ply_path:
            logger.info(f"Point cloud saved to: {ply_path}")
    
    logger.info(f"\n✅ Batch processing complete!")
    logger.info(f"Output: {args.output_dir}")
    logger.info("\nOutputs:")
    logger.info(f"  - transforms.json  (camera poses)")
    logger.info(f"  - images/          (processed frames)")
    logger.info(f"  - depth/           (depth maps)")
    if not args.no_ply:
        logger.info(f"  - points.ply       (colored point cloud)")


if __name__ == "__main__":
    main()
