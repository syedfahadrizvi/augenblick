"""
FILE: stage_training_manager.py
Multi-stage training orchestrator for Neuralangelo
Progressively increases resolution and model capacity
With GPU-aware optimization, memory management, and robust error handling
"""

import os
import sys
import yaml
import json
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import time
import random
import torch
import gc
import re
from contextlib import contextmanager

# Import GPU manager for hardware-aware configs
sys.path.append(str(Path(__file__).parent))
from modules.gpu_utils import GPUManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StageSchedule(NamedTuple):
    """Learning rate schedule for a stage"""
    warm_up: int
    steps: List[int]
    gamma: float = 10.0


@dataclass
class StageConfig:
    """Configuration for a single training stage"""
    name: str
    config_file: Path
    max_iter: int
    schedule: StageSchedule
    checkpoint_dir: Optional[Path] = None
    resume_from: Optional[Path] = None
    
    def __repr__(self):
        return f"Stage({self.name}, iters={self.max_iter})"


class MultiStageTrainer:
    """Manages multi-stage progressive training for Neuralangelo"""
    
    # Default stage configurations with proper LR schedules
    # (name, max_iter, config_file, schedule)
    STAGES = [
        ("coarse", 2000, "stage1_coarse.yaml", 
         StageSchedule(warm_up=200, steps=[800, 1400])),
        ("mid", 10000, "stage2_mid.yaml",
         StageSchedule(warm_up=500, steps=[3000, 7000])),
        ("fine", 30000, "stage3_fine.yaml",
         StageSchedule(warm_up=1000, steps=[10000, 20000])),
        ("ultra", 50000, "stage4_ultra.yaml",
         StageSchedule(warm_up=2000, steps=[15000, 35000])),
    ]
    
    def __init__(self, 
                 data_root: Path,
                 output_dir: Path,
                 neuralangelo_path: Path,
                 config_dir: Optional[Path] = None,
                 gpu_index: int = 0):
        """
        Initialize the multi-stage trainer
        
        Args:
            data_root: Path to Neuralangelo dataset
            output_dir: Output directory for all stages
            neuralangelo_path: Path to Neuralangelo source
            config_dir: Directory containing stage config files
            gpu_index: GPU to use
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.neuralangelo_path = Path(neuralangelo_path)
        self.config_dir = config_dir or Path(__file__).parent / "configs"
        self.gpu_index = gpu_index
        
        # Initialize GPU manager for hardware-aware configs
        self.gpu_manager = GPUManager()
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_base = self.output_dir / "logs"
        self.logs_base.mkdir(exist_ok=True)
        
        # Load dataset info
        self.dataset_info = self._load_dataset_info()
        
        # Setup stages
        self.stages = self._setup_stages()
        
    def _load_dataset_info(self) -> Dict:
        """Load dataset information from sphere_params.json or transforms.json"""
        tf = self.data_root / "transforms.json"
        sp = self.data_root / "sphere_params.json"

        source = None
        data = {}
        if sp.exists():
            with open(sp, 'r') as f:
                data = json.load(f)
            source = sp
        elif tf.exists():
            with open(tf, 'r') as f:
                data = json.load(f)
            source = tf
        else:
            raise FileNotFoundError(f"Neither sphere_params.json nor transforms.json found at {self.data_root}")

        # Prefer explicit fields
        center = data.get('sphere_center')
        radius = data.get('sphere_radius')
        scale  = data.get('sphere_scale')

        # Derive missing fields conservatively
        if radius is None and scale is not None and scale > 0:
            radius = 1.0 / float(scale)
        if scale is None and radius is not None and radius > 0:
            scale = 1.0 / float(radius)

        # Fallbacks
        center = center or [0.0, 0.0, 0.0]
        radius = float(radius or 1.0)
        scale  = float(scale or 1.0)

        # Pull num_images from transforms if available
        num_images = 0
        if tf.exists():
            try:
                with open(tf, 'r') as f:
                    tdata = json.load(f)
                num_images = len(tdata.get('frames', []))
            except Exception:
                pass

        info = {
            'num_images': num_images,
            'sphere_center': center,
            'sphere_radius': radius,
            'sphere_scale': scale,
            'sphere_source': str(source)
        }

        logger.info(f"Dataset: {info['num_images']} images, sphere_radius={info['sphere_radius']:.3f} (from {source.name})")
        return info
    
    def _setup_stages(self) -> List[StageConfig]:
        """Setup training stages"""
        stages = []
        previous_checkpoint = None
        
        for stage_name, max_iter, config_name, schedule in self.STAGES:
            config_file = self.config_dir / config_name
            
            # Stage-specific log directory
            stage_dir = self.logs_base / f"stage_{stage_name}"
            checkpoint_dir = stage_dir
            
            stage = StageConfig(
                name=stage_name,
                config_file=config_file,
                max_iter=max_iter,
                schedule=schedule,
                checkpoint_dir=checkpoint_dir,
                resume_from=previous_checkpoint
            )
            stages.append(stage)
            
            # Next stage resumes from this stage's checkpoint
            if checkpoint_dir.exists():
                checkpoints = sorted(checkpoint_dir.glob("*.pt"))
                if checkpoints:
                    previous_checkpoint = checkpoints[-1]
            
        return stages
    
    def _clear_gpu_memory(self):
        """Clear GPU memory between stages"""
        logger.info("Clearing GPU memory...")
        
        # Empty CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Log memory stats
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"Memory after cleanup: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

    @contextmanager
    def _temporary_meshing_radius(self, margin: float):
        """Temporarily enlarge sphere_radius for meshing, then restore both files."""
        if margin is None or abs(margin - 1.0) < 1e-6:
            # No-op context
            yield
            return

        tf = self.data_root / "transforms.json"
        sp = self.data_root / "sphere_params.json"

        # Read originals (if present)
        def _read(p):
            if p.exists():
                with open(p, 'r') as f:
                    return json.load(f)
            return None

        tf_orig = _read(tf)
        sp_orig = _read(sp)

        # Compute new radius from current values
        center = self.dataset_info['sphere_center']
        radius = self.dataset_info['sphere_radius']
        new_radius = float(radius) * float(margin)

        # Write patched copies to both files so any code path picks them up
        def _write(p, payload):
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'w') as f:
                json.dump(payload, f, indent=2)

        if sp_orig is not None:
            sp_new = dict(sp_orig)
            sp_new['sphere_center'] = center
            sp_new['sphere_radius'] = new_radius
            # Keep scale consistent inside the JSON if someone reads it
            sp_new['sphere_scale'] = 1.0 / new_radius
            _write(sp, sp_new)

        if tf_orig is not None:
            tf_new = dict(tf_orig)
            tf_new['sphere_center'] = center
            tf_new['sphere_radius'] = new_radius
            # Do not touch frames
            if 'sphere_scale' in tf_new:
                tf_new['sphere_scale'] = 1.0 / new_radius
            _write(tf, tf_new)

        logger.info(f"[Mesh] applied temporary radius margin x{margin:.2f} -> radius={new_radius:.6f}")

        try:
            yield
        finally:
            # Restore originals
            def _restore(p, payload):
                if payload is None:
                    return
                with open(p, 'w') as f:
                    json.dump(payload, f, indent=2)

            if tf_orig is not None:
                _restore(tf, tf_orig)
            if sp_orig is not None:
                _restore(sp, sp_orig)
            logger.info("[Mesh] restored original sphere JSONs after extraction")
    
    def _prepare_stage_config(self, stage: StageConfig) -> Path:
        """Prepare configuration file for a stage with proper values"""
        # Read template
        if not stage.config_file.exists():
            logger.error(f"Config file not found: {stage.config_file}")
            raise FileNotFoundError(f"Config file not found: {stage.config_file}")
        
        with open(stage.config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Direct dict manipulation instead of text replacement
        config['data']['root'] = str(self.data_root)
        config['data']['num_images'] = self.dataset_info['num_images']
        config['data']['readjust']['center'] = self.dataset_info['sphere_center']
        config['data']['readjust']['scale'] = self.dataset_info['sphere_scale']
        
        # GPU-aware parameters based on stage
        gpu_params = self.gpu_manager.get_training_params(
            self.dataset_info['num_images'], 
            stage.name
        )
        
        # Apply GPU-optimized settings
        config['data']['train']['batch_size'] = gpu_params['batch_size']
        config['data']['train']['image_size'] = gpu_params['train_resolution']
        config['data']['train']['rays_per_image'] = gpu_params['rays_per_image']
        config['data']['val']['image_size'] = gpu_params['val_resolution']
        
        # Set rand_rays (the actual field used by Neuralangelo)
        if 'render' not in config['model']:
            config['model']['render'] = {}
        config['model']['render']['rand_rays'] = gpu_params['rays_per_batch']
        
        # Auto-fix common issues
        fixed = []
        
        # Fix gradient taps if needed
        if 'object' in config['model'] and 'sdf' in config['model']['object']:
            if 'gradient' in config['model']['object']['sdf']:
                gradient = config['model']['object']['sdf']['gradient']
                if gradient.get('taps') not in [4, 6]:
                    gradient['taps'] = 4
                    fixed.append('gradient.taps=4')
        
        # Ensure trainer grad_accum_iter is set
        if 'trainer' not in config:
            config['trainer'] = {}
        config['trainer']['grad_accum_iter'] = gpu_params['grad_accum_iter']
        
        # Set empty_cache_every for memory management
        config['trainer']['empty_cache_every'] = gpu_params.get('empty_cache_every', 100)
        
        # Override max_iter from stage config (single source of truth)
        config['max_iter'] = stage.max_iter
        
        # Apply learning rate schedule from stage
        config['optim']['sched']['warm_up_end'] = stage.schedule.warm_up
        config['optim']['sched']['two_steps'] = stage.schedule.steps
        config['optim']['sched']['gamma'] = stage.schedule.gamma
        
        # Add resume checkpoint if available
        if stage.resume_from and stage.resume_from.exists():
            if 'checkpoint' not in config:
                config['checkpoint'] = {}
            config['checkpoint']['resume'] = str(stage.resume_from)
            logger.info(f"Stage {stage.name} will resume from {stage.resume_from.name}")
        
        if fixed:
            logger.info(f"Auto-fixed config issues: {', '.join(fixed)}")
        
        # Save prepared config
        prepared_config = self.logs_base / f"stage_{stage.name}" / "config.yaml"
        prepared_config.parent.mkdir(exist_ok=True)
        
        with open(prepared_config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Prepared config for stage {stage.name} with GPU-optimized params")
        logger.info(f"  Resolution: {gpu_params['train_resolution']}")
        logger.info(f"  Batch size: {gpu_params['batch_size']}")
        logger.info(f"  Rays per batch: {gpu_params['rays_per_batch']}")
        
        return prepared_config
    
    def _validate_config(self, config_path: Path) -> bool:
        """Validate that a config has all required fields"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        errors = []
        
        # Check critical fields
        if not config.get('data', {}).get('root'):
            errors.append("Missing data.root")
        
        render = config.get('model', {}).get('render', {})
        if render.get('rand_rays') is None:
            errors.append("Missing model.render.rand_rays")
        
        gradient = config.get('model', {}).get('object', {}).get('sdf', {}).get('gradient', {})
        if gradient.get('taps') not in [4, 6]:
            errors.append(f"Invalid gradient.taps: {gradient.get('taps')} (must be 4 or 6)")
        
        if config.get('max_iter') is None:
            errors.append("Missing max_iter")
        
        if errors:
            logger.error(f"Config validation failed for {config_path}:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        return True

    def _find_latest_checkpoint(self, stage_name: str) -> Optional[Path]:
        """Find the best checkpoint for a stage, robust to naming and layout."""
        # Typical location
        root = self.output_dir / "logs" / f"stage_{stage_name}"
        candidates: List[Path] = []

        # Collect .pt files from expected dir; fall back to a wider search
        if root.exists():
            candidates.extend(sorted(root.glob("*.pt")))
        if not candidates:
            candidates.extend(sorted(self.output_dir.rglob(f"stage_{stage_name}/**/*.pt")))

        if not candidates:
            return None

        def score(p: Path) -> Tuple[int, int]:
            # Prefer "latest" explicitly
            if "latest" in p.stem:
                return (10**9, 10**9)
            # Try to parse iteration_########
            m = re.search(r"iteration_(\d+)", p.name)
            if m:
                return (int(m.group(1)), 0)
            # Try a simpler fallback: any trailing number
            m = re.search(r"(\d+)", p.stem)
            return (int(m.group(1)) if m else -1, -1)

        return max(candidates, key=score)

    def run_stage(self, stage: StageConfig) -> bool:
        """Run a single training stage with memory cleanup"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Stage: {stage.name.upper()} ({stage.max_iter} iterations)")
        logger.info(f"{'='*60}")
        
        # MEMORY CLEANUP BEFORE STAGE
        if stage.name != "coarse":  # Don't clean before first stage
            logger.info("Performing memory cleanup before stage...")
            self._clear_gpu_memory()
            time.sleep(5)  # Give time for memory to settle
        
        # Check if already completed
        if self._is_stage_complete(stage):
            logger.info(f"Stage {stage.name} already complete, skipping...")
            return True
        
        # Prepare configuration
        config_path = self._prepare_stage_config(stage)
        
        # Validate configuration
        if not self._validate_config(config_path):
            logger.error(f"Configuration validation failed for stage {stage.name}")
            return False
        
        stage_dir = config_path.parent
        
        # Run training
        original_cwd = os.getcwd()
        try:
            os.chdir(self.neuralangelo_path)
            
            # Use random port to avoid conflicts
            port = random.randint(29500, 29900)
            
            cmd = [
                "torchrun", "--nproc_per_node=1", 
                f"--master_port={port}",
                "train.py",
                "--logdir", str(stage_dir),
                "--config", str(config_path),
                "--show_pbar"
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
            
            # Add memory management environment variable
            env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
            
            result = subprocess.run(cmd, env=env, check=True)
            
            logger.info(f"✓ Stage {stage.name} completed successfully")
            
            # MEMORY CLEANUP AFTER STAGE
            logger.info("Cleaning up after stage completion...")
            self._clear_gpu_memory()
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Stage {stage.name} failed: {e}")
            
            # Memory cleanup even on failure
            self._clear_gpu_memory()
            
            return False
        finally:
            os.chdir(original_cwd)
    
    def _is_stage_complete(self, stage: StageConfig) -> bool:
        """Check if a stage is already complete"""
        if not stage.checkpoint_dir or not stage.checkpoint_dir.exists():
            return False
        def _iter_num(p: Path) -> int:
            m = re.search(r"iteration_(\d+)", p.stem)
            if m:
                return int(m.group(1))
            # treat 'latest' as very large so it wins if present
            return 10**9 if "latest" in p.name else -1
        pts = list(stage.checkpoint_dir.glob("*.pt"))
        if not pts:
            return False
        best = max(pts, key=_iter_num)
        return _iter_num(best) >= int(0.95 * stage.max_iter)
    
    def run_all_stages(self, start_stage: Optional[str] = None, 
                       end_stage: Optional[str] = None,
                       max_retries: int = 2) -> bool:
        """Run all training stages sequentially with retry logic and memory management"""
        start_time = time.time()
        
        # Find stage indices
        stage_names = [s.name for s in self.stages]
        start_idx = 0
        end_idx = len(self.stages)
        
        if start_stage:
            if start_stage in stage_names:
                start_idx = stage_names.index(start_stage)
            else:
                logger.error(f"Unknown start stage: {start_stage}")
                return False
        
        if end_stage:
            if end_stage in stage_names:
                end_idx = stage_names.index(end_stage) + 1
            else:
                logger.error(f"Unknown end stage: {end_stage}")
                return False
        
        # Run stages
        logger.info(f"Running stages: {[s.name for s in self.stages[start_idx:end_idx]]}")
        
        for i, stage in enumerate(self.stages[start_idx:end_idx], start=start_idx):
            # Update resume checkpoint from previous stage if available
            if i > 0:
                prev_stage = self.stages[i-1]
                if prev_stage.checkpoint_dir and prev_stage.checkpoint_dir.exists():
                    checkpoints = sorted(prev_stage.checkpoint_dir.glob("*.pt"))
                    if checkpoints:
                        stage.resume_from = checkpoints[-1]
                
                # ADD DELAY BETWEEN STAGES for memory to settle
                logger.info("Waiting 10 seconds between stages for memory cleanup...")
                time.sleep(10)
            
            # Try running stage with retries
            success = False
            for retry in range(max_retries):
                if retry > 0:
                    logger.info(f"Retrying stage {stage.name} (attempt {retry + 1}/{max_retries})")
                    time.sleep(10)  # Wait before retry
                    
                    # Extra memory cleanup before retry
                    self._clear_gpu_memory()
                
                if self.run_stage(stage):
                    success = True
                    break
            
            if not success:
                logger.error(f"Training failed at stage {stage.name} after {max_retries} attempts")
                return False

            # Extract mesh after coarse and mid stages
            if stage.name in ["coarse", "mid"]:
                logger.info(f"Extracting mesh after {stage.name} stage (with radius margin)...")
                mesh_resolution = 256 if stage.name == "coarse" else 512
                mesh_path = self.extract_mesh(
                    stage_name=stage.name,
                    resolution=mesh_resolution,
                    block_res=64 if stage.name == "coarse" else 128,
                    radius_margin=1.30  # be generous; this is just for diagnostics
                )
                if mesh_path:
                    logger.info(f"✓ Mesh extracted for {stage.name} stage: {mesh_path}")
                else:
                    logger.warning(f"⚠ Failed to extract mesh for {stage.name} stage")
        
        elapsed = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ ALL STAGES COMPLETED SUCCESSFULLY")
        logger.info(f"Total time: {elapsed/3600:.1f} hours")
        logger.info(f"{'='*60}")
        
        return True
    
    def extract_mesh(self, stage_name: str = "ultra",
                     resolution: int = 2048,
                     block_res: int = 128,
                     radius_margin: float = 1.25) -> Optional[Path]:
        """Extract mesh from a trained stage with a temporary meshing radius margin."""
        # Find stage config and checkpoint directory
        stage = None
        for s in self.stages:
            if s.name == stage_name:
                stage = s
                break

        if not stage:
            logger.error(f"Unknown stage: {stage_name}")
            return None

        if not stage.checkpoint_dir or not stage.checkpoint_dir.exists():
            # Try default expected path
            fallback = self.logs_base / f"stage_{stage_name}"
            if fallback.exists():
                stage.checkpoint_dir = fallback
            else:
                logger.error(f"No checkpoints found for stage {stage_name} (looked in {fallback})")
                return None

        # fallback: search recursively if the exact dir isn't there yet
        candidates = []
        if stage.checkpoint_dir and stage.checkpoint_dir.exists():
            candidates += list(stage.checkpoint_dir.glob("*.pt"))
        if not candidates:
            candidates = list(self.output_dir.rglob(f"stage_{stage_name}/**/*.pt"))
        if not candidates:
            candidates = list(self.output_dir.rglob("*.pt"))

        if not candidates:
            logger.error(f"No checkpoint files anywhere under {self.output_dir}")
            return None

        checkpoint = max(candidates, key=lambda p: p.stat().st_mtime)

        # Prefer the stage-specific config
        stage_dir = self.output_dir / "logs" / f"stage_{stage_name}"
        cfg = stage_dir / "config.yaml"
        if not cfg.exists():
            # Fallback search
            cfgs = list(stage_dir.rglob("config.yaml")) or list(self.output_dir.rglob(f"stage_{stage_name}/**/config.yaml"))
            cfg = cfgs[0] if cfgs else None
        if not cfg or not cfg.exists():
            logger.error(f"Config not found for stage {stage_name}")
            return None

        mesh_path = self.output_dir / f"mesh_{stage_name}_res{resolution}.ply"
        logger.info(f"Extracting mesh (stage={stage_name}, res={resolution}, block_res={block_res})")
        logger.info(f"  checkpoint: {checkpoint.name}")
        logger.info(f"  config:     {cfg}")

        original_cwd = os.getcwd()
        try:
            os.chdir(self.neuralangelo_path)
            port = random.randint(29700, 29900)

            cmd = [
                "torchrun", "--nproc_per_node=1", f"--master_port={port}",
                "projects/neuralangelo/scripts/extract_mesh.py",
                "--config", str(cfg),
                "--checkpoint", str(checkpoint),
                "--output_file", str(mesh_path),
                "--resolution", str(resolution),
                "--block_res", str(block_res),
            ]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)

            # Temporarily enlarge radius for meshing, then restore
            with self._temporary_meshing_radius(radius_margin):
                # Log the sphere we will actually use
                logger.info(f"[Mesh] center={self.dataset_info['sphere_center']}  "
                            f"radius(original)={self.dataset_info['sphere_radius']:.6f}  "
                            f"margin={radius_margin:.2f}")
                subprocess.run(cmd, env=env, check=True)

            if mesh_path.exists():
                size_mb = mesh_path.stat().st_size / (1024 * 1024)
                logger.info(f"✓ Mesh extracted: {mesh_path} ({size_mb:.1f} MB)")
                return mesh_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Mesh extraction command failed with return code {e.returncode}")
            logger.error(f"Command: {' '.join(cmd)}")
        except Exception as e:
            logger.error(f"Unexpected error during mesh extraction: {e} ({type(e).__name__})")
        finally:
            os.chdir(original_cwd)

        return None


def main():
    """Main entry point for multi-stage training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-stage Neuralangelo training")
    parser.add_argument("data_root", type=Path, help="Path to Neuralangelo dataset")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("--neuralangelo-path", type=Path, 
                       default=Path.home() / "src/neuralangelo",
                       help="Path to Neuralangelo source")
    parser.add_argument("--config-dir", type=Path, help="Directory with stage configs")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--start-stage", choices=["coarse", "mid", "fine", "ultra"],
                       help="Start from specific stage")
    parser.add_argument("--end-stage", choices=["coarse", "mid", "fine", "ultra"],
                       help="End at specific stage")
    parser.add_argument("--extract-only", action="store_true",
                       help="Only extract mesh from latest stage")
    parser.add_argument("--mesh-resolution", type=int, default=2048,
                       help="Mesh extraction resolution")
    parser.add_argument("--mesh-stage", default="ultra",
                       help="Stage to extract mesh from")
    parser.add_argument("--mesh-radius-margin", type=float, default=1.25,
                       help="Temporary radius multiplier used only during mesh extraction")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = MultiStageTrainer(
        data_root=args.data_root,
        output_dir=args.output_dir,
        neuralangelo_path=args.neuralangelo_path,
        config_dir=args.config_dir,
        gpu_index=args.gpu
    )
    
    if args.extract_only:
        # Just extract mesh
        mesh = trainer.extract_mesh(
            stage_name=args.mesh_stage,
            resolution=args.mesh_resolution,
            radius_margin=args.mesh_radius_margin
        )
        return 0 if mesh else 1
    else:
        # Run training
        success = trainer.run_all_stages(
            start_stage=args.start_stage,
            end_stage=args.end_stage
        )
        
        if success and args.mesh_resolution > 0:
            # Extract mesh after training
            final_stage = args.end_stage or "ultra"
            trainer.extract_mesh(
                stage_name=final_stage,
                resolution=args.mesh_resolution,
                radius_margin=args.mesh_radius_margin
            )
        
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
