"""
modules/gpu_utils.py
GPU detection and optimization utilities - Fixed for mid stage memory management
"""

import subprocess
import logging
import torch
import os
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class GPUManager:
    """Manages GPU detection and configuration"""
    
    # Known GPU configurations with conservative settings
    GPU_CONFIGS = {
        # NVIDIA H100/H200 series
        "H100": {"memory_gb": 80, "compute_capability": (9, 0), "arch": "hopper"},
        "H200": {"memory_gb": 141, "compute_capability": (9, 0), "arch": "hopper"},
        
        # NVIDIA B series (Blackwell)
        "B100": {"memory_gb": 180, "compute_capability": (10, 0), "arch": "blackwell"},
        "B200": {"memory_gb": 180, "compute_capability": (10, 0), "arch": "blackwell"},
        
        # NVIDIA A100 series
        "A100": {"memory_gb": 80, "compute_capability": (8, 0), "arch": "ampere"},
        "A100-40GB": {"memory_gb": 40, "compute_capability": (8, 0), "arch": "ampere"},
        
        # NVIDIA A series workstation
        "A6000": {"memory_gb": 48, "compute_capability": (8, 6), "arch": "ampere"},
        "A5000": {"memory_gb": 24, "compute_capability": (8, 6), "arch": "ampere"},
        "A4000": {"memory_gb": 16, "compute_capability": (8, 6), "arch": "ampere"},
        
        # NVIDIA RTX 40 series
        "RTX 4090": {"memory_gb": 24, "compute_capability": (8, 9), "arch": "ada"},
        "RTX 4080": {"memory_gb": 16, "compute_capability": (8, 9), "arch": "ada"},
        
        # NVIDIA RTX 30 series
        "RTX 3090": {"memory_gb": 24, "compute_capability": (8, 6), "arch": "ampere"},
        "RTX 3080": {"memory_gb": 10, "compute_capability": (8, 6), "arch": "ampere"},
        
        # Older GPUs
        "V100": {"memory_gb": 32, "compute_capability": (7, 0), "arch": "volta"},
        "TITAN RTX": {"memory_gb": 24, "compute_capability": (7, 5), "arch": "turing"},
    }
    
    def __init__(self):
        self.gpu_info = self._detect_gpu()
        self.gpu_config = self._get_gpu_config()
        
    def _detect_gpu(self) -> Dict:
        """Detect GPU using nvidia-smi"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, check=True
            )
            
            output = result.stdout.strip()
            if output:
                parts = output.split(',')
                name = parts[0].strip()
                memory_mb = int(parts[1].strip().replace(' MiB', ''))
                memory_gb = memory_mb / 1024
                
                return {
                    "name": name,
                    "memory_gb": memory_gb,
                    "memory_mb": memory_mb
                }
        except Exception as e:
            logger.warning(f"Could not detect GPU via nvidia-smi: {e}")
        
        # Fallback to PyTorch detection
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            return {
                "name": torch.cuda.get_device_name(device),
                "memory_gb": torch.cuda.get_device_properties(device).total_memory / (1024**3),
                "memory_mb": torch.cuda.get_device_properties(device).total_memory / (1024**2)
            }
        
        return {"name": "Unknown", "memory_gb": 8, "memory_mb": 8192}
    
    def _get_gpu_config(self) -> Dict:
        """Get configuration for detected GPU"""
        gpu_name = self.gpu_info["name"]
        
        # Check for exact match first
        for known_gpu, config in self.GPU_CONFIGS.items():
            if known_gpu in gpu_name:
                logger.info(f"Detected {known_gpu} - Applying optimizations")
                return config
        
        # Estimate based on memory
        memory_gb = self.gpu_info["memory_gb"]
        if memory_gb >= 140:
            logger.info(f"High-memory GPU detected ({memory_gb:.0f}GB) - Using H200/B200 settings")
            return self.GPU_CONFIGS["B200"]
        elif memory_gb >= 70:
            logger.info(f"Large GPU detected ({memory_gb:.0f}GB) - Using H100 settings")
            return self.GPU_CONFIGS["H100"]
        elif memory_gb >= 40:
            logger.info(f"Medium GPU detected ({memory_gb:.0f}GB) - Using A100 settings")
            return self.GPU_CONFIGS["A100"]
        else:
            logger.info(f"Standard GPU detected ({memory_gb:.0f}GB) - Using conservative settings")
            return {"memory_gb": memory_gb, "compute_capability": (7, 0), "arch": "unknown"}
    
    def clear_memory(self):
        """Clear GPU memory between stages"""
        import torch
        import gc
        
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
    
    def get_training_params(self, num_images: int, stage: str = "coarse") -> Dict:
        """Get optimized training parameters based on GPU and stage"""
        memory_gb = self.gpu_config.get("memory_gb", 24)
        
        # Stage-specific configurations with memory management
        if stage == "coarse":
            # Conservative settings for initial training
            if memory_gb >= 140:  # B200/H200
                return {
                    "batch_size": 4,
                    "train_resolution": [780, 1170],
                    "val_resolution": [260, 260],
                    "rays_per_image": 1024,
                    "rays_per_batch": 4096,
                    "grad_accum_iter": 1,
                    "empty_cache_every": 100,
                    "num_workers": 0
                }
            elif memory_gb >= 70:  # H100/A100-80GB
                return {
                    "batch_size": 2,
                    "train_resolution": [650, 975],
                    "val_resolution": [200, 200],
                    "rays_per_image": 1024,
                    "rays_per_batch": 2048,
                    "grad_accum_iter": 1,
                    "empty_cache_every": 100,
                    "num_workers": 0
                }
            else:  # Standard GPUs
                return {
                    "batch_size": 1,
                    "train_resolution": [520, 780],
                    "val_resolution": [200, 200],
                    "rays_per_image": 512,
                    "rays_per_batch": 512,
                    "grad_accum_iter": 2,
                    "empty_cache_every": 50,
                    "num_workers": 0
                }
        
        elif stage == "mid":
            # CRITICAL: Conservative settings to prevent OOM after coarse stage
            # Memory from coarse stage may not be fully released
            if memory_gb >= 140:  # B200/H200
                return {
                    "batch_size": 1,  # Reduced from 2
                    "train_resolution": [1000, 1500],  # Reduced from [1170, 1755]
                    "val_resolution": [260, 260],
                    "rays_per_image": 2048,  # Reduced from 4096
                    "rays_per_batch": 4096,  # Reduced from 8192
                    "grad_accum_iter": 2,  # Increased to maintain effective batch size
                    "empty_cache_every": 50,  # More frequent cache clearing
                    "num_workers": 0,
                    "checkpoint_gradient": True  # Enable gradient checkpointing
                }
            elif memory_gb >= 70:  # H100/A100-80GB
                return {
                    "batch_size": 1,
                    "train_resolution": [800, 1200],
                    "val_resolution": [200, 200],
                    "rays_per_image": 1536,
                    "rays_per_batch": 3072,
                    "grad_accum_iter": 2,
                    "empty_cache_every": 50,
                    "num_workers": 0,
                    "checkpoint_gradient": True
                }
            else:  # Standard GPUs
                return {
                    "batch_size": 1,
                    "train_resolution": [650, 975],
                    "val_resolution": [200, 200],
                    "rays_per_image": 1024,
                    "rays_per_batch": 2048,
                    "grad_accum_iter": 3,
                    "empty_cache_every": 25,
                    "num_workers": 0,
                    "checkpoint_gradient": True
                }
        
        elif stage == "fine":
            # Fine detail stage - balanced between quality and memory
            if memory_gb >= 140:  # B200/H200
                return {
                    "batch_size": 1,
                    "train_resolution": [1560, 2340],
                    "val_resolution": [390, 390],
                    "rays_per_image": 3072,
                    "rays_per_batch": 6144,
                    "grad_accum_iter": 2,
                    "empty_cache_every": 50,
                    "num_workers": 0,
                    "checkpoint_gradient": True
                }
            elif memory_gb >= 70:  # H100/A100-80GB
                return {
                    "batch_size": 1,
                    "train_resolution": [1300, 1950],
                    "val_resolution": [325, 325],
                    "rays_per_image": 2048,
                    "rays_per_batch": 4096,
                    "grad_accum_iter": 2,
                    "empty_cache_every": 50,
                    "num_workers": 0,
                    "checkpoint_gradient": True
                }
            else:  # Standard GPUs
                return {
                    "batch_size": 1,
                    "train_resolution": [1040, 1560],
                    "val_resolution": [260, 260],
                    "rays_per_image": 1536,
                    "rays_per_batch": 3072,
                    "grad_accum_iter": 3,
                    "empty_cache_every": 25,
                    "num_workers": 0,
                    "checkpoint_gradient": True
                }
        
        elif stage == "ultra":
            # Ultra quality - maximum resolution within memory limits
            if memory_gb >= 140:  # B200/H200
                return {
                    "batch_size": 1,
                    "train_resolution": [2080, 3120],
                    "val_resolution": [520, 520],
                    "rays_per_image": 4096,
                    "rays_per_batch": 8192,
                    "grad_accum_iter": 2,
                    "empty_cache_every": 25,
                    "num_workers": 0,
                    "checkpoint_gradient": True
                }
            elif memory_gb >= 70:  # H100/A100-80GB
                return {
                    "batch_size": 1,
                    "train_resolution": [1560, 2340],
                    "val_resolution": [390, 390],
                    "rays_per_image": 3072,
                    "rays_per_batch": 6144,
                    "grad_accum_iter": 3,
                    "empty_cache_every": 25,
                    "num_workers": 0,
                    "checkpoint_gradient": True
                }
            else:  # Standard GPUs - skip ultra or use fine settings
                logger.warning(f"Ultra stage may not fit in {memory_gb:.0f}GB memory")
                return self.get_training_params(num_images, "fine")
        
        else:
            # Default/unknown stage
            return self.get_training_params(num_images, "coarse")
    
    def setup_environment(self):
        """Setup environment variables for optimal GPU usage"""
        gpu_name = self.gpu_info["name"]
        
        # Enable expandable segments for better memory management
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        
        # B200-specific optimizations
        if "B200" in gpu_name or "B100" in gpu_name:
            logger.info("Configuring for NVIDIA B200/B100 (Blackwell)")
            os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
            os.environ["TORCH_CUDNN_V8_USE_GRAPH_MODE"] = "1"
            
            # Conservative memory management for B200
            os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
            
            # Disable validation to save memory
            os.environ["NEURALANGELO_SKIP_VALIDATION"] = "1"
        
        # H100/H200 optimizations
        elif "H100" in gpu_name or "H200" in gpu_name:
            logger.info("Configuring for NVIDIA H100/H200 (Hopper)")
            os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
            
        # Set general NCCL settings
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
    
    def get_info(self) -> Dict:
        """Get GPU information"""
        return {
            **self.gpu_info,
            **self.gpu_config,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        }
    
    def log_info(self):
        """Log GPU information"""
        info = self.get_info()
        logger.info(f"GPU: {info['name']}")
        logger.info(f"Memory: {info['memory_gb']:.0f} GB ({info['memory_mb']:.0f} MB)")
        
        if "compute_capability" in info:
            cc = info["compute_capability"]
            logger.info(f"CUDA Capability: {cc[0]}.{cc[1]}")
            
            if cc[0] >= 10:
                logger.warning("⚠️  B200 (sm_100) may require PyTorch 2.5+ for full support")
        
        if "arch" in info:
            logger.info(f"Architecture: {info['arch']}")


# Convenience function
def get_gpu_manager() -> GPUManager:
    """Get or create GPU manager instance"""
    if not hasattr(get_gpu_manager, "_instance"):
        get_gpu_manager._instance = GPUManager()
    return get_gpu_manager._instance
