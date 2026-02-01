"""
modules/config.py
Pipeline configuration management
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml
import os


@dataclass
class VGGTConfig:
    """VGGT stage configuration"""
    enabled: bool = True
    required: bool = False
    script_path: Optional[Path] = None
    batch_size: int = 4
    confidence_threshold: float = 50.0
    use_depth_maps: bool = True
    
    def __post_init__(self):
        if self.script_path:
            self.script_path = Path(self.script_path)


@dataclass
class COLMAPConfig:
    """COLMAP stage configuration"""
    enabled: bool = True
    use_module: bool = False
    camera_model: str = "SIMPLE_PINHOLE"
    single_camera: bool = True
    num_threads: int = 8
    max_image_size: int = 3200
    max_features: int = 8192


@dataclass
class TrainingConfig:
    """Training configuration"""
    max_steps: int = 50000
    batch_size: Optional[int] = None  # Auto-determined if None
    rays_per_batch: Optional[int] = None
    learning_rate: float = 5e-4
    template_path: Optional[Path] = None
    checkpoint_interval: int = 5000
    
    def __post_init__(self):
        if self.template_path:
            self.template_path = Path(self.template_path)


@dataclass
class MeshConfig:
    """Mesh extraction configuration"""
    resolution: int = 2048
    block_resolution: int = 128
    threshold: float = 0.0
    format: str = "ply"


@dataclass
class StageConfigs:
    """All stage configurations"""
    vggt: VGGTConfig = field(default_factory=VGGTConfig)
    colmap: COLMAPConfig = field(default_factory=COLMAPConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)


@dataclass
class PipelineConfig:
    """Main pipeline configuration"""
    # Essential paths
    input_dir: Path = Path(".")
    output_dir: Path = Path("./output")
    
    # GPU settings
    gpu_index: int = 0
    
    # Source paths
    vggt_source: Path = Path.home() / "src/vggt"
    neuralangelo_source: Path = Path.home() / "src/neuralangelo"
    
    # Stage configurations
    stages: StageConfigs = field(default_factory=StageConfigs)
    
    # Aliases for backward compatibility
    @property
    def training(self) -> TrainingConfig:
        return self.stages.training
    
    @property
    def mesh(self) -> MeshConfig:
        return self.stages.mesh
    
    # Config file reference
    config_file: Optional[Path] = None
    
    def __post_init__(self):
        """Convert paths and validate"""
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        self.vggt_source = Path(self.vggt_source)
        self.neuralangelo_source = Path(self.neuralangelo_source)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'PipelineConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse nested configs
        if 'stages' in data:
            stages_data = data.pop('stages')
            stages = StageConfigs(
                vggt=VGGTConfig(**stages_data.get('vggt', {})),
                colmap=COLMAPConfig(**stages_data.get('colmap', {})),
                training=TrainingConfig(**stages_data.get('training', {})),
                mesh=MeshConfig(**stages_data.get('mesh', {}))
            )
            data['stages'] = stages
        
        config = cls(**data)
        config.config_file = Path(config_path)
        return config
    
    def to_file(self, path: Path):
        """Save configuration to YAML file"""
        data = {
            'input_dir': str(self.input_dir),
            'output_dir': str(self.output_dir),
            'gpu_index': self.gpu_index,
            'vggt_source': str(self.vggt_source),
            'neuralangelo_source': str(self.neuralangelo_source),
            'stages': {
                'vggt': {
                    'enabled': self.stages.vggt.enabled,
                    'required': self.stages.vggt.required,
                    'script_path': str(self.stages.vggt.script_path) if self.stages.vggt.script_path else None,
                    'batch_size': self.stages.vggt.batch_size,
                    'confidence_threshold': self.stages.vggt.confidence_threshold,
                    'use_depth_maps': self.stages.vggt.use_depth_maps,
                },
                'colmap': {
                    'enabled': self.stages.colmap.enabled,
                    'use_module': self.stages.colmap.use_module,
                    'camera_model': self.stages.colmap.camera_model,
                    'single_camera': self.stages.colmap.single_camera,
                    'num_threads': self.stages.colmap.num_threads,
                    'max_image_size': self.stages.colmap.max_image_size,
                    'max_features': self.stages.colmap.max_features,
                },
                'training': {
                    'max_steps': self.stages.training.max_steps,
                    'batch_size': self.stages.training.batch_size,
                    'rays_per_batch': self.stages.training.rays_per_batch,
                    'learning_rate': self.stages.training.learning_rate,
                    'template_path': str(self.stages.training.template_path) if self.stages.training.template_path else None,
                    'checkpoint_interval': self.stages.training.checkpoint_interval,
                },
                'mesh': {
                    'resolution': self.stages.mesh.resolution,
                    'block_resolution': self.stages.mesh.block_resolution,
                    'threshold': self.stages.mesh.threshold,
                    'format': self.stages.mesh.format,
                }
            }
        }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# Example configuration file content
EXAMPLE_CONFIG = """
# Pipeline Configuration Example
input_dir: /path/to/input
output_dir: /path/to/output
gpu_index: 0

# Source code paths
vggt_source: ~/src/vggt
neuralangelo_source: ~/src/neuralangelo

stages:
  vggt:
    enabled: true
    required: false
    script_path: ~/src/vggt/direct_neuralangelo.py
    batch_size: 4
    confidence_threshold: 50.0
    use_depth_maps: true
  
  colmap:
    enabled: true
    use_module: false
    camera_model: SIMPLE_PINHOLE
    single_camera: true
    num_threads: 8
    max_image_size: 3200
    max_features: 8192
  
  training:
    max_steps: 50000
    # batch_size: null  # Auto-determined
    # rays_per_batch: null  # Auto-determined
    learning_rate: 0.0005
    template_path: ~/configs/neuralangelo_template.yaml
    checkpoint_interval: 5000
  
  mesh:
    resolution: 2048
    block_resolution: 128
    threshold: 0.0
    format: ply
"""


def create_example_config(path: Path):
    """Create an example configuration file"""
    with open(path, 'w') as f:
        f.write(EXAMPLE_CONFIG)
    print(f"Example configuration created at: {path}")
