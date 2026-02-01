"""
modules/dataset.py
Dataset organization and validation - Modified to handle masks in masks/ subdirectory
"""

import shutil
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class DatasetOrganizer:
    """Handles dataset organization, validation, and preprocessing"""
    
    VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    MASK_PATTERN = '.mask.png'
    
    def __init__(self, config):
        self.config = config
        self.input_dir = Path(config.input_dir)
        self.output_dir = Path(config.output_dir)
        self.organized_dir = self.output_dir / "organized"
        self.images_dir = self.organized_dir / "images"
        self.masks_dir = self.organized_dir / "masks"
        
        # Dataset info
        self.image_count = 0
        self.mask_count = 0
        self.dataset_info = {}
    
    def organize(self) -> bool:
        """Main organization entry point"""
        try:
            logger.info("Organizing dataset...")
            
            # Validate input
            if not self._validate_input():
                return False
            
            # Create directories
            self._create_directories()
            
            # Find and organize files
            image_files, mask_files = self._find_files()
            
            if not image_files:
                logger.error("No valid images found in input directory")
                return False
            
            # Copy and organize files
            self._organize_files(image_files, mask_files)
            
            # Validate dataset
            if not self._validate_dataset():
                return False
            
            # Generate dataset report
            self._generate_report()
            
            logger.info(f"✓ Dataset organized: {self.image_count} images, {self.mask_count} masks")
            return True
            
        except Exception as e:
            logger.error(f"Dataset organization failed: {e}")
            return False
    
    def _validate_input(self) -> bool:
        """Validate input directory"""
        if not self.input_dir.exists():
            logger.error(f"Input directory does not exist: {self.input_dir}")
            return False
        
        if not self.input_dir.is_dir():
            logger.error(f"Input path is not a directory: {self.input_dir}")
            return False
        
        return True
    
    def _create_directories(self):
        """Create output directory structure"""
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directories under {self.organized_dir}")
    
    def _find_files(self) -> Tuple[List[Path], Dict[str, Path]]:
        """Find image and mask files - handles both .mask.png pattern and masks/ subdirectory"""
        
        # Check if we have standard subdirectory structure
        images_subdir = self.input_dir / "images"
        masks_subdir = self.input_dir / "masks"
        
        if images_subdir.exists() and images_subdir.is_dir():
            # We have an images/ subdirectory - use it
            logger.info("Found images/ subdirectory - using structured input")
            image_files = []
            for ext in self.VALID_IMAGE_EXTENSIONS:
                image_files.extend(images_subdir.glob(f"*{ext}"))
            
            # Build mask lookup from masks/ subdirectory
            mask_lookup = {}
            if masks_subdir.exists() and masks_subdir.is_dir():
                logger.info("Found masks/ subdirectory - looking for corresponding masks")
                for mask_file in masks_subdir.glob("*.png"):
                    # Match based on filename stem
                    mask_lookup[mask_file.stem] = mask_file
            
            logger.info(f"Found {len(image_files)} images and {len(mask_lookup)} masks")
            return sorted(image_files), mask_lookup
        
        else:
            # Fall back to original behavior - look for files with .mask.png pattern
            logger.info("No subdirectory structure found - looking for .mask.png pattern")
            all_files = list(self.input_dir.rglob("*"))
            image_files = []
            mask_files = []
            
            for f in all_files:
                if not f.is_file():
                    continue
                
                if self.MASK_PATTERN in f.name:
                    mask_files.append(f)
                elif f.suffix.lower() in self.VALID_IMAGE_EXTENSIONS:
                    if self.MASK_PATTERN not in f.name:
                        image_files.append(f)
            
            # Convert mask_files list to lookup dict
            mask_lookup = {}
            for mask_file in mask_files:
                # Extract base name from mask
                mask_name = mask_file.name.replace(self.MASK_PATTERN, '')
                base_name = Path(mask_name).stem
                mask_lookup[base_name] = mask_file
            
            logger.info(f"Found {len(image_files)} images and {len(mask_files)} masks")
            return sorted(image_files), mask_lookup
    
    def _organize_files(self, image_files: List[Path], mask_lookup: Dict[str, Path]):
        """Copy and organize files with proper naming"""
        
        self.image_count = 0
        self.mask_count = 0
        paired_count = 0
        
        for img_file in image_files:
            # Standardize naming
            img_name = f"image_{self.image_count:04d}{img_file.suffix}"
            dst_image = self.images_dir / img_name
            
            # Copy image
            if not dst_image.exists():
                shutil.copy2(img_file, dst_image)
                logger.debug(f"Copied image: {img_file.name} -> {img_name}")
            
            # Check for corresponding mask
            # Try to find mask using the image's stem
            img_stem = img_file.stem
            mask_found = False
            
            # Direct stem match
            if img_stem in mask_lookup:
                mask_file = mask_lookup[img_stem]
                mask_name = f"image_{self.image_count:04d}.png"
                dst_mask = self.masks_dir / mask_name
                
                if not dst_mask.exists():
                    shutil.copy2(mask_file, dst_mask)
                    logger.debug(f"Copied mask for {img_name} from {mask_file.name}")
                    self.mask_count += 1
                    paired_count += 1
                    mask_found = True
            
            if not mask_found:
                # Try case-insensitive match
                for mask_stem, mask_file in mask_lookup.items():
                    if mask_stem.lower() == img_stem.lower():
                        mask_name = f"image_{self.image_count:04d}.png"
                        dst_mask = self.masks_dir / mask_name
                        
                        if not dst_mask.exists():
                            shutil.copy2(mask_file, dst_mask)
                            logger.debug(f"Copied mask for {img_name} from {mask_file.name} (case-insensitive match)")
                            self.mask_count += 1
                            paired_count += 1
                            mask_found = True
                            break
            
            if not mask_found:
                logger.debug(f"No mask found for {img_file.name}")
            
            self.image_count += 1
        
        logger.info(f"Organized {self.image_count} images, {self.mask_count} masks ({paired_count} paired)")
    
    def _validate_dataset(self) -> bool:
        """Validate organized dataset"""
        # Check image sizes and formats
        image_files = sorted(self.images_dir.glob("*"))
        
        if not image_files:
            logger.error("No images in organized dataset")
            return False
        
        # Sample first image for reference
        first_img = Image.open(image_files[0])
        ref_size = first_img.size
        ref_mode = first_img.mode
        
        logger.info(f"Reference image: size={ref_size}, mode={ref_mode}")
        
        # Store dataset info
        self.dataset_info = {
            'image_count': len(image_files),
            'mask_count': len(list(self.masks_dir.glob("*"))),
            'image_size': ref_size,
            'image_mode': ref_mode,
            'consistent_sizes': True
        }
        
        # Check consistency (sample 10% of images)
        sample_size = max(1, len(image_files) // 10)
        sample_indices = np.random.choice(len(image_files), sample_size, replace=False)
        
        for idx in sample_indices:
            img = Image.open(image_files[idx])
            if img.size != ref_size:
                logger.warning(f"Inconsistent image size: {image_files[idx].name} has size {img.size}")
                self.dataset_info['consistent_sizes'] = False
        
        return True
    
    def _generate_report(self):
        """Generate dataset report"""
        report_path = self.organized_dir / "dataset_info.txt"
        
        with open(report_path, 'w') as f:
            f.write("Dataset Organization Report\n")
            f.write("="*50 + "\n\n")
            f.write(f"Source: {self.input_dir}\n")
            f.write(f"Output: {self.organized_dir}\n\n")
            f.write(f"Images: {self.dataset_info['image_count']}\n")
            f.write(f"Masks: {self.dataset_info['mask_count']}\n")
            f.write(f"Image size: {self.dataset_info['image_size']}\n")
            f.write(f"Image mode: {self.dataset_info['image_mode']}\n")
            f.write(f"Consistent sizes: {self.dataset_info['consistent_sizes']}\n")
        
        logger.info(f"Dataset report saved to {report_path}")
    
    def get_paths(self) -> Tuple[Path, Path]:
        """Get organized dataset paths"""
        return self.images_dir, self.masks_dir
    
    def get_info(self) -> Dict:
        """Get dataset information"""
        return self.dataset_info
