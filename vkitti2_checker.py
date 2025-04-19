import os
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# Try importing torch, but continue if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not found. Some functionality will be limited.")

class VKitti2Checker:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.rgb_dir = self.root_dir / "vkitti_2.0.3_rgb"
        self.depth_dir = self.root_dir / "vkitti_2.0.3_depth"
        self.scenes = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
        self.variants = ['15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right', 
                        'clone', 'fog', 'morning', 'overcast', 'rain', 'sunset']
        self.cameras = ['Camera_0', 'Camera_1']
        self.check_dependencies()
        
    def check_dependencies(self):
        """Check if all required dependencies are available."""
        missing_deps = []
        try:
            import PIL
        except ImportError:
            missing_deps.append("Pillow")
        
        if not TORCH_AVAILABLE:
            missing_deps.append("PyTorch")
            
        if missing_deps:
            print("Warning: Missing dependencies:", ", ".join(missing_deps))
            print("Install them using: pip install " + " ".join(missing_deps))
    
    def check_dataset(self) -> Dict:
        stats = {
            'total_samples': 0,
            'scene_stats': {},
            'sample_shapes': None
        }
        
        for scene in self.scenes:
            scene_stats = self._check_scene(scene)
            stats['scene_stats'][scene] = scene_stats
            stats['total_samples'] += sum(scene_stats['images_per_camera'].values()) * len(self.variants)
            
        # Check sample shapes
        sample_rgb, sample_depth = self._get_sample_shapes()
        stats['sample_shapes'] = {
            'rgb': sample_rgb,
            'depth': sample_depth
        }
        
        return stats
    
    def _check_scene(self, scene: str) -> Dict:
        scene_stats = {
            'variants': self.variants,
            'images_per_camera': {},
            'image_sizes': {},
            'image_formats': {}
        }
        
        for camera in self.cameras:
            # Check one variant (clone) for basic stats
            rgb_path = self.rgb_dir / scene / 'clone' / 'frames' / 'rgb' / camera
            depth_path = self.depth_dir / scene / 'clone' / 'frames' / 'depth' / camera
            
            rgb_files = list(rgb_path.glob('*.jpg'))
            depth_files = list(depth_path.glob('*.png'))
            
            if rgb_files:
                rgb_img = Image.open(rgb_files[0])
                depth_img = Image.open(depth_files[0])
                
                scene_stats['images_per_camera'][camera] = len(rgb_files)
                scene_stats['image_sizes'][camera] = {
                    'rgb': rgb_img.size,
                    'depth': depth_img.size
                }
                scene_stats['image_formats'][camera] = {
                    'rgb': rgb_img.mode,
                    'depth': depth_img.mode
                }
        
        return scene_stats
    
    def _get_sample_shapes(self) -> Tuple[tuple, tuple]:
        # Get first image from first scene
        rgb_path = next((self.rgb_dir / 'Scene01' / 'clone' / 'frames' / 'rgb' / 'Camera_0').glob('*.jpg'))
        depth_path = next((self.depth_dir / 'Scene01' / 'clone' / 'frames' / 'depth' / 'Camera_0').glob('*.png'))
        
        # Convert to tensor and resize
        rgb = Image.open(rgb_path)
        depth = Image.open(depth_path)
        
        rgb = rgb.resize((224, 224))
        depth = depth.resize((224, 224))
        
        if TORCH_AVAILABLE:
            rgb_tensor = torch.FloatTensor(rgb).permute(2, 0, 1)
            depth_tensor = torch.FloatTensor(depth)
            return rgb_tensor.shape, depth_tensor.shape
        else:
            # Return tuple representation when torch is not available
            rgb_shape = (3, 224, 224)  # C,H,W format
            depth_shape = (224, 224)    # H,W format
            return rgb_shape, depth_shape

    def print_stats(self, stats: Dict):
        for scene, scene_stats in stats['scene_stats'].items():
            print(f"\nScene: {scene}")
            print(f"Variants: {scene_stats['variants']}")
            
            for camera in self.cameras:
                print(f"\nImages in {camera}: {scene_stats['images_per_camera'][camera]}")
                print(f"{camera} RGB image size: {scene_stats['image_sizes'][camera]['rgb']}")
                print(f"{camera} Depth image size: {scene_stats['image_sizes'][camera]['depth']}")
                print(f"{camera} RGB format: {scene_stats['image_formats'][camera]['rgb']}")
                print(f"{camera} Depth format: {scene_stats['image_formats'][camera]['depth']}")
        
        print(f"\nTotal number of samples: {stats['total_samples']}")
        print("\nSample data shapes:")
        print(f"RGB tensor shape: {stats['sample_shapes']['rgb']}")
        print(f"Depth tensor shape: {stats['sample_shapes']['depth']}")

if __name__ == "__main__":
    checker = VKitti2Checker("/Users/nabeelnazeer/Documents/Project-s6/Datasets")
    stats = checker.check_dataset()
    checker.print_stats(stats)
