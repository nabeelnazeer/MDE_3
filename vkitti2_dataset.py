import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision.transforms as T

class VKitti2Dataset(Dataset):
    def __init__(self, root_dir, scene, variant, camera='Camera_0', split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.scene = scene
        self.variant = variant
        self.camera = camera
        
        # Paths to RGB and depth folders
        self.rgb_dir = self.root_dir / "vkitti_2.0.3_rgb" / scene / variant / "frames" / "rgb" / camera
        self.depth_dir = self.root_dir / "vkitti_2.0.3_depth" / scene / variant / "frames" / "depth" / camera
        
        print(f"Checking RGB directory: {self.rgb_dir}")
        
        # Collect all RGB files (JPG/PNG variants)
        self.rgb_files = []
        for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG']:
            self.rgb_files.extend(sorted(self.rgb_dir.glob(ext)))
        
        if len(self.rgb_files) == 0:
            print("WARNING: No image files found. Available files:")
            if self.rgb_dir.exists():
                for file in self.rgb_dir.iterdir():
                    print(f"  {file.name}")
            raise RuntimeError(f"No image files found in {self.rgb_dir}")
        
        print(f"Found {len(self.rgb_files)} image files")
        print(f"Sample filename: {self.rgb_files[0].name}")
        
        # Train/test split
        split_idx = int(0.8 * len(self.rgb_files))
        if split == 'train':
            self.rgb_files = self.rgb_files[:split_idx]
        else:
            self.rgb_files = self.rgb_files[split_idx:]
        
        # Default transform if none provided
        self.transform = transform if transform else self._get_default_transform()
    
    def _get_default_transform(self):
        return T.Compose([
            T.ToTensor(),
        ])
    
    def _depth_to_disparity(self, depth_tensor):
        """Convert depth to disparity space (1/depth)"""
        # Add small epsilon to avoid division by zero
        disparity = 1.0 / (depth_tensor + 1e-8)
        return disparity

    def _normalize_disparity(self, disparity_tensor):
        """Normalize disparity to [0,1] range"""
        normalized = (disparity_tensor - disparity_tensor.min()) / (disparity_tensor.max() - disparity_tensor.min())
        return normalized
    
    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]

        # Generate corresponding depth path (depth files are always .png)
        depth_filename = rgb_path.stem.replace('rgb_', 'depth_') + '.png'
        depth_path = self.depth_dir / depth_filename

        # Load images
        rgb_img = Image.open(rgb_path).convert('RGB')
        depth_img = Image.open(depth_path)
        
        # Convert depth to meters and then to disparity
        depth_np = np.array(depth_img, dtype=np.float32) / 100.0  # mm -> m
        disparity_np = 1.0 / (depth_np + 1e-8)  # raw disparity
        
        # All tensors need to be 224x224 for consistency
        transform_224 = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
        
        rgb_tensor = transform_224(rgb_img)
        depth_tensor = transform_224(depth_img)
        disparity_tensor = transform_224(Image.fromarray(disparity_np.astype(np.float32)))
        
        # Normalize disparity
        normalized_disparity = self._normalize_disparity(disparity_tensor)
        
        return rgb_tensor, depth_tensor, disparity_tensor, normalized_disparity
