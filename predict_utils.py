import torch
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

def predict_sample_depths(model, samples_dir, output_dir, device):
    """Predict depths for all samples using current model"""
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    
    model.eval()
    samples_dir = Path(samples_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    with torch.no_grad():
        for img_path in tqdm(list(samples_dir.glob('*.jpg')) + list(samples_dir.glob('*.png')), desc="Predicting depths"):
            # Load and transform image
            img = Image.open(img_path).convert('RGB')
            input_tensor = transform(img).unsqueeze(0).to(device)
            
            # Predict depth
            pred_disparity = model(input_tensor)
            pred_disparity = pred_disparity.squeeze().cpu()
            
            # Visualize
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            ax1.imshow(img)
            ax1.set_title('Input Image')
            ax1.axis('off')
            
            depth_vis = ax2.imshow(pred_disparity, cmap='plasma')
            ax2.set_title('Predicted Depth')
            ax2.axis('off')
            plt.colorbar(depth_vis, ax=ax2)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{img_path.stem}_depth.png', bbox_inches='tight', dpi=300)
            plt.close()
