import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from DYNO import DynoV2DepthEstimator
import matplotlib.pyplot as plt
from tqdm import tqdm

def predict_depth(model, image_path, device, transform):
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_disparity = model(input_tensor)
    
    return pred_disparity.squeeze().cpu()

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        try:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
        except:
            pass
    
    # Load best model
    model = DynoV2DepthEstimator(pretrained=False).to(device)
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    
    # Process all samples
    samples_dir = Path('/Users/nabeelnazeer/Documents/Project-s6/MDE_3/samples')
    output_dir = Path('predictions')
    output_dir.mkdir(exist_ok=True)
    
    for img_path in tqdm(list(samples_dir.glob('*.jpg')) + list(samples_dir.glob('*.png'))):
        # Predict depth
        pred_disparity = predict_depth(model, img_path, device, transform)
        
        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Original image
        img = Image.open(img_path).convert('RGB')
        ax1.imshow(img)
        ax1.set_title('Input Image')
        ax1.axis('off')
        
        # Predicted depth
        depth_vis = ax2.imshow(pred_disparity, cmap='plasma')
        ax2.set_title('Predicted Depth')
        ax2.axis('off')
        plt.colorbar(depth_vis, ax=ax2)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{img_path.stem}_depth.png', bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == '__main__':
    main()
