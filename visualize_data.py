try:
    import torch
    import matplotlib.pyplot as plt
    from vkitti2_dataset import VKitti2Dataset
    from pathlib import Path
    import numpy as np
except ImportError as e:
    print(f"Error: {e}")
    print("\nPlease run the following commands:")
    print("python3 -m venv venv")
    print(". venv/bin/activate")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu")
    print("pip install matplotlib numpy Pillow tqdm")
    exit(1)

def visualize_sample(rgb, depth, disparity, norm_disparity, save_path=None):
    rgb = rgb.permute(1, 2, 0).numpy()
    
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    axs[0].imshow(rgb)
    axs[0].set_title("RGB")
    axs[0].axis("off")
    
    depth_vis = axs[1].imshow(depth, cmap="plasma")
    axs[1].set_title("Depth (meters)")
    axs[1].axis("off")
    plt.colorbar(depth_vis, ax=axs[1])
    
    disp_vis = axs[2].imshow(disparity, cmap="plasma")
    axs[2].set_title("Raw Disparity")
    axs[2].axis("off")
    plt.colorbar(disp_vis, ax=axs[2])
    
    norm_disp_vis = axs[3].imshow(norm_disparity.squeeze(), cmap="plasma", vmin=0, vmax=1)
    axs[3].set_title("Normalized Disparity")
    axs[3].axis("off")
    plt.colorbar(norm_disp_vis, ax=axs[3])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def main():
    root_dir = Path('/Users/nabeelnazeer/Documents/Project-s6/Datasets')
    scenes = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
    variants = ['clone', 'fog', 'morning', 'overcast', 'rain', 'sunset']
    
    output_dir = Path('visualization_samples')
    output_dir.mkdir(exist_ok=True)
    
    # Sample from each variant
    for scene in scenes:
        for variant in variants:
            print(f"\nVisualizing {scene} - {variant}")
            dataset = VKitti2Dataset(root_dir, scene, variant, split='train')
            
            idx = np.random.randint(len(dataset))
            rgb, depth, disparity, norm_disparity = dataset[idx]
            
            save_path = output_dir / f'sample_{scene}_{variant}_{idx}.png'
            visualize_sample(rgb, depth, disparity.squeeze().numpy(), norm_disparity, save_path)
            print(f"Saved to: {save_path}")

if __name__ == '__main__':
    main()
