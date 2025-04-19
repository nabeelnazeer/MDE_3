import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.nn import functional as F
from vkitti2_dataset import VKitti2Dataset
from DYNO import DynoV2DepthEstimator
from pathlib import Path
import numpy as np
from tqdm import tqdm
import math
from predict_utils import predict_sample_depths

def compute_silog_loss(pred, target, mask=None, variance_focus=0.85):
    """
    Compute scale-invariant logarithmic loss
    """
    def log10(x):
        return torch.log(x) / torch.log(torch.tensor(10.0, device=x.device))
    
    pred = pred.squeeze(1)
    target = target.squeeze(1)
    
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    
    diff = log10(pred) - log10(target)
    silog = torch.sqrt((diff ** 2).mean() - variance_focus * (diff.mean() ** 2))
    
    return silog

def compute_depth_metrics(pred, target, mask=None):
    """Compute depth estimation metrics"""
    # Resize prediction to match target size
    if pred.shape != target.shape:
        pred = F.interpolate(pred.unsqueeze(1), size=target.shape[-2:], mode='bilinear', align_corners=True).squeeze(1)
    
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    
    thresh = torch.max((target / pred), (pred / target))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    
    rmse = (target - pred).abs().pow(2).mean().sqrt()
    rmse_log = (torch.log(target) - torch.log(pred)).pow(2).mean().sqrt()
    
    abs_rel = (target - pred).abs().mean() / target.mean()
    sq_rel = ((target - pred) ** 2).mean() / target.mean()
    
    return {
        'a1': a1.item(),
        'a2': a2.item(),
        'a3': a3.item(),
        'rmse': rmse.item(),
        'rmse_log': rmse_log.item(),
        'abs_rel': abs_rel.item(),
        'sq_rel': sq_rel.item()
    }

def train_model(model, train_loader, val_loader, device, save_dir, samples_dir, num_epochs=10):
    optimizer = AdamW(model.decoder.parameters(), lr=1e-4)  # Only train decoder
    
    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    best_metrics = {'silog': float('inf')}
    metrics_log = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_metrics = {'loss': 0, 'silog': 0}
        
        for rgb, depth, _, norm_disparity in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            rgb, depth, norm_disparity = rgb.to(device), depth.to(device), norm_disparity.to(device)
            
            pred_disparity = model(rgb)  # Output is already 224x224
            
            # norm_disparity is also 224x224, no resizing needed
            l1_loss = F.l1_loss(pred_disparity, norm_disparity)
            silog_loss = compute_silog_loss(pred_disparity, norm_disparity)
            loss = l1_loss + 0.85 * silog_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_metrics['loss'] += loss.item()
            train_metrics['silog'] += silog_loss.item()
        
        # Validation
        model.eval()
        val_metrics = {'loss': 0, 'silog': 0}
        all_metrics = []
        
        with torch.no_grad():
            for rgb, depth, _, norm_disparity in tqdm(val_loader, desc='Validating'):
                rgb, depth, norm_disparity = rgb.to(device), depth.to(device), norm_disparity.to(device)
                
                pred_disparity = model(rgb)
                
                # All tensors are already 224x224, no resizing needed
                pred_depth = 1.0 / pred_disparity.squeeze(1)
                metrics = compute_depth_metrics(pred_depth, depth.squeeze(1))
                all_metrics.append(metrics)
                
                # For loss computation, use the model output size
                val_metrics['loss'] += F.l1_loss(pred_disparity, norm_disparity).item()
                val_metrics['silog'] += compute_silog_loss(pred_disparity, norm_disparity).item()
        
        # Average metrics
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
        for k in val_metrics:
            val_metrics[k] /= len(val_loader)
        
        avg_metrics = {}
        for k in all_metrics[0].keys():
            avg_metrics[k] = np.mean([m[k] for m in all_metrics])
        
        # Log metrics
        print(f'\nEpoch {epoch+1} Metrics:')
        print(f'Training - Loss: {train_metrics["loss"]:.4f}, SiLog: {train_metrics["silog"]:.4f}')
        print(f'Validation - Loss: {val_metrics["loss"]:.4f}, SiLog: {val_metrics["silog"]:.4f}')
        print('Depth Metrics:')
        for k, v in avg_metrics.items():
            print(f'{k}: {v:.4f}')
        
        # Save best model and predict on samples
        if val_metrics['silog'] < best_metrics['silog']:
            best_metrics = {**val_metrics, **avg_metrics}
            model_save_path = save_dir / f'best_model_epoch_{epoch}.pth'
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': best_metrics,
            }, model_save_path)
            print(f'\nSaved new best model: {model_save_path}')
            
            # Predict on samples
            predictions_dir = save_dir / f'predictions_epoch_{epoch}'
            predict_sample_depths(model, samples_dir, predictions_dir, device)
            print(f'Generated predictions in: {predictions_dir}')
        
        metrics_log.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'depth': avg_metrics
        })

    return metrics_log

def main():
    # Device configuration with error handling
    device = torch.device("cpu")
    try:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
    except:
        pass
    print(f"Using device: {device}")
    
    # Setup directories
    project_dir = Path('/Users/nabeelnazeer/Documents/Project-s6/MDE_3')
    data_dir = Path('/Users/nabeelnazeer/Documents/Project-s6/Datasets')  # Directory containing vkitti dataset
    checkpoints_dir = project_dir / 'checkpoints'
    samples_dir = project_dir / 'samples'
    
    # Create necessary directories
    checkpoints_dir.mkdir(exist_ok=True)
    samples_dir.mkdir(exist_ok=True)
    
    # Initialize model
    model = DynoV2DepthEstimator(pretrained=True).to(device)
    
    # Create combined datasets
    train_datasets = []
    val_datasets = []
    
    for scene in ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']:
        for variant in ['clone', 'fog', 'morning', 'overcast', 'rain', 'sunset']:
            print(f"Loading {scene} - {variant}")
            train_datasets.append(VKitti2Dataset(data_dir, scene, variant, split='train'))
            val_datasets.append(VKitti2Dataset(data_dir, scene, variant, split='val'))
    
    # Combine datasets
    combined_train = ConcatDataset(train_datasets)
    combined_val = ConcatDataset(val_datasets)
    
    print(f"Total training samples: {len(combined_train)}")
    print(f"Total validation samples: {len(combined_val)}")
    
    # Create dataloaders
    train_loader = DataLoader(combined_train, batch_size=2, shuffle=True, num_workers=4)
    val_loader = DataLoader(combined_val, batch_size=2, shuffle=False, num_workers=4)
    
    # Train on entire dataset
    metrics_log = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=checkpoints_dir,
        samples_dir=samples_dir
    )

if __name__ == '__main__':
    main()
