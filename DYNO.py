import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as T
from torch.hub import load_state_dict_from_url
import math

class DynoV2DepthEstimator(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        self.encoder = DynoV2Encoder(pretrained=pretrained)
        
        # DPT-style decoder with proper dimensions
        self.decoder = DPTDecoder(
            encoder_channels=[1024, 1024, 1024, 1024],
            decoder_channels=512,
            output_size=(224, 224)  # Match dataset image size
        )
        
        self.depth_head = DepthPredictionHead(
            in_channels=512,
            scale_factor=1  # No need for additional scaling
        )
    
    def forward(self, x):
        # Get encoder features
        features = self.encoder(x)
        
        # Decode features
        decoded = self.decoder(features)
        
        # Predict normalized disparity
        disparity = self.depth_head(decoded)
        return disparity

class DynoV2Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        
        if pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        B = x.shape[0]
        features = []

        # Use model's prepare_tokens method which handles patching and position embeddings
        x = self.model.prepare_tokens_with_masks(x)
        
        # Process through transformer blocks and collect features
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in [5, 11, 17, 23]:  # DINOv2-Large has 24 blocks
                # Remove CLS token and reshape to spatial dimensions
                tokens = x[:, 1:, :]  # Remove CLS token
                H = W = int(math.sqrt(tokens.shape[1]))
                feat = tokens.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                features.append(feat)
        
        return features

class DPTDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, output_size):
        super().__init__()
        self.output_size = output_size
        self.reassemble_blocks = nn.ModuleList([
            ReassembleBlock(in_ch, decoder_channels)
            for in_ch in encoder_channels
        ])
    
    def forward(self, features):
        # Process features
        decoded = []
        for feat, reassemble in zip(features, self.reassemble_blocks):
            x = reassemble(feat)
            # Upscale to target size
            x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=True)
            decoded.append(x)
        
        # Progressive fusion
        x = decoded[-1]
        for feat in reversed(decoded[:-1]):
            x = x + feat
        
        return x

class DepthPredictionHead(nn.Module):
    def __init__(self, in_channels, scale_factor=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )
        self.scale_factor = scale_factor
    
    def forward(self, x):
        x = self.conv(x)
        if self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        return x

# Utility blocks
class DynoBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 8)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm(x))[0]
        x = x + self.mlp(self.norm(x))
        return x

class ReassembleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class FusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, x1, x2):
        x2 = F.interpolate(x2, size=x1.shape[-2:], mode='bilinear', align_corners=True)
        return self.norm(self.conv(torch.cat([x1, x2], dim=1)))
