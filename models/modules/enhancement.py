"""
Enhancement Module - Low-light image enhancement with Swin Transformer and ResNeXt
Combines attention mechanisms and grouped convolutions for image enhancement
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_
import math
from .common import Mlp, window_partition, window_reverse, WindowAttention, SwinTransformerBlock, init_weights, create_stochastic_depth_decay


class ResNeXtBottleneckC(nn.Module):
    """
    ResNeXt Bottleneck Block - Form C (with grouped convolution)
    Based on the original ResNeXt implementation
    
    Args:
        in_channels: number of input channels
        out_channels: number of output channels  
        cardinality: number of groups for grouped convolution
        base_width: base width per group
        stride: stride for convolution
    """
    def __init__(self, in_channels, out_channels, cardinality=32, base_width=4, stride=1):
        super().__init__()
        
        # Calculate group width D
        D = int(math.floor(out_channels * (base_width / 64)) * cardinality)
        
        self.conv1 = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D)
        
        # Grouped convolution
        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=stride, 
                              padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(D)
        
        self.conv3 = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNeXtBottleneckB(nn.Module):
    """
    ResNeXt Bottleneck Block - Form B (with split-transform-merge)
    Based on the original ResNeXt implementation
    
    This form explicitly splits the transformation into cardinality paths
    """
    def __init__(self, in_channels, out_channels, cardinality=32, base_width=4, stride=1):
        super().__init__()
        
        # Calculate group width D
        D = int(math.floor(out_channels * (base_width / 64)))
        
        self.cardinality = cardinality
        self.D = D
        
        # Create cardinality parallel paths
        self.paths = nn.ModuleList()
        for i in range(cardinality):
            path = nn.Sequential(
                nn.Conv2d(in_channels, D, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(D),
                nn.ReLU(inplace=True),
                nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(D),
                nn.ReLU(inplace=True)
            )
            self.paths.append(path)
        
        # Final 1x1 convolution
        self.conv_aggregate = nn.Conv2d(D * cardinality, out_channels, 
                                       kernel_size=1, stride=1, bias=False)
        self.bn_aggregate = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        # Split and transform
        path_outputs = []
        for path in self.paths:
            path_outputs.append(path(x))
        
        # Concatenate along channel dimension
        out = torch.cat(path_outputs, dim=1)
        
        # Merge
        out = self.conv_aggregate(out)
        out = self.bn_aggregate(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class LowLightEnhancementBlock(nn.Module):
    """
    Low-light Image Enhancement Block
    Combines Swin Transformer and ResNeXt blocks for image enhancement
    
    Args:
        in_channels: number of input image channels (default: 3 for RGB)
        embed_dim: embedding dimension for features
        num_heads: number of attention heads in Swin blocks
        window_size: window size for Swin attention
        img_size: input image size (H, W)
        num_blocks: number of Swin+ResNeXt block pairs
        mlp_ratio: ratio of mlp hidden dim to embedding dim
        cardinality: cardinality for ResNeXt (number of groups)
        base_width: base width for ResNeXt
        drop_path_rate: stochastic depth rate
        resnext_type: 'B' or 'C' for different ResNeXt implementations
    """
    def __init__(self, in_channels=3, embed_dim=64, num_heads=8, window_size=7,
                 img_size=(224, 224), num_blocks=3, mlp_ratio=4., 
                 cardinality=32, base_width=4, drop_path_rate=0.1, 
                 resnext_type='C'):
        super().__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.img_size = img_size
        
        # Initial 1x1 convolution to project to embedding dimension
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1, bias=False)
        self.input_bn = nn.BatchNorm2d(embed_dim)
        self.input_relu = nn.ReLU(inplace=True)
        
        # Build N blocks of Swin Transformer + ResNeXt pairs
        self.swin_blocks = nn.ModuleList()
        self.resnext_blocks = nn.ModuleList()
        
        # Stochastic depth decay rule
        dpr = create_stochastic_depth_decay(num_blocks, drop_path_rate)
        
        # Select ResNeXt bottleneck type
        if resnext_type == 'B':
            print('Using ResNeXt Bottleneck Form B (split-transform-merge)')
            ResNeXtBlock = ResNeXtBottleneckB
        elif resnext_type == 'C':
            print('Using ResNeXt Bottleneck Form C (grouped convolution)')
            ResNeXtBlock = ResNeXtBottleneckC
        else:
            raise ValueError(f"Invalid resnext_type: {resnext_type}. Use 'B' or 'C'")
        
        for i in range(num_blocks):
            # Swin Transformer Block
            swin_block = SwinTransformerBlock(
                dim=embed_dim,
                input_resolution=img_size,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                drop=0.,
                attn_drop=0.,
                drop_path=dpr[i],
                norm_layer=nn.LayerNorm
            )
            self.swin_blocks.append(swin_block)
            
            # ResNeXt Block
            resnext_block = ResNeXtBlock(
                in_channels=embed_dim,
                out_channels=embed_dim,
                cardinality=cardinality,
                base_width=base_width,
                stride=1
            )
            self.resnext_blocks.append(resnext_block)
        
        # Final 3x3 convolution to produce output
        self.output_conv = nn.Conv2d(embed_dim, in_channels, kernel_size=3, 
                                    stride=1, padding=1, bias=False)
        self.output_bn = nn.BatchNorm2d(in_channels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following ResNeXt convention"""
        init_weights(self)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: input low-light image [B, C, H, W]
        Returns:
            enhanced image [B, C, H, W]
        """
        # Save input for residual connection
        identity = x
        B, C, H, W = x.shape
        
        # Initial projection: 1x1 Conv
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = self.input_relu(x)  # [B, embed_dim, H, W]
        
        # Process through N blocks
        for i in range(self.num_blocks):
            # Save input for fusion
            block_input = x
            
            # Swin Transformer path
            # Convert to sequence format [B, H*W, C]
            x_swin = x.flatten(2).transpose(1, 2)
            x_swin = self.swin_blocks[i](x_swin)
            # Convert back to image format [B, C, H, W]
            x_swin = x_swin.transpose(1, 2).view(B, self.embed_dim, H, W)
            
            # ResNeXt path
            x_res = self.resnext_blocks[i](block_input)
            
            # Fusion: element-wise addition
            x = x_swin + x_res
        
        # Final projection: 3x3 Conv
        x = self.output_conv(x)
        x = self.output_bn(x)  # [B, C, H, W]
        
        # Residual connection with input image
        out = x + identity
        
        return out


# Model configurations
def create_enhancement_model(config='base', img_size=None):
    """
    Create low-light enhancement model from configuration file
    
    Args:
        config: model configuration name ('tiny', 'small', 'base', 'large')
        img_size: input image size (optional, overrides config)
    """
    from utils.config import load_config
    
    # Load configuration from YAML file
    try:
        cfg_loader = load_config(config)
        cfg = cfg_loader.get('enhancement', {})
    except FileNotFoundError:
        # Fallback to hardcoded configs if YAML not found
        print(f"Warning: Config file for '{config}' not found, using fallback")
        configs = {
            'tiny': {
                'embed_dim': 48, 'num_heads': 6, 'num_blocks': 2,
                'window_size': 7, 'cardinality': 16, 'base_width': 4
            },
            'small': {
                'embed_dim': 64, 'num_heads': 8, 'num_blocks': 3,
                'window_size': 7, 'cardinality': 32, 'base_width': 4
            },
            'base': {
                'embed_dim': 96, 'num_heads': 12, 'num_blocks': 4,
                'window_size': 7, 'cardinality': 32, 'base_width': 4
            },
            'large': {
                'embed_dim': 128, 'num_heads': 16, 'num_blocks': 6,
                'window_size': 7, 'cardinality': 32, 'base_width': 4
            }
        }
        cfg = configs.get(config, configs['base'])
    
    # Override img_size if provided
    if img_size is not None:
        cfg['img_size'] = img_size
    
    model = LowLightEnhancementBlock(
        in_channels=3,
        embed_dim=cfg.get('embed_dim', 64),
        num_heads=cfg.get('num_heads', 8),
        window_size=cfg.get('window_size', 7),
        img_size=cfg.get('img_size', (224, 224)),
        num_blocks=cfg.get('num_blocks', 3),
        mlp_ratio=cfg.get('mlp_ratio', 4.0),
        cardinality=cfg.get('cardinality', 32),
        base_width=cfg.get('base_width', 4),
        drop_path_rate=cfg.get('drop_path_rate', 0.1),
        resnext_type=cfg.get('resnext_type', 'C')
    )
    
    return model
