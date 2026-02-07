"""
Depth Module - Swin Transformer based depth processing
Contains Swin Transformer blocks for hierarchical feature extraction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_
import math
from .common import Mlp, window_partition, window_reverse, WindowAttention, SwinTransformerBlock, init_weights, create_stochastic_depth_decay


class DepthModule(nn.Module):
    """
    Depth Module with sequential Swin Transformer blocks
    
    This module processes features through multiple Swin Transformer blocks
    to capture hierarchical depth information with self-attention mechanism.
    
    Args:
        dim (int): Feature dimension
        input_resolution (tuple[int]): Input resolution (H, W)
        depth (int): Number of Swin Transformer blocks
        num_heads (int): Number of attention heads
        window_size (int): Window size for attention
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim
        qkv_bias (bool): If True, add learnable bias to qkv
        qk_scale (float): Override default qk scale if set
        drop (float): Dropout rate
        attn_drop (float): Attention dropout rate
        drop_path_rate (float): Stochastic depth rate
        norm_layer: Normalization layer
    """
    
    def __init__(self, dim, input_resolution, depth=3, num_heads=8, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        
        # Build blocks with alternating regular and shifted window attention
        self.blocks = nn.ModuleList()
        
        # Stochastic depth decay rule
        dpr = create_stochastic_depth_decay(depth, drop_path_rate)
        
        for i in range(depth):
            block = SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            self.blocks.append(block)
        
        # Final normalization
        self.norm = norm_layer(dim)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input features of shape (B, H*W, C) or (B, C, H, W)
        Returns:
            Output features of same shape as input
        """
        # Handle both sequence and image format inputs
        if len(x.shape) == 4:  # (B, C, H, W)
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
            need_reshape = True
        else:  # (B, H*W, C)
            B, L, C = x.shape
            H, W = self.input_resolution
            need_reshape = False
        
        # Pass through Swin blocks sequentially
        for blk in self.blocks:
            x = blk(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Reshape back to image format if needed
        if need_reshape:
            x = x.transpose(1, 2).view(B, C, H, W)
        
        return x
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class DepthModuleWithProjection(nn.Module):
    """
    Depth Module with input/output projection layers
    
    This version includes convolutional projection layers before and after
    the Swin Transformer blocks for better feature transformation.
    
    Args:
        in_channels (int): Number of input channels
        embed_dim (int): Embedding dimension for transformers
        out_channels (int): Number of output channels
        input_resolution (tuple[int]): Input resolution (H, W)
        depth (int): Number of Swin blocks
        num_heads (int): Number of attention heads
        window_size (int): Window size
        mlp_ratio (float): MLP ratio
        drop_path_rate (float): Stochastic depth rate
    """
    
    def __init__(self, in_channels=64, embed_dim=96, out_channels=64,
                 input_resolution=(56, 56), depth=3, num_heads=8, window_size=7,
                 mlp_ratio=4., drop_path_rate=0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.input_resolution = input_resolution
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        
        # Depth module (Swin blocks)
        self.depth_module = DepthModule(
            dim=embed_dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # Residual connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        init_weights(self)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Output tensor of shape (B, out_channels, H, W)
        """
        identity = self.shortcut(x)
        
        # Project to embedding dimension
        x = self.input_proj(x)  # (B, embed_dim, H, W)
        
        # Process through depth module
        x = self.depth_module(x)  # (B, embed_dim, H, W)
        
        # Project back to output channels
        x = self.output_proj(x)  # (B, out_channels, H, W)
        
        # Residual connection
        x = x + identity
        
        return x
