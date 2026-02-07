"""
Detection Head for Detection Module
Based on: YOLOv10 with dual assignments for NMS-free inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import math

from .common import ConvBNSiLU


class DWConv(nn.Module):
    """Depthwise Separable Convolution."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, p: int = None, act: bool = True):
        super().__init__()
        self.dw = ConvBNSiLU(in_ch, in_ch, k, s, p, groups=in_ch)
        self.pw = ConvBNSiLU(in_ch, out_ch, 1, 1, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class DFL(nn.Module):
    """Distribution Focal Loss for box regression."""
    
    def __init__(self, c1: int = 16):
        super().__init__()
        self.c1 = c1
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class DetectionHead(nn.Module):
    """
    Decoupled Detection Head with dual assignments.
    
    For each scale, outputs:
    - Classification scores (num_classes)
    - Box regression (4 values: x, y, w, h)
    
    Uses dual assignment strategy:
    - One-to-many: Rich supervision during training
    - One-to-one: NMS-free inference
    
    Args:
        num_classes: Number of object classes
        in_channels: Channels for each scale [C3, C4, C5]
        reg_max: Maximum value for DFL
    """
    
    def __init__(
        self,
        num_classes: int = 80,
        in_channels: List[int] = [256, 512, 1024],
        reg_max: int = 16,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.num_scales = len(in_channels)
        
        # Dual Assignment Heads: one-to-many and one-to-one
        self.cv2 = nn.ModuleList() # classification one-to-many
        self.cv3 = nn.ModuleList() # regression one-to-many
        self.cv4 = nn.ModuleList() # classification one-to-one
        self.cv5 = nn.ModuleList() # regression one-to-one
        
        for c in in_channels:
            # One-to-many branch (Training)
            self.cv2.append(nn.Sequential(
                DWConv(c, c, 3),
                DWConv(c, c, 3),
                nn.Conv2d(c, num_classes, 1)
            ))
            self.cv3.append(nn.Sequential(
                ConvBNSiLU(c, c, 3),
                ConvBNSiLU(c, c, 3),
                nn.Conv2d(c, 4 * reg_max, 1)
            ))
            
            # One-to-one branch (Inference, NMS-free)
            self.cv4.append(nn.Sequential(
                DWConv(c, c, 3),
                DWConv(c, c, 3),
                nn.Conv2d(c, num_classes, 1)
            ))
            self.cv5.append(nn.Sequential(
                ConvBNSiLU(c, c, 3),
                ConvBNSiLU(c, c, 3),
                nn.Conv2d(c, 4 * reg_max, 1)
            ))
        
        # DFL for box regression
        self.dfl = DFL(reg_max)
        
        self._initialize_biases()
    
    def _initialize_biases(self):
        """Initialize biases for stable training."""
        for m in [self.cv2, self.cv4]: # Cls heads
            for head in m:
                conv = head[-1]
                b = conv.bias.view(-1)
                b.data.fill_(-math.log((1 - 0.01) / 0.01))
    
    def forward(
        self,
        features: List[torch.Tensor],
        training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [P3, P4, P5] from neck
            training: Whether in training mode
            
        Returns:
            outputs: Dict containing predictions
                - cls_scores: (B, num_anchors, num_classes)
                - box_regs: (B, num_anchors, 4)
                - one2one_cls: One-to-one classification (inference)
                - one2one_reg: One-to-one regression (inference)
        """
        outputs = {
            "cls_scores": [],
            "box_regs": [],
            "one2one_cls": [],
            "one2one_reg": [],
            "strides": [],
        }
        
        for i, feat in enumerate(features):
            B, C, H, W = feat.shape
            stride = 8 * (2 ** i)
            
            # 1. One-to-many predictions
            res_cls = self.cv2[i](feat)
            res_reg = self.cv3[i](feat)
            
            cls_out = res_cls.view(B, self.num_classes, -1).permute(0, 2, 1)
            reg_out = res_reg.view(B, 4 * self.reg_max, -1).permute(0, 2, 1)
            
            outputs["cls_scores"].append(cls_out)
            outputs["box_regs"].append(reg_out)
            outputs["strides"].append(stride)
            
            # 2. One-to-one predictions (Inference)
            o2o_cls = self.cv4[i](feat)
            o2o_reg = self.cv5[i](feat)
            
            o2o_cls_out = o2o_cls.view(B, self.num_classes, -1).permute(0, 2, 1)
            o2o_reg_out = o2o_reg.view(B, 4 * self.reg_max, -1).permute(0, 2, 1)
            
            outputs["one2one_cls"].append(o2o_cls_out)
            outputs["one2one_reg"].append(o2o_reg_out)
        
        # Concatenate across scales
        outputs["cls_scores"] = torch.cat(outputs["cls_scores"], dim=1)
        outputs["box_regs"] = torch.cat(outputs["box_regs"], dim=1)
        outputs["one2one_cls"] = torch.cat(outputs["one2one_cls"], dim=1)
        outputs["one2one_reg"] = torch.cat(outputs["one2one_reg"], dim=1)
        
        return outputs
    
    def decode_boxes(
        self,
        reg_outputs: torch.Tensor,
        anchors: torch.Tensor,
        strides: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode box predictions to absolute coordinates.
        
        Args:
            reg_outputs: (B, num_anchors, 4*reg_max)
            anchors: (num_anchors, 2) anchor centers
            strides: (num_anchors,) stride for each anchor
            
        Returns:
            boxes: (B, num_anchors, 4) in xyxy format
        """
        B = reg_outputs.shape[0]
        
        # Apply DFL
        reg = self.dfl(reg_outputs.permute(0, 2, 1)).permute(0, 2, 1)  # (B, num_anchors, 4)
        
        # Decode ltrb to xyxy
        lt = reg[..., :2]
        rb = reg[..., 2:]
        
        anchors = anchors.unsqueeze(0).expand(B, -1, -1)
        strides = strides.unsqueeze(0).unsqueeze(-1).expand(B, -1, 2)
        
        x1y1 = anchors - lt * strides
        x2y2 = anchors + rb * strides
        
        boxes = torch.cat([x1y1, x2y2], dim=-1)
        return boxes
