import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional

from .common import ConvBNSiLU
from .head import DetectionHead


class DetectionModule(nn.Module):
    
    def __init__(
        self,
        num_classes: int = 80,
        in_channels: List[int] = [64, 128, 256], # Features from Transformation Module
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # 3 Conv blocks as shown in the diagram (one for each scale)
        self.conv_p3 = ConvBNSiLU(in_channels[0], in_channels[0], 3)
        self.conv_p4 = ConvBNSiLU(in_channels[1], in_channels[1], 3)
        self.conv_p5 = ConvBNSiLU(in_channels[2], in_channels[2], 3)
        
        # Detection Head for predicting boxes and classes from the 3 scales
        self.head = DetectionHead(
            num_classes=num_classes,
            in_channels=in_channels,
        )
        
    def _make_anchors(
        self,
        features: List[torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate anchor points for each scale."""
        anchors = []
        strides = []
        
        for i, feat in enumerate(features):
            _, _, H, W = feat.shape
            # Standard strides for 3-scale detection
            stride = 8 * (2 ** i)
            
            # Generate grid centers
            sy, sx = torch.meshgrid(
                torch.arange(H, device=device) + 0.5,
                torch.arange(W, device=device) + 0.5,
                indexing='ij'
            )
            anchor = torch.stack([sx, sy], dim=-1).view(-1, 2) * stride
            anchors.append(anchor)
            strides.append(torch.full((H * W,), stride, device=device))
        
        return torch.cat(anchors, dim=0), torch.cat(strides, dim=0)
    
    def forward(
        self,
        features: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [P3, P4, P5] list from Transformation Module
            
        Returns:
            outputs: Detection outputs
        """
        p3, p4, p5 = features
        
        # Pass through the 3 Conv blocks shown in the diagram
        feat_list = [
            self.conv_p3(p3),
            self.conv_p4(p4),
            self.conv_p5(p5),
        ]
        
        # Generate anchors
        anchors, strides = self._make_anchors(feat_list, feat_list[0].device)
        
        # Head
        outputs = self.head(feat_list, training=self.training)
        outputs['anchors'] = anchors
        outputs['strides_tensor'] = strides
        
        return outputs
    
    def postprocess(
        self,
        outputs: Dict[str, torch.Tensor],
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
    ) -> List[torch.Tensor]:
        """Decodes boxes and filters by confidence."""
        cls_scores = outputs['one2one_cls'].sigmoid()
        reg_outputs = outputs['one2one_reg']
        anchors = outputs['anchors']
        strides = outputs['strides_tensor']
        
        # Decode boxes to xyxy
        boxes = self.head.decode_boxes(reg_outputs, anchors, strides)
        
        B = cls_scores.shape[0]
        results = []
        
        for b in range(B):
            scores, class_ids = cls_scores[b].max(dim=-1)
            mask = scores > conf_thres
            
            box = boxes[b][mask]
            score = scores[mask]
            cls = class_ids[mask].float()
            
            if len(score) > max_det:
                topk = score.topk(max_det).indices
                box = box[topk]
                score = score[topk]
                cls = cls[topk]
            
            det = torch.cat([box, score.unsqueeze(-1), cls.unsqueeze(-1)], dim=-1)
            results.append(det)
        
        return results
