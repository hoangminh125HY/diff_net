import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from .modules.enhancement import LowLightEnhancementBlock
from .modules.depth import DepthModuleWithProjection
from .modules.evolution import FeatureEvolutionSubnetwork
from .detection.detection_module import DetectionModule as LightweightDetectionSubnetwork


class DiffNet(nn.Module):
    
    def __init__(
        self,
        num_classes: int = 80,
        img_size: Tuple[int, int] = (224, 224),
        enhancement_cfg: str = 'base',
        depth_cfg: Dict = None,
        evolution_cfg: Dict = None,
    ):
        super().__init__()
        
        self.img_size = img_size
        
        # 1. Enhancement Module
        self.enhancement = LowLightEnhancementBlock(
            img_size=img_size,
        )
        
        # 2. Depth Module
        self.depth_estimator = DepthModuleWithProjection(
            in_channels=3, 
            embed_dim=96,
            out_channels=3, 
            input_resolution=img_size,
        )
        
        # 3. Evolution Module
        if evolution_cfg is None:
            evolution_cfg = {
                'encoding_channels': [64, 128, 256],
                'decoding_channels': [128, 64, 32],
            }
            
        self.evolution = FeatureEvolutionSubnetwork(
            in_channels=6,
            backbone="resnet18"
        )

        self.detection_subnetwork = LightweightDetectionSubnetwork(
            num_classes=num_classes,
            in_channels=[64,128,256]   # vì đã align
        )
        
    def forward(
        self, 
        x: torch.Tensor,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> Dict[str, torch.Tensor]:
        # Step 1: Enhancement
        enhanced = self.enhancement(x)
        
        # Step 2: Depth Estimation
        depth = self.depth_estimator(enhanced)
        
        # Step 3: Feature Evolution (Transformation part)
        # Combine enhanced image and depth for transformation
        combined = torch.cat([enhanced, depth], dim=1)
        # recovered is the Evolution output, encoding_features are the Transformation output
        recovered, encoding_features, _ = self.evolution(combined)
        
        # Step 4: Lightweight Detection Subnetwork
        # Uses Transformation output (encoding_features) to predict objects
        raw_outputs = self.detection_subnetwork(features=encoding_features)
        
        # Post-processing for inference
        if not self.training:
            detections = self.detection_subnetwork.postprocess(
                raw_outputs, 
                conf_thres=conf_thres, 
                iou_thres=iou_thres
            )
        else:
            detections = None
            
        return {
            'detections': detections,
            'raw_detect': raw_outputs,
            'enhanced': enhanced,
            'depth': depth,
            'recovered': recovered
        }


def create_model(model_config: str = 'base', num_classes: int = 80, img_size: Tuple[int, int] = (224, 224)):
    """Factory function to create a DiffNet model."""
    # Note: In a real implementation, this would load parameters from YAML
    return DiffNet(num_classes=num_classes, img_size=img_size)

# Alias for compatibility with README
ObjectDetectionModel = DiffNet