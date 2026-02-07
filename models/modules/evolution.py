"""
Evolution Module - Feature Evolution Subnetwork
Combines Transformation Module (encoder) and Recovery Module (decoder)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones.resnet import ResNetBackbone

class TransformationModule(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=[64, 128, 256]):
        super(TransformationModule, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels[1]),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels[2]),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        features = []
        x1 = self.conv1(x)
        features.append(x1)
        x2 = self.conv2(x1)
        features.append(x2)
        x3 = self.conv3(x2)
        features.append(x3)
        return features, x3


class RecoveryModule(nn.Module):
    def __init__(self, encoded_channels=256, hidden_channels=[128, 64, 32], out_channels=3):
        super(RecoveryModule, self).__init__()
        
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(encoded_channels, hidden_channels[0], 
                              kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels[0], hidden_channels[1], 
                              kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_channels[1]),
            nn.ReLU(inplace=True)
        )
        
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels[1], hidden_channels[2], 
                              kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_channels[2]),
            nn.ReLU(inplace=True)
        )
        
        self.output_conv = nn.Conv2d(hidden_channels[2], out_channels, kernel_size=3, padding=1)
        
    def forward(self, encoded):
        reconstructions = []
        x1 = self.deconv1(encoded)
        reconstructions.append(x1)
        x2 = self.deconv2(x1)
        reconstructions.append(x2)
        x3 = self.deconv3(x2)
        reconstructions.append(x3)
        output = self.output_conv(x3)
        return reconstructions, output


class FeatureEvolutionSubnetwork(nn.Module):

    def __init__(self, in_channels=6,
                 encoding_channels=[64,128,256],
                 decoding_channels=[128,64,32],
                 out_channels=3,
                 backbone="resnet18"):

        super().__init__()

        # ===== Backbone =====
        self.transformation = ResNetBackbone(
            arch=backbone,
            return_indices=(1,2,3),  # C3 C4 C5
            freeze_indices=()
        )

        # lấy channels thật từ backbone
        self.out_channels = self.transformation.num_channels
        backbone_out = self.out_channels[-1]

        # ===== Feature align (C3,C4,C5 → 64,128,256 cho head cũ) =====
        self.align = nn.ModuleList([
            nn.Conv2d(c, o, 1)
            for c, o in zip(self.out_channels, encoding_channels)
        ])

        # ===== Decoder =====
        self.recovery = RecoveryModule(
            encoded_channels=backbone_out,
            hidden_channels=decoding_channels,
            out_channels=out_channels
        )

    def forward(self, x):

        # Backbone features
        features_dict = self.transformation(x)
        backbone_features = list(features_dict.values())

        # Align channels cho detection head
        encoding_features = [
            conv(f) for conv, f in zip(self.align, backbone_features)
        ]

        encoded = backbone_features[-1]

        # Decode ảnh
        decoding_features, output = self.recovery(encoded)

        return output, encoding_features, decoding_features

    # -----------------------------------------------------

    def encode(self, x):
        """Only encoder"""
        features_dict = self.transformation(x)
        backbone_features = list(features_dict.values())

        encoding_features = [
            conv(f) for conv, f in zip(self.align, backbone_features)
        ]

        return encoding_features[-1]

    # -----------------------------------------------------

    def decode(self, encoded):
        """Only decoder"""
        _, output = self.recovery(encoded)
        return output

