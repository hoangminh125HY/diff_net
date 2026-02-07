import torch
import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    """
    Clean ResNet backbone for detection
    Return C3 C4 C5 feature maps
    """

    def __init__(self, name="resnet18", pretrained=True, in_channels=6):
        super().__init__()

        # ===== load resnet =====
        if name == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.out_channels = [128, 256, 512]

        elif name == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.out_channels = [512, 1024, 2048]

        else:
            raise ValueError("Only support resnet18 / resnet50")

        # ===== FIX CONV1 FOR 6 CHANNEL INPUT =====
        old_conv = net.conv1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if pretrained:
            with torch.no_grad():
                # old weight: [64, 3, 7, 7]
                self.conv1.weight[:, :3] = old_conv.weight

                # copy RGB mean to extra channels
                mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                self.conv1.weight[:, 3:] = mean_weight

        # ===== keep rest =====
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool

        self.layer1 = net.layer1
        self.layer2 = net.layer2  # C3
        self.layer3 = net.layer3  # C4
        self.layer4 = net.layer4  # C5

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return [c3, c4, c5]
