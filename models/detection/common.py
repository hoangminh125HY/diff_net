import torch
import torch.nn as nn

class ConvBNSiLU(nn.Module):
    """Standard Conv + BatchNorm + SiLU block."""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """Standard bottleneck block."""
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvBNSiLU(in_channels, hidden_channels, 1)
        self.cv2 = ConvBNSiLU(hidden_channels, out_channels, 3)
        self.add = shortcut and in_channels == out_channels
    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out

class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""
    def __init__(self, in_channels, out_channels, num_blocks=1, shortcut=True, expansion=0.5):
        super().__init__()
        self.hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvBNSiLU(in_channels, 2 * self.hidden_channels, 1)
        self.cv2 = ConvBNSiLU((2 + num_blocks) * self.hidden_channels, out_channels, 1)
        self.blocks = nn.ModuleList([Bottleneck(self.hidden_channels, self.hidden_channels, shortcut) for _ in range(num_blocks)])
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        for block in self.blocks:
            y.append(block(y[-1]))
        return self.cv2(torch.cat(y, dim=1))
