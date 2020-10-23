import torch.nn.functional as F
import torch.nn as nn
import torch

class MiniUNet(nn.Module):
    def __init__(self, input_channels, num_classes, bilinear=False):
        super(MiniUNet, self).__init__()
        self.input_channels = input_channels
        self.n_classes = num_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(input_channels, 64)
        self.down1 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = nn.Conv2d(64, self.n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x = self.up4(x2, x1)
        logits = self.outc(x1)
        return logits

class DoubleConv(nn.Module):
    def __init__(self, input_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(input_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(input_channels, out_channels)
        )
    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, input_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(input_channels, out_channels, input_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(input_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffY // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)