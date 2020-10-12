import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class OCDNet(nn.Module):
    def __init__(self, input_channels, num_classes, pretrained=True):
        super(OCDNet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        assert self.input_channels == 3, 'The input of vgg16 should be 3 channels'

        vgg16 = models.vgg16(pretrained=pretrained)

        self.features = vgg16.features

        self.mid_conv = nn.Conv2d(512, 4096, 7, stride=1, padding=3)
        nn.init.xavier_uniform_(self.mid_conv.weight)
        nn.init.constant_(self.mid_conv.bias, 0.1)

        self.deconv6 = nn.Conv2d(4096, 512, 1, stride=1, padding=0)
        nn.init.xavier_uniform_(self.deconv6.weight)
        nn.init.constant_(self.deconv6.bias, 0.1)

        self.up5 = Up(512, 512)
        self.up4 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up1 = Up(64, 32)

        self.pred = nn.Conv2d(32, self.num_classes, 5, stride=1, padding=2)
        nn.init.xavier_uniform_(self.pred.weight)
        nn.init.constant_(self.pred.bias, 0.1)
    
    def forward(self, x):
        x = self.features(x)

        x = F.relu(self.mid_conv(x))
        x = F.dropout(x, 0.5)

        x = F.relu(self.deconv6(x))
        x = F.dropout(x, 0.5)

        x = self.up5(x)
        x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)

        x = self.pred(x)

        return x

class Up(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(Up, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_channels, out_channels, 2, stride=2, padding=0)
        self.conv = nn.Conv2d(out_channels, out_channels, 5, stride=1, padding=2)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.1)

    def forward(self, x):
        x = F.relu(self.deconv(x))
        x = F.relu(self.conv(x))
        x = F.dropout(x, 0.5)

        return x