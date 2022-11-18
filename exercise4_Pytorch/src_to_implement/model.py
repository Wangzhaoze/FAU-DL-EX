
import torch
import torch.nn as nn



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, (1, 1), stride=stride)
        self.bn1_1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x0 = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x1 = self.bn1_1(self.conv1x1(x0))
        x += x1
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, (7, 7), (2, 2))
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(3, stride=2)
        self.ResBlock1 = ResBlock(64, 64, 1)
        self.ResBlock2 = ResBlock(64, 128, 2)
        self.ResBlock3 = ResBlock(128, 256, 2)
        self.ResBlock4 = ResBlock(256, 512, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()
        self.dropout1 = nn.Dropout(0.7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.ResBlock1(x)
        x = self.ResBlock2(x)
        x = self.ResBlock3(x)
        x = self.ResBlock4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
