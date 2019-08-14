import torch
import torch.nn as nn

from torchvision import models


class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x


class MRNetSqueeze(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.squeezenet1_0(pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x


class MRNetAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Linear(256, 1)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)  # (seq_len, n_feat)
        a = torch.softmax(self.attention(x), dim=0)  # (1, seq_len)
        x = torch.sum(a.view(-1, 1) * x, dim=0, keepdim=True)
        x = self.classifier(x)
        return x


class MRNetSqueezeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.squeezenet1_0(pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Linear(512, 1)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)  # (seq_len, n_feat)
        a = torch.softmax(self.attention(x), dim=0)  # (1, seq_len)
        x = torch.sum(a.view(-1, 1) * x, dim=0, keepdim=True)
        x = self.classifier(x)
        return x
