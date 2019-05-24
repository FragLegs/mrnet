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


class MRNetVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vgg11_bn(pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x


class MRNetVGGFixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vgg11_bn(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x


class MRNetDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.densenet121(pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, 1)

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


class MRNetRes(nn.Module):
    def __init__(self):
        super().__init__()
        res = models.resnet18(pretrained=True)

        # skip avg pool and fc
        self.model = nn.Sequential(*list(res.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x


class MRNetResFixed(nn.Module):
    def __init__(self):
        super().__init__()
        res = models.resnet18(pretrained=True)

        # skip avg pool and fc
        self.model = nn.Sequential(*list(res.children())[:-2])

        finetune_above = -1

        for i, child in enumerate(self.model.children()):
            for param in child.parameters():
                param.requires_grad = i >= finetune_above

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x


class MRNetRes7(nn.Module):
    def __init__(self):
        super().__init__()
        res = models.resnet18(pretrained=True)

        # skip avg pool and fc
        self.model = nn.Sequential(*list(res.children())[:-2])

        finetune_above = 7

        for i, child in enumerate(self.model.children()):
            for param in child.parameters():
                param.requires_grad = i >= finetune_above

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x


class MRNetRes7_1(nn.Module):
    def __init__(self):
        super().__init__()
        res = models.resnet18(pretrained=True)

        # skip avg pool and fc
        self.model = nn.Sequential(*list(res.children())[:-2])

        for i, child in enumerate(self.model.children()):
            if i == 7:
                for j, subchild in enumerate(child.children()):
                    if j == 0:
                        for param in subchild.parameters():
                            param.requires_grad = False
                    else:
                        for param in subchild.parameters():
                            param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x


class MRNetRes7_1_conv2(nn.Module):
    def __init__(self):
        super().__init__()
        res = models.resnet18(pretrained=True)

        # skip avg pool and fc
        self.model = nn.Sequential(*list(res.children())[:-2])

        for i, child in enumerate(self.model.children()):
            if i == 7:
                for j, subchild in enumerate(child.children()):
                    if j == 0:
                        for param in subchild.parameters():
                            param.requires_grad = False
                    else:
                        for param in subchild.parameters():
                            param.requires_grad = False

                        for param in subchild.conv2.parameters():
                            param.requires_grad = True
                        for param in subchild.bn2.parameters():
                            param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x


class MRNetRes7Dropout(nn.Module):
    def __init__(self):
        super().__init__()
        res = models.resnet18(pretrained=True)

        # skip avg pool and fc
        self.model = nn.Sequential(*list(res.children())[:-2])

        finetune_above = 7

        for i, child in enumerate(self.model.children()):
            for param in child.parameters():
                param.requires_grad = i >= finetune_above

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class MRNetRes7Dropout75(nn.Module):
    def __init__(self):
        super().__init__()
        res = models.resnet18(pretrained=True)

        # skip avg pool and fc
        self.model = nn.Sequential(*list(res.children())[:-2])

        finetune_above = 7

        for i, child in enumerate(self.model.children()):
            for param in child.parameters():
                param.requires_grad = i >= finetune_above

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.75)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class MRNetResCut1(nn.Module):
    def __init__(self):
        super().__init__()
        res = models.resnet18(pretrained=True)

        # skip avg pool and fc
        self.model = nn.Sequential(*list(res.children())[:-3])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x


class MRNetResCut2(nn.Module):
    def __init__(self):
        super().__init__()
        res = models.resnet18(pretrained=True)

        # skip avg pool and fc
        self.model = nn.Sequential(*list(res.children())[:-4])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x
