import torch
import torch.nn as nn

from torchvision import models


class MRNet(nn.Module):
    def __init__(self, pretrained=pretrained):
        super().__init__()
        self.model = models.alexnet(pretrained=pretrained)
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
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = models.vgg11_bn(pretrained=pretrained)
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
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = models.vgg11_bn(pretrained=pretrained)

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
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = models.densenet121(pretrained=pretrained)
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
    def __init__(self, pretrained=pretrained):
        super().__init__()
        self.model = models.squeezenet1_0(pretrained=pretrained)
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
    def __init__(self, pretrained=True):
        super().__init__()
        res = models.resnet18(pretrained=pretrained)

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
    def __init__(self, pretrained=True):
        super().__init__()
        res = models.resnet18(pretrained=pretrained)

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
    def __init__(self, pretrained=True):
        super().__init__()
        res = models.resnet18(pretrained=pretrained)

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
    def __init__(self, pretrained=True):
        super().__init__()
        res = models.resnet18(pretrained=pretrained)

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
    def __init__(self, pretrained=True):
        super().__init__()
        res = models.resnet18(pretrained=pretrained)

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
    def __init__(self, pretrained=True):
        super().__init__()
        res = models.resnet18(pretrained=pretrained)

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
    def __init__(self, pretrained=True):
        super().__init__()
        res = models.resnet18(pretrained=pretrained)

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
    def __init__(self, pretrained=True):
        super().__init__()
        res = models.resnet18(pretrained=pretrained)

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
    def __init__(self, pretrained=True):
        super().__init__()
        res = models.resnet18(pretrained=pretrained)

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


class MRNetResCut1_5(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        res = models.resnet18(pretrained=pretrained)

        # skip avg pool and fc
        self.model = nn.Sequential(*list(res.children())[:-4])
        self.model2 = nn.Sequential(res.layer3[0])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model(x)
        x = self.model2(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x


class MRNetResCut_5(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        res = models.resnet18(pretrained=pretrained)

        # skip avg pool and fc
        self.model = nn.Sequential(*list(res.children())[:-3])
        self.model2 = nn.Sequential(res.layer4[0])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model(x)
        x = self.model2(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x


class MRNetLstm(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = models.alexnet(pretrained=pretrained)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.hidden_size = 152
        self.h0 = torch.randn(1, 1, self.hidden_size, requires_grad=True)
        self.c0 = torch.zeros(1, 1, self.hidden_size)
        try:
            self.h0 = self.h0.cuda()
            self.c0 = self.c0.cuda()
        except AssertionError:
            pass
        self.lstm = nn.LSTM(256, 152)
        self.classifier = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), 1, -1)  # (seq_len, "batch", n_feat)
        _, hc = self.lstm(x, (self.h0, self.c0))
        x, _ = hc
        x = self.classifier(x.view(1, self.hidden_size))
        return x


class MRNetBiLstm(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = models.alexnet(pretrained=pretrained)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.hidden_size = 152
        self.h0 = torch.randn(2, 1, self.hidden_size, requires_grad=True)
        self.c0 = torch.zeros(2, 1, self.hidden_size)
        try:
            self.h0 = self.h0.cuda()
            self.c0 = self.c0.cuda()
        except AssertionError:
            pass
        self.lstm = nn.LSTM(256, 152, bidirectional=True)
        self.classifier = nn.Linear(2 * self.hidden_size, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), 1, -1)  # (seq_len, "batch", n_feat)
        _, hc = self.lstm(x, (self.h0, self.c0))
        x, _ = hc
        x = self.classifier(x.view(1, 2 * self.hidden_size))
        return x


class MRNetAttention(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = models.alexnet(pretrained=pretrained)
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
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = models.squeezenet1_0(pretrained=pretrained)
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


class MRNetResFixedBN(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        res = models.resnet18(pretrained=pretrained)

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

        # resnet is fixed, don't update the batch norm
        self.model.eval()

        x = self.model(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x


# class MRNetRes7(nn.Module):
#     def __init__(self, pretrained=True):
#         super().__init__()
#         res = models.resnet18(pretrained=pretrained)

#         # skip avg pool and fc
#         self.model = nn.Sequential(*list(res.children())[:-2])

#         finetune_above = 7

#         for i, child in enumerate(self.model.children()):
#             for param in child.parameters():
#                 param.requires_grad = i >= finetune_above

#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(512, 1)

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
#         x = self.model(x)
#         x = self.gap(x).view(x.size(0), -1)
#         x = torch.max(x, 0, keepdim=True)[0]
#         x = self.classifier(x)
#         return x


# class MRNetRes7_1(nn.Module):
#     def __init__(self, pretrained=True):
#         super().__init__()
#         res = models.resnet18(pretrained=pretrained)

#         # skip avg pool and fc
#         self.model = nn.Sequential(*list(res.children())[:-2])

#         for i, child in enumerate(self.model.children()):
#             if i == 7:
#                 for j, subchild in enumerate(child.children()):
#                     if j == 0:
#                         for param in subchild.parameters():
#                             param.requires_grad = False
#                     else:
#                         for param in subchild.parameters():
#                             param.requires_grad = True
#             else:
#                 for param in child.parameters():
#                     param.requires_grad = False

#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(512, 1)

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
#         x = self.model(x)
#         x = self.gap(x).view(x.size(0), -1)
#         x = torch.max(x, 0, keepdim=True)[0]
#         x = self.classifier(x)
#         return x


# class MRNetRes7_1_conv2(nn.Module):
#     def __init__(self, pretrained=True):
#         super().__init__()
#         res = models.resnet18(pretrained=pretrained)

#         # skip avg pool and fc
#         self.model = nn.Sequential(*list(res.children())[:-2])

#         for i, child in enumerate(self.model.children()):
#             if i == 7:
#                 for j, subchild in enumerate(child.children()):
#                     if j == 0:
#                         for param in subchild.parameters():
#                             param.requires_grad = False
#                     else:
#                         for param in subchild.parameters():
#                             param.requires_grad = False

#                         for param in subchild.conv2.parameters():
#                             param.requires_grad = True
#                         for param in subchild.bn2.parameters():
#                             param.requires_grad = True
#             else:
#                 for param in child.parameters():
#                     param.requires_grad = False

#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(512, 1)

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
#         x = self.model(x)
#         x = self.gap(x).view(x.size(0), -1)
#         x = torch.max(x, 0, keepdim=True)[0]
#         x = self.classifier(x)
#         return x


# class MRNetRes7Dropout(nn.Module):
#     def __init__(self, pretrained=True):
#         super().__init__()
#         res = models.resnet18(pretrained=pretrained)

#         # skip avg pool and fc
#         self.model = nn.Sequential(*list(res.children())[:-2])

#         finetune_above = 7

#         for i, child in enumerate(self.model.children()):
#             for param in child.parameters():
#                 param.requires_grad = i >= finetune_above

#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.dropout = nn.Dropout()
#         self.classifier = nn.Linear(512, 1)

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
#         x = self.model(x)
#         x = self.gap(x).view(x.size(0), -1)
#         x = torch.max(x, 0, keepdim=True)[0]
#         x = self.dropout(x)
#         x = self.classifier(x)
#         return x


# class MRNetRes7Dropout75(nn.Module):
#     def __init__(self, pretrained=True):
#         super().__init__()
#         res = models.resnet18(pretrained=pretrained)

#         # skip avg pool and fc
#         self.model = nn.Sequential(*list(res.children())[:-2])

#         finetune_above = 7

#         for i, child in enumerate(self.model.children()):
#             for param in child.parameters():
#                 param.requires_grad = i >= finetune_above

#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.dropout = nn.Dropout(p=0.75)
#         self.classifier = nn.Linear(512, 1)

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
#         x = self.model(x)
#         x = self.gap(x).view(x.size(0), -1)
#         x = torch.max(x, 0, keepdim=True)[0]
#         x = self.dropout(x)
#         x = self.classifier(x)
#         return x


# class MRNetResCut1(nn.Module):
#     def __init__(self, pretrained=True):
#         super().__init__()
#         res = models.resnet18(pretrained=pretrained)

#         # skip avg pool and fc
#         self.model = nn.Sequential(*list(res.children())[:-3])
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(256, 1)

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
#         x = self.model(x)
#         x = self.gap(x).view(x.size(0), -1)
#         x = torch.max(x, 0, keepdim=True)[0]
#         x = self.classifier(x)
#         return x


# class MRNetResCut2(nn.Module):
#     def __init__(self, pretrained=True):
#         super().__init__()
#         res = models.resnet18(pretrained=pretrained)

#         # skip avg pool and fc
#         self.model = nn.Sequential(*list(res.children())[:-4])
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(128, 1)

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
#         x = self.model(x)
#         x = self.gap(x).view(x.size(0), -1)
#         x = torch.max(x, 0, keepdim=True)[0]
#         x = self.classifier(x)
#         return x


# class MRNetResCut1_5(nn.Module):
#     def __init__(self, pretrained=True):
#         super().__init__()
#         res = models.resnet18(pretrained=pretrained)

#         # skip avg pool and fc
#         self.model = nn.Sequential(*list(res.children())[:-4])
#         self.model2 = nn.Sequential(res.layer3[0])
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(256, 1)

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
#         x = self.model(x)
#         x = self.model2(x)
#         x = self.gap(x).view(x.size(0), -1)
#         x = torch.max(x, 0, keepdim=True)[0]
#         x = self.classifier(x)
#         return x


# class MRNetResCut_5(nn.Module):
#     def __init__(self, pretrained=True):
#         super().__init__()
#         res = models.resnet18(pretrained=pretrained)

#         # skip avg pool and fc
#         self.model = nn.Sequential(*list(res.children())[:-3])
#         self.model2 = nn.Sequential(res.layer4[0])
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(512, 1)

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
#         x = self.model(x)
#         x = self.model2(x)
#         x = self.gap(x).view(x.size(0), -1)
#         x = torch.max(x, 0, keepdim=True)[0]
#         x = self.classifier(x)
#         return x
