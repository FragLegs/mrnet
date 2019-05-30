import torch
import torch.nn as nn

from torchvision import models


class MRNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256 * 3, 1)

    def forward(self, x):
        a, c, s = x

        a = self.single_forward(a)
        c = self.single_forward(c)
        s = self.single_forward(s)

        acs = torch.cat((a, c ,s), 1)
        acs = self.classifier(acs)
        return acs

    def single_forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        return x


class MRNet3Sep(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_a = models.alexnet(pretrained=True)
        self.model_c = models.alexnet(pretrained=True)
        self.model_s = models.alexnet(pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256 * 3, 1)

    def forward(self, x):
        a, c, s = x

        a = self.single_forward(a, self.model_a)
        c = self.single_forward(c, self.model_c)
        s = self.single_forward(s, self.model_s)

        acs = torch.cat((a, c ,s), 1)
        acs = self.classifier(acs)
        return acs

    def single_forward(self, x, model):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = model.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        return x

class MRNetSqueeze3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.squeezenet1_0(pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512 * 3, 1)

    def forward(self, x):
        a, c, s = x

        a = self.single_forward(a)
        c = self.single_forward(c)
        s = self.single_forward(s)

        acs = torch.cat((a, c ,s), 1)
        acs = self.classifier(acs)
        return acs

    def single_forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        return x


class MRNetSqueeze3Sep(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_a = models.squeezenet1_0(pretrained=True)
        self.model_b = models.squeezenet1_0(pretrained=True)
        self.model_c = models.squeezenet1_0(pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512 * 3, 1)

    def forward(self, x):
        a, c, s = x

        a = self.single_forward(a, self.model_a)
        c = self.single_forward(c, self.model_c)
        s = self.single_forward(s, self.model_s)

        acs = torch.cat((a, c ,s), 1)
        acs = self.classifier(acs)
        return acs

    def single_forward(self, x, model):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = model.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        return x


class MRNetAttention3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Linear(256, 1)
        self.classifier = nn.Linear(256 * 3, 1)

    def forward(self, x):
        a, c, s = x

        a = self.single_forward(a)
        c = self.single_forward(c)
        s = self.single_forward(s)

        acs = torch.cat((a, c ,s), 1)
        acs = self.classifier(acs)
        return acs

    def single_forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)  # (seq_len, n_feat)
        a = torch.softmax(self.attention(x), dim=0)  # (1, seq_len)
        x = torch.sum(a.view(-1, 1) * x, dim=0, keepdim=True)
        return x


class MRNetAttention3Sep(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_a = models.alexnet(pretrained=True)
        self.model_c = models.alexnet(pretrained=True)
        self.model_s = models.alexnet(pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention_a = nn.Linear(256, 1)
        self.attention_c = nn.Linear(256, 1)
        self.attention_s = nn.Linear(256, 1)
        self.classifier = nn.Linear(256 * 3, 1)

    def forward(self, x):
        a, c, s = x

        a = self.single_forward(a, self.model_a, self.attention_a)
        c = self.single_forward(c, self.model_c, self.attention_c)
        s = self.single_forward(s, self.model_s, self.attention_s)

        acs = torch.cat((a, c ,s), 1)
        acs = self.classifier(acs)
        return acs

    def single_forward(self, x, model, attention):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = model.features(x)
        x = self.gap(x).view(x.size(0), -1)  # (seq_len, n_feat)
        a = torch.softmax(attention(x), dim=0)  # (1, seq_len)
        x = torch.sum(a.view(-1, 1) * x, dim=0, keepdim=True)
        return x


class MRNetSqueezeAttention3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.squeezenet1_0(pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Linear(512, 1)
        self.classifier = nn.Linear(512 * 3, 1)

    def forward(self, x):
        a, c, s = x

        a = self.single_forward(a)
        c = self.single_forward(c)
        s = self.single_forward(s)

        acs = torch.cat((a, c ,s), 1)
        acs = self.classifier(acs)
        return acs

    def single_forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)  # (seq_len, n_feat)
        a = torch.softmax(self.attention(x), dim=0)  # (1, seq_len)
        x = torch.sum(a.view(-1, 1) * x, dim=0, keepdim=True)
        return x


class MRNetSqueezeAttention3Sep(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_a = models.squeezenet1_0(pretrained=True)
        self.model_b = models.squeezenet1_0(pretrained=True)
        self.model_c = models.squeezenet1_0(pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention_a = nn.Linear(512, 1)
        self.attention_c = nn.Linear(512, 1)
        self.attention_s = nn.Linear(512, 1)
        self.classifier = nn.Linear(512 * 3, 1)

    def forward(self, x):
        a, c, s = x

        a = self.single_forward(a, self.model_a, self.attention_a)
        c = self.single_forward(c, self.model_c, self.attention_c)
        s = self.single_forward(s, self.model_s, self.attention_s)

        acs = torch.cat((a, c ,s), 1)
        acs = self.classifier(acs)
        return acs

    def single_forward(self, x, model, attention):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = model.features(x)
        x = self.gap(x).view(x.size(0), -1)  # (seq_len, n_feat)
        a = torch.softmax(attention(x), dim=0)  # (1, seq_len)
        x = torch.sum(a.view(-1, 1) * x, dim=0, keepdim=True)
        return x


class MRNetAttention3Hidden(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Linear(256, 1)
        self.hidden = nn.Linear(256 * 3, 256)
        self.activation = nn.LeakyReLU()
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        a, c, s = x

        a = self.single_forward(a)
        c = self.single_forward(c)
        s = self.single_forward(s)

        acs = torch.cat((a, c ,s), 1)
        acs = self.hidden(acs)
        acs = self.activation(acs)
        acs = self.classifier(acs)
        return acs

    def single_forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)  # (seq_len, n_feat)
        a = torch.softmax(self.attention(x), dim=0)  # (1, seq_len)
        x = torch.sum(a.view(-1, 1) * x, dim=0, keepdim=True)
        return x


class MRNetSqueezeAttention3Hidden(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.squeezenet1_0(pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Linear(512, 1)
        self.hidden = nn.Linear(512 * 3, 512)
        self.activation = nn.LeakyReLU()
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        a, c, s = x

        a = self.single_forward(a)
        c = self.single_forward(c)
        s = self.single_forward(s)

        acs = torch.cat((a, c ,s), 1)
        acs = self.hidden(acs)
        acs = self.activation(acs)
        acs = self.classifier(acs)
        return acs

    def single_forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)  # (seq_len, n_feat)
        a = torch.softmax(self.attention(x), dim=0)  # (1, seq_len)
        x = torch.sum(a.view(-1, 1) * x, dim=0, keepdim=True)
        return x

# class MRNetVGG(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = models.vgg11_bn(pretrained=True)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(512, 1)

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
#         x = self.model.features(x)
#         x = self.gap(x).view(x.size(0), -1)
#         x = torch.max(x, 0, keepdim=True)[0]
#         x = self.classifier(x)
#         return x


# class MRNetVGGFixed(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = models.vgg11_bn(pretrained=True)

#         for param in self.model.parameters():
#             param.requires_grad = False

#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(512, 1)

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
#         x = self.model.features(x)
#         x = self.gap(x).view(x.size(0), -1)
#         x = torch.max(x, 0, keepdim=True)[0]
#         x = self.classifier(x)
#         return x


# class MRNetDense(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = models.densenet121(pretrained=True)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(1024, 1)

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
#         x = self.model.features(x)
#         x = self.gap(x).view(x.size(0), -1)
#         x = torch.max(x, 0, keepdim=True)[0]
#         x = self.classifier(x)
#         return x


# class MRNetSqueeze(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = models.squeezenet1_0(pretrained=True)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(512, 1)

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
#         x = self.model.features(x)
#         x = self.gap(x).view(x.size(0), -1)
#         x = torch.max(x, 0, keepdim=True)[0]
#         x = self.classifier(x)
#         return x


# class MRNetRes(nn.Module):
#     def __init__(self):
#         super().__init__()
#         res = models.resnet18(pretrained=True)

#         # skip avg pool and fc
#         self.model = nn.Sequential(*list(res.children())[:-2])
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(512, 1)

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
#         x = self.model(x)
#         x = self.gap(x).view(x.size(0), -1)
#         x = torch.max(x, 0, keepdim=True)[0]
#         x = self.classifier(x)
#         return x


# class MRNetResFixed(nn.Module):
#     def __init__(self):
#         super().__init__()
#         res = models.resnet18(pretrained=True)

#         # skip avg pool and fc
#         self.model = nn.Sequential(*list(res.children())[:-2])

#         finetune_above = -1

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


# class MRNetRes7(nn.Module):
#     def __init__(self):
#         super().__init__()
#         res = models.resnet18(pretrained=True)

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
#     def __init__(self):
#         super().__init__()
#         res = models.resnet18(pretrained=True)

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
#     def __init__(self):
#         super().__init__()
#         res = models.resnet18(pretrained=True)

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
#     def __init__(self):
#         super().__init__()
#         res = models.resnet18(pretrained=True)

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
#     def __init__(self):
#         super().__init__()
#         res = models.resnet18(pretrained=True)

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
#     def __init__(self):
#         super().__init__()
#         res = models.resnet18(pretrained=True)

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
#     def __init__(self):
#         super().__init__()
#         res = models.resnet18(pretrained=True)

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
#     def __init__(self):
#         super().__init__()
#         res = models.resnet18(pretrained=True)

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
#     def __init__(self):
#         super().__init__()
#         res = models.resnet18(pretrained=True)

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


# class MRNetLstm(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = models.alexnet(pretrained=True)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.hidden_size = 152
#         self.h0 = torch.randn(1, 1, self.hidden_size, requires_grad=True)
#         self.c0 = torch.zeros(1, 1, self.hidden_size)
#         try:
#             self.h0 = self.h0.cuda()
#             self.c0 = self.c0.cuda()
#         except AssertionError:
#             pass
#         self.lstm = nn.LSTM(256, 152)
#         self.classifier = nn.Linear(self.hidden_size, 1)

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
#         x = self.model.features(x)
#         x = self.gap(x).view(x.size(0), 1, -1)  # (seq_len, "batch", n_feat)
#         _, hc = self.lstm(x, (self.h0, self.c0))
#         x, _ = hc
#         x = self.classifier(x.view(1, self.hidden_size))
#         return x


# class MRNetBiLstm(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = models.alexnet(pretrained=True)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.hidden_size = 152
#         self.h0 = torch.randn(2, 1, self.hidden_size, requires_grad=True)
#         self.c0 = torch.zeros(2, 1, self.hidden_size)
#         try:
#             self.h0 = self.h0.cuda()
#             self.c0 = self.c0.cuda()
#         except AssertionError:
#             pass
#         self.lstm = nn.LSTM(256, 152, bidirectional=True)
#         self.classifier = nn.Linear(2 * self.hidden_size, 1)

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
#         x = self.model.features(x)
#         x = self.gap(x).view(x.size(0), 1, -1)  # (seq_len, "batch", n_feat)
#         _, hc = self.lstm(x, (self.h0, self.c0))
#         x, _ = hc
#         x = self.classifier(x.view(1, 2 * self.hidden_size))
#         return x


# class MRNetAttention(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = models.alexnet(pretrained=True)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.attention = nn.Linear(256, 1)
#         self.classifier = nn.Linear(256, 1)

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
#         x = self.model.features(x)
#         x = self.gap(x).view(x.size(0), -1)  # (seq_len, n_feat)
#         a = torch.softmax(self.attention(x), dim=0)  # (1, seq_len)
#         x = torch.sum(a.view(-1, 1) * x, dim=0, keepdim=True)
#         x = self.classifier(x)
#         return x


# class MRNetResFixedBN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         res = models.resnet18(pretrained=True)

#         # skip avg pool and fc
#         self.model = nn.Sequential(*list(res.children())[:-2])

#         finetune_above = -1

#         for i, child in enumerate(self.model.children()):
#             for param in child.parameters():
#                 param.requires_grad = i >= finetune_above

#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(512, 1)

#     def forward(self, x):
#         x = torch.squeeze(x, dim=0)  # only batch size 1 supported

#         # resnet is fixed, don't update the batch norm
#         self.model.eval()

#         x = self.model(x)
#         x = self.gap(x).view(x.size(0), -1)
#         x = torch.max(x, 0, keepdim=True)[0]
#         x = self.classifier(x)
#         return x


# # class MRNetRes7(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         res = models.resnet18(pretrained=True)

# #         # skip avg pool and fc
# #         self.model = nn.Sequential(*list(res.children())[:-2])

# #         finetune_above = 7

# #         for i, child in enumerate(self.model.children()):
# #             for param in child.parameters():
# #                 param.requires_grad = i >= finetune_above

# #         self.gap = nn.AdaptiveAvgPool2d(1)
# #         self.classifier = nn.Linear(512, 1)

# #     def forward(self, x):
# #         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
# #         x = self.model(x)
# #         x = self.gap(x).view(x.size(0), -1)
# #         x = torch.max(x, 0, keepdim=True)[0]
# #         x = self.classifier(x)
# #         return x


# # class MRNetRes7_1(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         res = models.resnet18(pretrained=True)

# #         # skip avg pool and fc
# #         self.model = nn.Sequential(*list(res.children())[:-2])

# #         for i, child in enumerate(self.model.children()):
# #             if i == 7:
# #                 for j, subchild in enumerate(child.children()):
# #                     if j == 0:
# #                         for param in subchild.parameters():
# #                             param.requires_grad = False
# #                     else:
# #                         for param in subchild.parameters():
# #                             param.requires_grad = True
# #             else:
# #                 for param in child.parameters():
# #                     param.requires_grad = False

# #         self.gap = nn.AdaptiveAvgPool2d(1)
# #         self.classifier = nn.Linear(512, 1)

# #     def forward(self, x):
# #         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
# #         x = self.model(x)
# #         x = self.gap(x).view(x.size(0), -1)
# #         x = torch.max(x, 0, keepdim=True)[0]
# #         x = self.classifier(x)
# #         return x


# # class MRNetRes7_1_conv2(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         res = models.resnet18(pretrained=True)

# #         # skip avg pool and fc
# #         self.model = nn.Sequential(*list(res.children())[:-2])

# #         for i, child in enumerate(self.model.children()):
# #             if i == 7:
# #                 for j, subchild in enumerate(child.children()):
# #                     if j == 0:
# #                         for param in subchild.parameters():
# #                             param.requires_grad = False
# #                     else:
# #                         for param in subchild.parameters():
# #                             param.requires_grad = False

# #                         for param in subchild.conv2.parameters():
# #                             param.requires_grad = True
# #                         for param in subchild.bn2.parameters():
# #                             param.requires_grad = True
# #             else:
# #                 for param in child.parameters():
# #                     param.requires_grad = False

# #         self.gap = nn.AdaptiveAvgPool2d(1)
# #         self.classifier = nn.Linear(512, 1)

# #     def forward(self, x):
# #         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
# #         x = self.model(x)
# #         x = self.gap(x).view(x.size(0), -1)
# #         x = torch.max(x, 0, keepdim=True)[0]
# #         x = self.classifier(x)
# #         return x


# # class MRNetRes7Dropout(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         res = models.resnet18(pretrained=True)

# #         # skip avg pool and fc
# #         self.model = nn.Sequential(*list(res.children())[:-2])

# #         finetune_above = 7

# #         for i, child in enumerate(self.model.children()):
# #             for param in child.parameters():
# #                 param.requires_grad = i >= finetune_above

# #         self.gap = nn.AdaptiveAvgPool2d(1)
# #         self.dropout = nn.Dropout()
# #         self.classifier = nn.Linear(512, 1)

# #     def forward(self, x):
# #         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
# #         x = self.model(x)
# #         x = self.gap(x).view(x.size(0), -1)
# #         x = torch.max(x, 0, keepdim=True)[0]
# #         x = self.dropout(x)
# #         x = self.classifier(x)
# #         return x


# # class MRNetRes7Dropout75(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         res = models.resnet18(pretrained=True)

# #         # skip avg pool and fc
# #         self.model = nn.Sequential(*list(res.children())[:-2])

# #         finetune_above = 7

# #         for i, child in enumerate(self.model.children()):
# #             for param in child.parameters():
# #                 param.requires_grad = i >= finetune_above

# #         self.gap = nn.AdaptiveAvgPool2d(1)
# #         self.dropout = nn.Dropout(p=0.75)
# #         self.classifier = nn.Linear(512, 1)

# #     def forward(self, x):
# #         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
# #         x = self.model(x)
# #         x = self.gap(x).view(x.size(0), -1)
# #         x = torch.max(x, 0, keepdim=True)[0]
# #         x = self.dropout(x)
# #         x = self.classifier(x)
# #         return x


# # class MRNetResCut1(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         res = models.resnet18(pretrained=True)

# #         # skip avg pool and fc
# #         self.model = nn.Sequential(*list(res.children())[:-3])
# #         self.gap = nn.AdaptiveAvgPool2d(1)
# #         self.classifier = nn.Linear(256, 1)

# #     def forward(self, x):
# #         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
# #         x = self.model(x)
# #         x = self.gap(x).view(x.size(0), -1)
# #         x = torch.max(x, 0, keepdim=True)[0]
# #         x = self.classifier(x)
# #         return x


# # class MRNetResCut2(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         res = models.resnet18(pretrained=True)

# #         # skip avg pool and fc
# #         self.model = nn.Sequential(*list(res.children())[:-4])
# #         self.gap = nn.AdaptiveAvgPool2d(1)
# #         self.classifier = nn.Linear(128, 1)

# #     def forward(self, x):
# #         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
# #         x = self.model(x)
# #         x = self.gap(x).view(x.size(0), -1)
# #         x = torch.max(x, 0, keepdim=True)[0]
# #         x = self.classifier(x)
# #         return x


# # class MRNetResCut1_5(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         res = models.resnet18(pretrained=True)

# #         # skip avg pool and fc
# #         self.model = nn.Sequential(*list(res.children())[:-4])
# #         self.model2 = nn.Sequential(res.layer3[0])
# #         self.gap = nn.AdaptiveAvgPool2d(1)
# #         self.classifier = nn.Linear(256, 1)

# #     def forward(self, x):
# #         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
# #         x = self.model(x)
# #         x = self.model2(x)
# #         x = self.gap(x).view(x.size(0), -1)
# #         x = torch.max(x, 0, keepdim=True)[0]
# #         x = self.classifier(x)
# #         return x


# # class MRNetResCut_5(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         res = models.resnet18(pretrained=True)

# #         # skip avg pool and fc
# #         self.model = nn.Sequential(*list(res.children())[:-3])
# #         self.model2 = nn.Sequential(res.layer4[0])
# #         self.gap = nn.AdaptiveAvgPool2d(1)
# #         self.classifier = nn.Linear(512, 1)

# #     def forward(self, x):
# #         x = torch.squeeze(x, dim=0)  # only batch size 1 supported
# #         x = self.model(x)
# #         x = self.model2(x)
# #         x = self.gap(x).view(x.size(0), -1)
# #         x = torch.max(x, 0, keepdim=True)[0]
# #         x = self.classifier(x)
# #         return x
