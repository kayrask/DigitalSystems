import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SpatialPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBNReLU(3, 64, k=7, s=2, p=3)
        self.conv2 = ConvBNReLU(64, 64, k=3, s=2, p=1)
        self.conv3 = ConvBNReLU(64, 64, k=3, s=2, p=1)
        self.conv_out = ConvBNReLU(64, 128, k=1, s=1, p=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv_out(x)
        return x


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, out_ch, k=3, s=1, p=1)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.conv(x)
        attn = self.attn(feat)
        return feat * attn


class FeatureFusionModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, out_ch, k=1, s=1, p=0)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // 4, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 4, out_ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, sp, cp):
        x = torch.cat([sp, cp], dim=1)
        x = self.conv(x)
        attn = self.attn(x)
        return x + x * attn


class ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=None)
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_avg = ConvBNReLU(512, 128, k=1, s=1, p=0)

    def forward(self, x):
        h, w = x.shape[2:]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feat16 = self.layer3(x)  # 1/16
        feat32 = self.layer4(feat16)  # 1/32

        avg = F.adaptive_avg_pool2d(feat32, 1)
        avg = self.conv_avg(avg)
        avg = F.interpolate(avg, size=feat32.shape[2:], mode="bilinear", align_corners=False)

        feat32 = self.arm32(feat32) + avg
        feat32_up = F.interpolate(feat32, size=feat16.shape[2:], mode="bilinear", align_corners=False)

        feat16 = self.arm16(feat16) + feat32_up
        feat16_up = F.interpolate(feat16, size=(h // 8, w // 8), mode="bilinear", align_corners=False)

        return feat16_up, feat32


class BiSeNet(nn.Module):
    def __init__(self, n_classes=19):
        super().__init__()
        self.spatial_path = SpatialPath()
        self.context_path = ContextPath()
        self.ffm = FeatureFusionModule(128 + 128, 256)

        self.conv_out = nn.Sequential(
            ConvBNReLU(256, 256, k=3, s=1, p=1),
            nn.Conv2d(256, n_classes, kernel_size=1, bias=True),
        )
        self.conv_out16 = nn.Sequential(
            ConvBNReLU(128, 128, k=3, s=1, p=1),
            nn.Conv2d(128, n_classes, kernel_size=1, bias=True),
        )
        self.conv_out32 = nn.Sequential(
            ConvBNReLU(128, 128, k=3, s=1, p=1),
            nn.Conv2d(128, n_classes, kernel_size=1, bias=True),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        feat_sp = self.spatial_path(x)
        feat_cp8, feat_cp32 = self.context_path(x)
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        out = self.conv_out(feat_fuse)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)

        out16 = self.conv_out16(feat_cp8)
        out16 = F.interpolate(out16, size=(h, w), mode="bilinear", align_corners=False)

        out32 = self.conv_out32(feat_cp32)
        out32 = F.interpolate(out32, size=(h, w), mode="bilinear", align_corners=False)

        return out, out16, out32
