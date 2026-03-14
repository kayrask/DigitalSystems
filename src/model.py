import torch.nn as nn
from torchvision import models


def build_resnet18_multilabel(num_classes: int, pretrained: bool = True):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def build_efficientnet_b3_multilabel(num_classes: int, pretrained: bool = True):
    """EfficientNet-B3 multilabel head — higher accuracy than ResNet18 at ~12M params.
    Train with: python scripts/train_multilabel.py --arch efficientnet_b3 --batch_size 16
    """
    m = models.efficientnet_b3(
        weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
    )
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m
