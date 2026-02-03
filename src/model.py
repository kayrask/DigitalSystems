import torch.nn as nn
from torchvision import models

def build_resnet18_multilabel(num_classes: int, pretrained: bool=True):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m
