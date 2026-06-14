"""torchvision pretrained 백본 (plan.md §6) — 3채널 mel 입력, 2-class 헤드 교체.

- MobileNetV3-Small : 경량 프리필터 후보(주력)
- EfficientNet-B0   : 소형 비교군
- ConvNeXt-Tiny     : 성능 상한 비교군
"""
import torch.nn as nn
from torchvision import models as tvm


def mobilenet_v3_small(n_classes=2):
    m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, n_classes)
    return m


def efficientnet_b0(n_classes=2):
    m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, n_classes)
    return m


def convnext_tiny(n_classes=2):
    m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, n_classes)
    return m
