"""모델 팩토리 — 이름 → (model, in_channels).

simple_cnn 만 1채널, 나머지 pretrained 백본은 3채널(mel 복제).
"""
from .backbones import convnext_tiny, efficientnet_b0, mobilenet_v3_small
from .simple_cnn import SimpleCNN

# 이름 -> (생성자, 입력 채널)
_REGISTRY = {
    "simple_cnn":        (lambda: SimpleCNN(in_ch=1), 1),
    "mobilenet_v3_small": (lambda: mobilenet_v3_small(), 3),
    "efficientnet_b0":   (lambda: efficientnet_b0(), 3),
    "convnext_tiny":     (lambda: convnext_tiny(), 3),
}

MODEL_NAMES = list(_REGISTRY)


def build_model(name):
    if name not in _REGISTRY:
        raise ValueError(f"unknown model '{name}'. choices: {MODEL_NAMES}")
    ctor, channels = _REGISTRY[name]
    return ctor(), channels
