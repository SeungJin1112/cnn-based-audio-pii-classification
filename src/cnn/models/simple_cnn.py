"""직접 설계한 경량 CNN baseline (plan.md §6)."""
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, in_ch=1, n_classes=2):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_ch, 16), block(16, 32), block(32, 64), block(64, 64),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))
