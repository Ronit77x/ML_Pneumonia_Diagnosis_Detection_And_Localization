import torch
import torch.nn as nn


class WeakPneumoniaCNN(nn.Module):
    """CAM-compatible 10-convolution-layer CNN.

    There is exactly one FC layer after global average pooling, matching the
    structural requirement described in the supplied reference paper.
    """

    def __init__(self):
        super().__init__()
        blocks = []
        channels = [1, 16, 16, 32, 32, 64, 64, 96, 96, 128, 128]
        for i in range(10):
            blocks += [
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
            if i in {1, 3, 5, 7}:
                blocks.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 2)

    def forward(self, x, return_features=False):
        fmap = self.features(x)
        pooled = self.gap(fmap).flatten(1)
        logits = self.fc(pooled)
        if return_features:
            return logits, fmap
        return logits
