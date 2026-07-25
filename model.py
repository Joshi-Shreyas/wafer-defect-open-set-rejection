"""
model.py

ResNet-18-style CNN backbone, sized for 2-channel (wafer map + validity mask) 96x96 input.
Used identically across both training regimes (supervised-only and SSL-pretrained),
so that any performance difference reflects the training regime, not architecture differences.
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Standard ResNet basic block: two 3x3 convs with a skip connection."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class WaferResNet(nn.Module):
    """
    ResNet-18-style backbone, adapted for small (96x96) 2-channel input.

    Standard ResNet-18 is designed for 224x224 3-channel ImageNet input, with an
    aggressive initial 7x7 stride-2 conv + maxpool that would over-downsample our
    much smaller 96x96 wafer maps. We use a gentler 3x3 stride-1 stem instead,
    common practice for small-image CNNs (similar to CIFAR-style ResNet variants).

    Returns both the final classification logits AND the pooled feature embedding
    (needed for open-set scoring methods like Mahalanobis distance, and for MAE
    pretraining's encoder reuse).
    """

    def __init__(self, num_classes=8, in_channels=2, base_width=64):
        super().__init__()
        self.in_channels = base_width

        # Gentle stem: no aggressive downsampling for our small 96x96 input
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )

        # Four residual stages, each downsampling by 2x (except the first)
        self.layer1 = self._make_layer(base_width, base_width, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(base_width, base_width * 2, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(base_width * 2, base_width * 4, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(base_width * 4, base_width * 8, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = base_width * 8
        self.fc = nn.Linear(self.feature_dim, num_classes)

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        """Returns the pooled feature embedding (before the classification head)."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x, return_features=False):
        features = self.forward_features(x)
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits


def build_model(num_classes=8, in_channels=2):
    return WaferResNet(num_classes=num_classes, in_channels=in_channels)


if __name__ == "__main__":
    # Quick sanity check: verify shapes flow correctly end-to-end
    model = build_model(num_classes=8, in_channels=2)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    dummy_input = torch.randn(4, 2, 96, 96)  # batch of 4, 2 channels, 96x96
    logits, features = model(dummy_input, return_features=True)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Feature embedding shape: {features.shape}  (expected: [4, 512])")
    print(f"Output logits shape: {logits.shape}  (expected: [4, 8])")

    # Confirm GPU compatibility
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        dummy_input = dummy_input.to(device)
        logits, features = model(dummy_input, return_features=True)
        print(f"\nGPU test passed. Logits device: {logits.device}")
    else:
        print("\nCUDA not available - running on CPU only.")
