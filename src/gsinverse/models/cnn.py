"""CNN backbone regressor for the Gray-Scott inverse problem."""

import timm
import torch.nn as nn


class CNNRegressor(nn.Module):
    """A timm CNN backbone with global average pooling and an MLP head.

    Defaults to a ResNet-18 backbone adapted to ``in_chans=3`` inputs at
    native 64x64 resolution, with a small MLP head producing
    ``num_outputs`` regression targets.
    """

    def __init__(
        self,
        in_chans: int = 3,
        num_outputs: int = 2,
        backbone: str = "resnet18",
        pretrained: bool = False,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,  # remove classifier head, keep pooled features
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_outputs),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
