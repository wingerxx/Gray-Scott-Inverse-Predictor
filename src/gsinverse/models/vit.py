"""Small-input Vision Transformer regressor for the Gray-Scott inverse problem.

Configured for native 64x64 input with patch size 8, giving an 8x8 = 64
token sequence -- a meaningful sequence length without upscaling.
"""

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


class ViTRegressor(VisionTransformer):
    """A "vit_tiny"-sized ViT for native 64x64, patch-8 input.

    Subclasses timm's ``VisionTransformer`` with an explicit regression
    forward pass: CLS token features → dropout → linear head → (num_outputs,).
    """

    def __init__(
        self,
        in_chans: int = 3,
        num_outputs: int = 2,
        img_size: int = 64,
        patch_size: int = 8,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,  # disable parent classifier head
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        self.reg_dropout = nn.Dropout(dropout)
        self.reg_head = nn.Linear(embed_dim, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)   # (B, embed_dim) CLS token
        x = self.reg_dropout(x)
        return self.reg_head(x)
