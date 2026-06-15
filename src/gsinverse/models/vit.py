"""Small-input Vision Transformer regressor for the Gray-Scott inverse problem.

Configured for native 64x64 input with patch size 8, giving an 8x8 = 64
token sequence -- a meaningful sequence length without upscaling.
"""

from timm.models.vision_transformer import VisionTransformer


class ViTRegressor(VisionTransformer):
    """A "vit_tiny"-sized ViT for native 64x64, patch-8 input.

    Subclasses timm's ``VisionTransformer`` directly (rather than going
    through ``timm.create_model``) so that ``img_size`` and ``patch_size``
    can be freely configured for small inputs.
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
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_outputs,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
