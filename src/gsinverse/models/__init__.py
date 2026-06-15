"""Model factory: build a regressor from a config dict."""

import torch.nn as nn

from .cnn import CNNRegressor
from .vit import ViTRegressor


def build_model(config: dict) -> nn.Module:
    """Build the model specified by ``config["model"]``.

    Args:
        config: Full configuration dict. Reads ``model.name``,
            ``model.in_chans``, and ``model.num_outputs``.

    Returns:
        An ``nn.Module`` mapping ``(N, in_chans, H, W)`` inputs to
        ``(N, num_outputs)`` predictions.

    Raises:
        ValueError: If ``model.name`` is not a recognized model name.
    """
    model_cfg = config["model"]
    name = model_cfg["name"]
    in_chans = model_cfg.get("in_chans", 3)
    num_outputs = model_cfg.get("num_outputs", 2)

    if name == "cnn":
        return CNNRegressor(in_chans=in_chans, num_outputs=num_outputs)
    if name == "vit":
        return ViTRegressor(in_chans=in_chans, num_outputs=num_outputs)

    raise ValueError(f"Unknown model name: {name!r} (expected 'cnn' or 'vit')")


__all__ = ["build_model", "CNNRegressor", "ViTRegressor"]
