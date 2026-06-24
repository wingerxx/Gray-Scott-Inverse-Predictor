"""Model factory: build a regressor from a config dict."""

import torch.nn as nn

from .cnn import CNNRegressor
from .vit import ViTRegressor

_VALID_MODELS = ("cnn", "vit")


def build_model(config: dict) -> nn.Module:
    """Build the model specified by ``config["model"]``.

    Args:
        config: Full configuration dict. Reads ``model.name``,
            ``model.in_chans``, ``model.num_outputs``, ``model.backbone``,
            and ``model.dropout``.

    Returns:
        An ``nn.Module`` mapping ``(N, in_chans, H, W)`` inputs to
        ``(N, num_outputs)`` predictions.

    Raises:
        ValueError: If ``model.name`` is not one of ``'cnn'`` or ``'vit'``.
    """
    model_cfg = config["model"]
    name = model_cfg["name"]

    if name not in _VALID_MODELS:
        raise ValueError(
            f"Unknown model name: {name!r}. "
            f"Expected one of {_VALID_MODELS}."
        )

    in_chans = model_cfg.get("in_chans", 3)
    num_outputs = model_cfg.get("num_outputs", 2)
    dropout = model_cfg.get("dropout", 0.0)

    if name == "cnn":
        backbone = model_cfg.get("backbone", "resnet18")
        return CNNRegressor(in_chans=in_chans, num_outputs=num_outputs, backbone=backbone, dropout=dropout)
    if name == "vit":
        return ViTRegressor(in_chans=in_chans, num_outputs=num_outputs, dropout=dropout)


__all__ = ["build_model", "CNNRegressor", "ViTRegressor"]
