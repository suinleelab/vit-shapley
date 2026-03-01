"""ViT classifier builder using timm.

This module wraps :func:`timm.create_model` to build a Vision Transformer
classifier with optional pretrained weights and configurable dropout.
"""

import timm
import torch.nn as nn


def build_vit_classifier(
    model_name: str = "vit_base_patch16_224",
    num_classes: int = 10,
    pretrained: bool = True,
    dropout: float = 0.0,
) -> nn.Module:
    """Build a ViT classifier via timm.

    Args:
        model_name: Any timm ViT model name, e.g. ``"vit_tiny_patch16_224"``,
                    ``"vit_small_patch16_224"``, ``"vit_base_patch16_224"``.
        num_classes: Number of output classes (10 for ImageNette).
        pretrained: If ``True`` load ImageNet-pretrained weights from timm hub.
        dropout: Dropout probability applied to the classifier head.
                 Set to 0.0 to disable.

    Returns:
        A :class:`torch.nn.Module` ready for training or inference.

    Example::

        model = build_vit_classifier("vit_tiny_patch16_224", num_classes=10)
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )
    return model
