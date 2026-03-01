"""Surrogate ViT with patch masking for partial-input evaluation (Stage 2).

The surrogate is a fine-tuned copy of the classifier that handles masked image
patches.  Two masking strategies are supported:

``"attn_mask"`` (default, Section 4 and Appendix A of the ViT-Shapley paper):
    Add ``-inf`` to attention logits for masked patch columns before softmax,
    preventing every token from attending to masked patches (Eq. 9).
    The CLS token is always visible.  timm 1.0.0+ propagates ``attn_mask``
    natively through the call chain::

        vit(x, attn_mask=bias)
            → forward_features(x, attn_mask=bias)
                → blk(x, attn_mask=bias)
                    → attn(normed_x, attn_mask=bias)
                        → F.scaled_dot_product_attention(q, k, v, attn_mask=bias)

    No attention-module replacement is required.

``"zero_input"``:
    Replace masked patch pixel regions with zeros *before* the ViT forward
    pass.  The model sees black patches for masked positions and standard pixels
    for visible ones.  No attention modification is applied.

Surrogate training objective (Section 4, Eq. 2):
    min_β E_{p(x)} E_{p(s)} [DKL(f(x;η) || g(x_s;β))]

where f(x;η) is the frozen classifier and g(x_s;β) is the surrogate.
"""

from __future__ import annotations

import os
from typing import Optional

import timm
import torch
import torch.nn as nn


MASKING_STRATEGIES = ("attn_mask", "zero_input")


class SurrogateViT(nn.Module):
    """ViT surrogate model for partial-input evaluation with patch masking.

    Wraps a timm ``VisionTransformer`` and applies one of two masking
    strategies to handle randomly masked patch subsets during training and
    inference.

    Args:
        vit: A timm ViT model (output of :func:`timm.create_model`).
        masking_strategy: One of ``"attn_mask"`` (default) or ``"zero_input"``.

            - ``"attn_mask"``: adds ``-inf`` to attention logits for masked
              patch columns so no token can attend to them (Appendix A, Eq. 9).
            - ``"zero_input"``: replaces masked patch pixel regions with zeros
              before the ViT forward pass.
    """

    def __init__(self, vit: nn.Module, masking_strategy: str = "attn_mask") -> None:
        if masking_strategy not in MASKING_STRATEGIES:
            raise ValueError(
                f"Unknown masking_strategy {masking_strategy!r}. "
                f"Choose from {MASKING_STRATEGIES}."
            )
        super().__init__()
        self.vit = vit
        self.masking_strategy = masking_strategy
        self.patch_size: int = vit.patch_embed.patch_size[0]

    def _build_attn_bias(
        self, patch_mask: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """Convert a patch mask to an additive attention bias.

        The CLS token (position 0) is always visible.  Masked patch positions
        receive ``-inf`` so that no query can attend to them.

        Args:
            patch_mask: Float/bool tensor ``(B, num_patches)`` — 1/True = visible.
            dtype: Dtype of the bias tensor (should match query dtype).

        Returns:
            Additive bias ``(B, 1, 1, num_patches+1)`` with 0 for visible
            positions and ``-inf`` for masked positions.
        """
        B, P = patch_mask.shape
        # Prepend 1 for CLS token (always visible).
        cls_vis = torch.ones(B, 1, device=patch_mask.device, dtype=patch_mask.dtype)
        token_mask = torch.cat([cls_vis, patch_mask], dim=1)  # (B, P+1)

        # Build additive bias: 0 = attend, -inf = block.
        bias = torch.zeros(B, 1, 1, P + 1, device=patch_mask.device, dtype=dtype)
        bias = bias.masked_fill(~token_mask.bool()[:, None, None, :], float("-inf"))
        return bias

    def _zero_masked_patches(
        self, x: torch.Tensor, patch_mask: torch.Tensor
    ) -> torch.Tensor:
        """Zero out masked patch pixel regions in the input image.

        Args:
            x: Image batch ``(B, C, H, W)``.
            patch_mask: Float tensor ``(B, num_patches)`` — ``1`` = visible,
                        ``0`` = replace with zeros.

        Returns:
            Image tensor with masked patches replaced by zeros, same shape as
            ``x``.
        """
        B, C, H, W = x.shape
        ps = self.patch_size
        grid_h, grid_w = H // ps, W // ps
        # (B, num_patches) → (B, 1, grid_h, grid_w) → (B, 1, H, W)
        mask_spatial = (
            patch_mask.view(B, 1, grid_h, grid_w)
            .repeat_interleave(ps, dim=2)
            .repeat_interleave(ps, dim=3)
        )
        return x * mask_spatial

    def forward(
        self,
        x: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional patch masking.

        Args:
            x: Image batch ``(B, C, H, W)``.
            patch_mask: Binary tensor ``(B, num_patches)`` — ``1`` = visible,
                        ``0`` = masked.  ``None`` runs without masking
                        (identical to the original ViT).

        Returns:
            Class logits ``(B, num_classes)``.
        """
        if patch_mask is None:
            return self.vit(x)

        if self.masking_strategy == "attn_mask":
            attn_bias = self._build_attn_bias(patch_mask, dtype=x.dtype)
            return self.vit(x, attn_mask=attn_bias)

        # zero_input: blank out masked patch pixels, then standard forward
        x_masked = self._zero_masked_patches(x, patch_mask)
        return self.vit(x_masked)


def build_vit_surrogate(
    model_name: str = "vit_base_patch16_224",
    num_classes: int = 10,
    classifier_ckpt_path: Optional[str | os.PathLike] = None,
    masking_strategy: str = "attn_mask",
) -> SurrogateViT:
    """Build a :class:`SurrogateViT`, optionally initialised from a classifier checkpoint.

    Classifier weights are loaded before the model is wrapped in
    :class:`SurrogateViT` so that state-dict keys match timm's standard layout.

    Args:
        model_name: timm ViT model name (e.g. ``"vit_tiny_patch16_224"``).
        num_classes: Number of output classes.
        classifier_ckpt_path: Path to a ``best_classifier.pth`` file produced by
                              :func:`~vit_shapley.training.train_classifier`.
                              If ``None``, the model is randomly initialised.
        masking_strategy: Patch masking method — ``"attn_mask"`` (default) or
                          ``"zero_input"``.  See :class:`SurrogateViT` for details.

    Returns:
        :class:`SurrogateViT` ready for fine-tuning.

    Example::

        surrogate = build_vit_surrogate(
            "vit_tiny_patch16_224",
            num_classes=10,
            classifier_ckpt_path="checkpoints/classifier/best_classifier.pth",
            masking_strategy="zero_input",
        )
    """
    vit = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    if classifier_ckpt_path is not None:
        ckpt = torch.load(
            classifier_ckpt_path, map_location="cpu", weights_only=True
        )
        state_dict = ckpt.get("model_state_dict", ckpt)
        vit.load_state_dict(state_dict)

    return SurrogateViT(vit, masking_strategy=masking_strategy)
