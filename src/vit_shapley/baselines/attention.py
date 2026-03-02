"""Attention-based explanation baselines for ViT-Shapley.

Implements attention rollout (Abnar & Zuidema, 2020) and raw/layer attention
baselines for ViT classifiers.

Functions:
    compute_joint_attention  — attention rollout: propagate attention across layers
    attentions_to_explanation — aggregate multi-head attention to per-patch scores
    extract_attention_maps    — extract raw attention maps from a timm ViT model
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn


def compute_joint_attention(
    attentions: np.ndarray,
    add_residual: bool = True,
) -> np.ndarray:
    """Compute attention rollout by propagating attention across layers.

    Following Abnar & Zuidema (2020): at each layer, the effective attention
    is the matrix product of the current layer's attention and the previous
    layer's rollout.  Optionally adds the identity (residual stream) before
    normalising, which prevents attention weights from collapsing to zero.

    Args:
        attentions: Float array ``(B, L, N, N)`` — per-layer, per-token
            attention weights *after* averaging over heads (rows must sum
            to 1 for meaningful rollout).
        add_residual: If ``True``, add the identity matrix to each layer's
            attention before normalising (standard rollout).

    Returns:
        Float array ``(B, L, N, N)`` — cumulative rollout at each layer.
        ``result[:, -1, :, :]`` is the full rollout.
    """
    assert attentions.ndim == 4, f"Expected (B, L, N, N); got shape {attentions.shape}"
    B, L, N, _ = attentions.shape

    if add_residual:
        residual = np.eye(N)[np.newaxis, np.newaxis, :, :]  # (1, 1, N, N)
        aug = attentions + residual
        aug = aug / aug.sum(axis=-1, keepdims=True)
    else:
        aug = attentions

    joint = np.zeros_like(aug)
    for i in range(L):
        if i == 0:
            joint[:, 0, :, :] = aug[:, 0, :, :]
        else:
            # (B, N, N) @ (B, N, N) → (B, N, N)
            joint[:, i, :, :] = aug[:, i, :, :] @ joint[:, i - 1, :, :]

    return joint


def attentions_to_explanation(
    attentions: np.ndarray,
    mode: Union[str, int] = "rollout",
) -> np.ndarray:
    """Aggregate multi-head attention maps into per-patch explanation scores.

    Pipeline:
    1. Average attention weights across heads.
    2. Add identity (residual) and re-normalise.
    3. Extract CLS→patch attention according to ``mode``.

    Args:
        attentions: Float array ``(B, L, H, N, N)`` — raw attention weights
            per batch, layer, head, query token, key token.  Typically the
            softmax output, so rows sum to 1.
        mode: How to select the layer:
            - ``"rollout"`` — propagate across all layers (attention rollout).
            - ``"raw"`` — use only the last layer's attention.
            - ``int`` — use the layer at index ``mode`` (0-indexed).

    Returns:
        Float array ``(B, P)`` — per-patch attention scores from the CLS
        token (index 0) to all ``P = N−1`` patch tokens.  Non-negative.
    """
    assert attentions.ndim == 5, (
        f"Expected (B, L, H, N, N); got shape {attentions.shape}"
    )
    assert attentions.shape[-1] == attentions.shape[-2], (
        f"Last two dims must be equal (N, N); got {attentions.shape}"
    )

    # Average over heads: (B, L, H, N, N) → (B, L, N, N)
    attn_avg = attentions.mean(axis=2)

    # Add residual + normalise
    N = attn_avg.shape[-1]
    residual = np.eye(N)[np.newaxis, np.newaxis, :, :]  # (1, 1, N, N)
    attn_res = attn_avg + residual
    attn_res = attn_res / attn_res.sum(axis=-1, keepdims=True)  # (B, L, N, N)

    if isinstance(mode, int):
        # Raw attention at a specific layer index.
        return attn_res[:, mode, 0, 1:]  # (B, P)
    elif mode == "raw":
        return attn_res[:, -1, 0, 1:]  # (B, P) — last layer
    elif mode == "rollout":
        rollout = compute_joint_attention(attn_res, add_residual=False)
        return rollout[:, -1, 0, 1:]  # (B, P)
    else:
        raise ValueError(
            f"'mode' must be 'rollout', 'raw', or an int layer index; got {mode!r}"
        )


def extract_attention_maps(
    vit: nn.Module,
    x: torch.Tensor,
) -> np.ndarray:
    """Extract per-layer, per-head attention maps from a timm ViT model.

    Uses forward hooks on each block's ``attn.attn_drop`` layer to capture
    the attention weights after softmax.  When a block uses fused attention
    (``block.attn.fused_attn = True``), ``fused_attn`` is temporarily set to
    ``False`` for the forward pass so that the attention weights pass through
    ``attn_drop`` and can be intercepted.

    Args:
        vit: A timm ``VisionTransformer`` (or any model with a ``blocks``
            attribute that is a list of attention blocks, each with
            ``block.attn.attn_drop``).
        x: Image batch ``(B, C, H, W)`` on the model's device.

    Returns:
        Float numpy array ``(B, L, H, N, N)`` — attention weights for each
        batch item, layer, head, query token, key token.  ``N = num_patches + 1``
        (includes the CLS token).  Values sum to 1 along the last axis
        (softmax property) unless ``attn_drop`` drops weights.
    """
    blocks: List[nn.Module] = list(vit.blocks)
    collected: List[np.ndarray] = []
    hooks = []

    # Per-block storage so that we capture in layer order.
    layer_attn: List[Optional[torch.Tensor]] = [None] * len(blocks)

    for layer_idx, block in enumerate(blocks):
        attn_module = block.attn

        # Temporarily disable fused attention so that attn weights are
        # materialised and passed through attn_drop.
        orig_fused = getattr(attn_module, "fused_attn", False)
        if orig_fused:
            attn_module.fused_attn = False

        def _make_hook(idx: int, orig: bool, module: nn.Module):
            def hook(mod, inp, out):
                # inp[0] has shape (B, H, N, N) — the attention weights
                layer_attn[idx] = inp[0].detach().cpu()

            def cleanup():
                if orig:
                    module.fused_attn = True

            return hook, cleanup

        hook_fn, cleanup_fn = _make_hook(layer_idx, orig_fused, attn_module)
        handle = attn_module.attn_drop.register_forward_hook(hook_fn)
        hooks.append((handle, cleanup_fn))

    device = next(vit.parameters()).device
    x = x.to(device)

    was_training = vit.training
    vit.eval()
    with torch.no_grad():
        vit(x)
    if was_training:
        vit.train()

    # Remove hooks and restore fused_attn.
    for handle, cleanup_fn in hooks:
        handle.remove()
        cleanup_fn()

    # Stack layers: list of (B, H, N, N) → (B, L, H, N, N)
    result = np.stack([a.numpy() for a in layer_attn], axis=1)
    return result
