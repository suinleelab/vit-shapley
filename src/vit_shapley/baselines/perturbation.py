"""Perturbation-based explanation baselines for ViT-Shapley.

Implements leave-one-out (LOO) and RISE baselines that use the surrogate
model to evaluate the importance of each image patch.

Functions:
    leave_one_out — LOO importance via single forward pass over P+1 masks
    rise          — RISE via Monte Carlo sampling of random binary masks
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def leave_one_out(
    image: torch.Tensor,
    surrogate: nn.Module,
    num_patches: Optional[int] = None,
    binary: bool = False,
) -> np.ndarray:
    """Compute leave-one-out attribution scores using the surrogate model.

    Evaluates the model once on the full image (all patches visible) and once
    for each patch removed individually.  The attribution for patch ``i`` is:

        φ_i = prob(full image) − prob(full image without patch i)

    All ``P+1`` masks are batched into a single forward pass for efficiency.

    Args:
        image: Single image tensor ``(C, H, W)`` or batch ``(1, C, H, W)``.
        surrogate: :class:`~vit_shapley.models.SurrogateViT` instance.  Must
            expose ``surrogate(x, patch_mask=mask)`` and, unless
            ``num_patches`` is provided, ``surrogate.vit.patch_embed.num_patches``.
        num_patches: Total number of patches ``P``.  Inferred from
            ``surrogate.vit.patch_embed.num_patches`` when ``None``.
        binary: If ``True``, treat the model output as a binary classification
            (sigmoid instead of softmax) and return shape ``(1, P)``.

    Returns:
        Float numpy array ``(num_classes, P)`` or ``(1, P)`` when
        ``binary=True`` — per-class, per-patch attribution scores.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)  # (1, C, H, W)
    try:
        device = next(surrogate.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    image = image.to(device)

    if num_patches is None:
        num_patches = surrogate.vit.patch_embed.num_patches
    P = num_patches

    # Build (P+1, P) mask: first row all-ones (grand), then identity-subtracted.
    # grand mask: all visible
    grand_row = torch.ones(1, P, device=device)
    # LOO masks: remove each patch one at a time → (P, P)
    loo_rows = torch.ones(P, P, device=device) - torch.eye(P, device=device)
    all_masks = torch.cat([grand_row, loo_rows], dim=0)  # (P+1, P)

    # Repeat image for each mask.
    images_rep = image.expand(P + 1, -1, -1, -1)  # (P+1, C, H, W)

    was_training = surrogate.training
    surrogate.eval()
    with torch.no_grad():
        logits = surrogate(images_rep, patch_mask=all_masks)  # (P+1, num_classes)
        if binary:
            prob = logits.sigmoid()  # (P+1, 1)
        else:
            prob = logits.softmax(dim=-1)  # (P+1, num_classes)
    if was_training:
        surrogate.train()

    prob_np = prob.cpu().numpy()  # (P+1, C)
    # Attribution = grand − leave-one-out
    grand_prob = prob_np[0:1, :]  # (1, C)
    loo_prob = prob_np[1:, :]     # (P, C)
    attributions = (grand_prob - loo_prob).T  # (C, P)
    return attributions


def rise(
    image: torch.Tensor,
    surrogate: nn.Module,
    include_prob: float = 0.5,
    N: int = 2000,
    batch_size: int = 100,
    num_patches: Optional[int] = None,
    binary: bool = False,
) -> np.ndarray:
    """Compute RISE attribution scores using random mask sampling.

    RISE (Randomised Input Sampling for Explanation) estimates patch importance
    as the weighted average of the model output over random binary masks:

        φ_i ≈ (1/N) Σ_j  prob_j · mask_j[i] / include_prob

    which is implemented as ``prob.T @ mask / mask.sum(axis=0)``.

    Args:
        image: Single image tensor ``(C, H, W)`` or batch ``(1, C, H, W)``.
        surrogate: :class:`~vit_shapley.models.SurrogateViT` instance.  Must
            expose ``surrogate(x, patch_mask=mask)`` and, unless
            ``num_patches`` is provided, ``surrogate.vit.patch_embed.num_patches``.
        include_prob: Probability of each patch being visible in a random mask
            (paper default: 0.5).
        N: Total number of random masks to sample.  Must be divisible by
            ``batch_size``.
        batch_size: Number of masks to evaluate in each forward pass.
        num_patches: Total number of patches ``P``.  Inferred from the
            surrogate when ``None``.
        binary: If ``True``, treat the model output as binary (sigmoid) and
            return shape ``(1, P)``.

    Returns:
        Float numpy array ``(num_classes, P)`` or ``(1, P)`` when
        ``binary=True`` — per-class, per-patch RISE attribution scores.

    Raises:
        ValueError: If ``N`` is not divisible by ``batch_size``.
    """
    if N % batch_size != 0:
        raise ValueError(
            f"N must be divisible by batch_size; got N={N}, batch_size={batch_size}"
        )

    if image.dim() == 3:
        image = image.unsqueeze(0)  # (1, C, H, W)
    try:
        device = next(surrogate.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    image = image.to(device)

    if num_patches is None:
        num_patches = surrogate.vit.patch_embed.num_patches
    P = num_patches

    prob_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []

    was_training = surrogate.training
    surrogate.eval()
    with torch.no_grad():
        for _ in range(N // batch_size):
            # Random binary mask: shape (batch_size, P)
            mask = (torch.rand(batch_size, P, device=device) < include_prob).float()
            images_rep = image.expand(batch_size, -1, -1, -1)

            logits = surrogate(images_rep, patch_mask=mask)  # (batch_size, C)
            if binary:
                prob = logits.sigmoid()
            else:
                prob = logits.softmax(dim=-1)

            prob_list.append(prob.cpu().numpy())    # (batch_size, C)
            mask_list.append(mask.cpu().numpy())    # (batch_size, P)
    if was_training:
        surrogate.train()

    prob_arr = np.concatenate(prob_list, axis=0)  # (N, C)
    mask_arr = np.concatenate(mask_list, axis=0)  # (N, P)

    # Weighted average: (C, N) @ (N, P) / (N,) → (C, P)
    numerator = prob_arr.T @ mask_arr              # (C, P)
    denominator = mask_arr.sum(axis=0)             # (P,)
    # Avoid division by zero (unlikely but guard against it).
    denominator = np.maximum(denominator, 1e-8)
    return numerator / denominator                 # (C, P)
