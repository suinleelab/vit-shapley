"""Gradient-based explanation baselines for ViT-Shapley.

Wraps a timm ViT classifier in two ways (pixel space and embedding space) for
use with `captum` attribution methods, then aggregates the resulting
pixel/embedding gradients into per-patch importance scores.

Requires ``captum`` (optional dependency):
    pip install captum

Functions:
    get_vanilla_gradient   — vanilla saliency (absolute gradient)
    get_smoothgrad         — SmoothGrad (averaged gradient over noisy copies)
    get_vargrad            — VarGrad (variance of gradients over noisy copies)
    get_integrated_gradients — Integrated Gradients (path integral, zero baseline)

All functions return numpy arrays of shape ``(output_dim, P)`` where ``P`` is
the number of patches.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_captum():
    try:
        import captum  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The gradient baselines require 'captum'. "
            "Install it with: pip install captum"
        ) from e


class _PixelWrapper(nn.Module):
    """Wrap a timm ViT classifier for pixel-space gradient attribution.

    The forward method applies the ViT then softmax (or sigmoid for binary
    classification) to return class probabilities.

    Args:
        vit: timm ViT model (the bare ViT, not SurrogateViT).
        binary: If ``True``, apply sigmoid instead of softmax.
    """

    def __init__(self, vit: nn.Module, binary: bool = False) -> None:
        super().__init__()
        self.vit = vit
        self.binary = binary

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities from pixel input ``(B, C, H, W)``."""
        logits = self.vit(x)
        return logits.sigmoid() if self.binary else logits.softmax(dim=-1)


class _EmbeddingWrapper(nn.Module):
    """Wrap a timm ViT classifier for embedding-space gradient attribution.

    The forward method accepts patch embeddings (before CLS prepend and
    positional encoding), then runs the remaining ViT pipeline to produce
    class probabilities.

    Args:
        vit: timm ViT model.  Must have ``_pos_embed``, ``patch_drop``,
            ``norm_pre``, ``blocks``, ``norm``, and ``forward_head`` methods
            (standard timm 1.0+ layout).
        binary: If ``True``, apply sigmoid instead of softmax.
    """

    def __init__(self, vit: nn.Module, binary: bool = False) -> None:
        super().__init__()
        self.vit = vit
        self.binary = binary

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Return class probabilities from patch embeddings ``(B, P, D)``.

        Args:
            embedding: Patch embeddings of shape ``(B, P, D)`` — the output
                of ``vit.patch_embed(x)``, before CLS / positional encoding.

        Returns:
            Tensor ``(B, num_classes)`` of class probabilities.
        """
        vit = self.vit
        x = vit._pos_embed(embedding)   # adds CLS token + positional encoding
        x = vit.patch_drop(x)
        x = vit.norm_pre(x)
        x = vit.blocks(x)
        x = vit.norm(x)
        logits = vit.forward_head(x)
        return logits.sigmoid() if self.binary else logits.softmax(dim=-1)


def _aggregate_pixel(
    attributions: torch.Tensor,
    patch_size: int,
    signed: bool = False,
) -> np.ndarray:
    """Aggregate pixel-level attributions to patch-level scores.

    Sums (absolute) attribution magnitudes within each ``patch_size × patch_size``
    patch region using a 2-D convolution.

    Args:
        attributions: Tensor ``(output_dim, C, H, W)`` — per-class pixel
            attribution gradients.
        patch_size: Spatial size of each patch in pixels.
        signed: If ``True``, sum signed values; otherwise sum absolute values.

    Returns:
        Float numpy array ``(output_dim, P)`` where ``P = (H/patch_size)²``.
    """
    x = attributions if signed else attributions.abs()
    weight = torch.ones(
        1, x.shape[1], patch_size, patch_size,
        dtype=x.dtype, device=x.device,
    )
    # F.conv2d with stride=patch_size sums each patch region.
    patch_scores = F.conv2d(x, weight=weight, stride=patch_size)  # (D, 1, Ph, Pw)
    return patch_scores.squeeze(1).reshape(x.shape[0], -1).detach().cpu().numpy()


def _aggregate_embedding(
    attributions: torch.Tensor,
    signed: bool = False,
) -> np.ndarray:
    """Aggregate embedding-level attributions to patch-level scores.

    Sums (absolute) gradient magnitudes over the embedding dimension.

    Args:
        attributions: Tensor ``(output_dim, P, D)`` — per-class patch
            embedding attribution gradients.
        signed: If ``True``, sum signed values; otherwise sum absolute values.

    Returns:
        Float numpy array ``(output_dim, P)``.
    """
    x = attributions if signed else attributions.abs()
    return x.sum(dim=-1).detach().cpu().numpy()


def get_vanilla_gradient(
    image: torch.Tensor,
    classifier: nn.Module,
    output_dim: int,
    space: str = "embedding",
    patch_size: int = 16,
) -> np.ndarray:
    """Compute vanilla gradient (saliency) attribution scores.

    Computes ``|∂ prob_c / ∂ input|`` for each class ``c`` and aggregates
    to patch-level via absolute summation.

    Args:
        image: Single image ``(C, H, W)`` or batch ``(1, C, H, W)``.
        classifier: timm ViT model (bare, not :class:`SurrogateViT`).
        output_dim: Number of output classes.
        space: ``"pixel"`` or ``"embedding"`` — which input space to compute
            gradients in.
        patch_size: Patch size in pixels (needed for pixel-space aggregation).

    Returns:
        Float numpy array ``(output_dim, P)`` — per-class, per-patch
        absolute-gradient attribution scores.
    """
    _check_captum()
    from captum.attr import Saliency

    if image.dim() == 3:
        image = image.unsqueeze(0)

    binary = output_dim == 1
    device = next(classifier.parameters()).device
    image = image.to(device)

    if space == "pixel":
        wrapper = _PixelWrapper(classifier, binary=binary).to(device)
        attributions_list = []
        for c in range(output_dim):
            sal = Saliency(wrapper)
            attr = sal.attribute(image, target=c)  # (1, C, H, W)
            attributions_list.append(attr)
        attributions = torch.cat(attributions_list, dim=0)  # (D, C, H, W)
        return _aggregate_pixel(attributions, patch_size)

    else:  # embedding
        wrapper = _EmbeddingWrapper(classifier, binary=binary).to(device)
        with torch.no_grad():
            embedding = classifier.patch_embed(image)  # (1, P, D)
        attributions_list = []
        for c in range(output_dim):
            sal = Saliency(wrapper)
            attr = sal.attribute(embedding.detach(), target=c)  # (1, P, D)
            attributions_list.append(attr)
        attributions = torch.cat(attributions_list, dim=0)  # (D, P, D_emb)
        return _aggregate_embedding(attributions)


def get_smoothgrad(
    image: torch.Tensor,
    classifier: nn.Module,
    output_dim: int,
    space: str = "embedding",
    n_samples: int = 10,
    patch_size: int = 16,
) -> np.ndarray:
    """Compute SmoothGrad attribution scores.

    Averages the gradient magnitude over ``n_samples`` noisy copies of the
    input, reducing variance compared to vanilla gradients.

    Args:
        image: Single image ``(C, H, W)`` or batch ``(1, C, H, W)``.
        classifier: timm ViT model.
        output_dim: Number of output classes.
        space: ``"pixel"`` or ``"embedding"``.
        n_samples: Number of noisy samples (captum ``nt_samples``).
        patch_size: Patch size for pixel-space aggregation.

    Returns:
        Float numpy array ``(output_dim, P)``.
    """
    _check_captum()
    from captum.attr import NoiseTunnel, Saliency

    if image.dim() == 3:
        image = image.unsqueeze(0)

    binary = output_dim == 1
    device = next(classifier.parameters()).device
    image = image.to(device)

    if space == "pixel":
        wrapper = _PixelWrapper(classifier, binary=binary).to(device)
        attributions_list = []
        for c in range(output_dim):
            nt = NoiseTunnel(Saliency(wrapper))
            attr = nt.attribute(image, nt_type="smoothgrad", nt_samples=n_samples, target=c)
            attributions_list.append(attr)
        attributions = torch.cat(attributions_list, dim=0)  # (D, C, H, W)
        return _aggregate_pixel(attributions, patch_size)

    else:
        wrapper = _EmbeddingWrapper(classifier, binary=binary).to(device)
        with torch.no_grad():
            embedding = classifier.patch_embed(image)
        attributions_list = []
        for c in range(output_dim):
            nt = NoiseTunnel(Saliency(wrapper))
            attr = nt.attribute(embedding.detach(), nt_type="smoothgrad", nt_samples=n_samples, target=c)
            attributions_list.append(attr)
        attributions = torch.cat(attributions_list, dim=0)  # (D, P, D_emb)
        return _aggregate_embedding(attributions)


def get_vargrad(
    image: torch.Tensor,
    classifier: nn.Module,
    output_dim: int,
    space: str = "embedding",
    n_samples: int = 10,
    patch_size: int = 16,
) -> np.ndarray:
    """Compute VarGrad attribution scores.

    Uses the variance of gradients over noisy copies as the attribution
    signal, which reduces bias compared to SmoothGrad.

    Args:
        image: Single image ``(C, H, W)`` or batch ``(1, C, H, W)``.
        classifier: timm ViT model.
        output_dim: Number of output classes.
        space: ``"pixel"`` or ``"embedding"``.
        n_samples: Number of noisy samples (captum ``nt_samples``).
        patch_size: Patch size for pixel-space aggregation.

    Returns:
        Float numpy array ``(output_dim, P)``.
    """
    _check_captum()
    from captum.attr import NoiseTunnel, Saliency

    if image.dim() == 3:
        image = image.unsqueeze(0)

    binary = output_dim == 1
    device = next(classifier.parameters()).device
    image = image.to(device)

    if space == "pixel":
        wrapper = _PixelWrapper(classifier, binary=binary).to(device)
        attributions_list = []
        for c in range(output_dim):
            nt = NoiseTunnel(Saliency(wrapper))
            attr = nt.attribute(image, nt_type="vargrad", nt_samples=n_samples, target=c)
            attributions_list.append(attr)
        attributions = torch.cat(attributions_list, dim=0)  # (D, C, H, W)
        return _aggregate_pixel(attributions, patch_size)

    else:
        wrapper = _EmbeddingWrapper(classifier, binary=binary).to(device)
        with torch.no_grad():
            embedding = classifier.patch_embed(image)
        attributions_list = []
        for c in range(output_dim):
            nt = NoiseTunnel(Saliency(wrapper))
            attr = nt.attribute(embedding.detach(), nt_type="vargrad", nt_samples=n_samples, target=c)
            attributions_list.append(attr)
        attributions = torch.cat(attributions_list, dim=0)  # (D, P, D_emb)
        return _aggregate_embedding(attributions)


def get_integrated_gradients(
    image: torch.Tensor,
    classifier: nn.Module,
    output_dim: int,
    space: str = "embedding",
    n_steps: int = 50,
    patch_size: int = 16,
) -> np.ndarray:
    """Compute Integrated Gradients attribution scores.

    Integrates gradients along a straight-line path from a zero baseline to
    the input.  Uses **signed** aggregation (not absolute value), so positive
    scores indicate patches that increase the class probability.

    Args:
        image: Single image ``(C, H, W)`` or batch ``(1, C, H, W)``.
        classifier: timm ViT model.
        output_dim: Number of output classes.
        space: ``"pixel"`` or ``"embedding"``.
        n_steps: Number of integration steps (captum ``n_steps``).
        patch_size: Patch size for pixel-space aggregation.

    Returns:
        Float numpy array ``(output_dim, P)`` — per-class, per-patch signed
        integrated gradient scores.
    """
    _check_captum()
    from captum.attr import IntegratedGradients

    if image.dim() == 3:
        image = image.unsqueeze(0)

    binary = output_dim == 1
    device = next(classifier.parameters()).device
    image = image.to(device)

    if space == "pixel":
        wrapper = _PixelWrapper(classifier, binary=binary).to(device)
        baseline_pixel = torch.zeros_like(image)
        attributions_list = []
        for c in range(output_dim):
            ig = IntegratedGradients(wrapper)
            attr = ig.attribute(image, baselines=baseline_pixel, target=c, n_steps=n_steps)
            attributions_list.append(attr)
        attributions = torch.cat(attributions_list, dim=0)  # (D, C, H, W)
        return _aggregate_pixel(attributions, patch_size, signed=True)

    else:
        wrapper = _EmbeddingWrapper(classifier, binary=binary).to(device)
        with torch.no_grad():
            embedding = classifier.patch_embed(image)
            baseline_emb = classifier.patch_embed(torch.zeros_like(image))
        attributions_list = []
        for c in range(output_dim):
            ig = IntegratedGradients(wrapper)
            attr = ig.attribute(
                embedding.detach(),
                baselines=baseline_emb.detach(),
                target=c,
                n_steps=n_steps,
            )
            attributions_list.append(attr)
        attributions = torch.cat(attributions_list, dim=0)  # (D, P, D_emb)
        return _aggregate_embedding(attributions, signed=True)
