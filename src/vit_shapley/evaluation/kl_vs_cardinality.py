"""KL divergence vs. mask cardinality evaluation (Figure 2 of ViT-Shapley).

Computes mean KL(classifier(full_image) || surrogate(masked_image)) across
a range of masked-patch cardinalities to evaluate how well the surrogate
approximates the classifier at different levels of patch visibility.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def sample_fixed_cardinality_masks(
    batch_size: int,
    num_patches: int,
    num_masked: int,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample binary patch masks with exactly ``num_masked`` zeros per row.

    Args:
        batch_size: Number of masks to sample.
        num_patches: Total number of patches (e.g. 196 for 14×14 grid).
        num_masked: Number of patches to mask (set to 0) per sample.
        device: Target device for the output tensor.
        generator: Optional :class:`torch.Generator` for reproducible sampling.

    Returns:
        Float tensor ``(batch_size, num_patches)`` with values in ``{0.0, 1.0}``.
        Each row has exactly ``num_masked`` zeros.

    Raises:
        ValueError: If ``num_masked < 0`` or ``num_masked > num_patches``.
    """
    if num_masked < 0 or num_masked > num_patches:
        raise ValueError(
            f"num_masked must be in [0, num_patches] but got "
            f"num_masked={num_masked}, num_patches={num_patches}."
        )

    masks = torch.ones(batch_size, num_patches, device=device)
    if num_masked > 0:
        perm = torch.rand(
            batch_size, num_patches, device=device, generator=generator
        ).argsort(dim=1)
        masks.scatter_(1, perm[:, :num_masked], 0.0)
    return masks


@torch.no_grad()
def compute_kl_vs_cardinality(
    surrogate: torch.nn.Module,
    classifier: torch.nn.Module,
    images: torch.Tensor,
    num_patches: int,
    num_masks_per_cardinality: int = 50,
    cardinality_step: int = 10,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> dict[int, list[float]]:
    """Compute per-sample KL divergence for each mask cardinality.

    For each cardinality m (number of masked patches), samples
    ``num_masks_per_cardinality`` random masks and evaluates::

        KL(classifier(x) || surrogate(x_s))

    where ``x_s`` is the image with m patches masked.

    When ``seed`` is provided, mask sampling is fully deterministic via a
    dedicated :class:`torch.Generator`.  Calling this function with the same
    ``seed`` (and identical ``num_patches``, ``num_masks_per_cardinality``,
    ``cardinality_step``, and batch size) produces exactly the same masks,
    making it safe to compare different surrogates on equal footing.

    Args:
        surrogate: :class:`~vit_shapley.models.SurrogateViT` instance in eval mode.
        classifier: Frozen classifier (timm ViT) in eval mode.
        images: Image batch ``(N, C, H, W)`` on ``device``.
        num_patches: Total number of patches per image.
        num_masks_per_cardinality: Number of random masks to draw per cardinality.
        cardinality_step: Step size between cardinalities.
        device: Device for mask tensors (defaults to ``images.device``).
        seed: Optional RNG seed.  When set, mask generation is deterministic
            and independent of the global RNG state.

    Returns:
        Dictionary mapping each cardinality (int) to a list of per-sample KL
        values (``N * num_masks_per_cardinality`` floats total per key).
        Always includes cardinality 0 and ``num_patches``.
    """
    if device is None:
        device = images.device

    generator: Optional[torch.Generator] = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    # Compute teacher probabilities once (full image, no masking).
    teacher_probs = classifier(images).softmax(dim=-1)  # (N, C)

    # Build cardinality list, always including num_patches.
    cardinalities = list(range(0, num_patches + 1, cardinality_step))
    if num_patches not in cardinalities:
        cardinalities.append(num_patches)

    N = images.shape[0]
    results: dict[int, list[float]] = {m: [] for m in cardinalities}

    for m in cardinalities:
        for _ in range(num_masks_per_cardinality):
            # Sample one mask per image in the batch.
            mask = sample_fixed_cardinality_masks(
                N, num_patches, m, device=device, generator=generator
            )
            # surrogate expects patch_mask (B, num_patches).
            surrogate_logits = surrogate(images, patch_mask=mask)
            surrogate_log_probs = surrogate_logits.log_softmax(dim=-1)  # (N, C)

            # KL per sample: sum over classes.
            kl_per_sample = F.kl_div(
                surrogate_log_probs, teacher_probs, reduction="none"
            ).sum(-1)  # (N,)

            results[m].extend(kl_per_sample.cpu().tolist())

    return results
