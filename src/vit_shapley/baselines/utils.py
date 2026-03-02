"""Utility functions for explanation baselines (pure numpy, no torch dependency).

Functions:
    generate_mask         — sample binary patch masks (uniform or Shapley distribution)
    get_random_explanation — random near-zero ordering baseline
    get_relative_value    — convert raw attribution scores to ordinal ranks
    explanation_to_mask   — convert per-patch scores to insertion/deletion mask sequences
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def generate_mask(
    num_players: int,
    num_mask_samples: Optional[int] = None,
    paired_mask_samples: bool = True,
    mode: str = "uniform",
    random_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Sample binary patch masks from a specified distribution.

    Args:
        num_players: Number of patches ``P``.
        num_mask_samples: How many masks to generate.  ``None`` returns a
            single mask of shape ``(P,)`` instead of ``(N, P)``.
        paired_mask_samples: When ``True`` each base mask ``S`` is immediately
            followed by its complement ``1−S``, halving the number of
            independent samples.  ``num_mask_samples`` must be even.
        mode: Distribution for the subset size:
            - ``"uniform"`` — subset size drawn uniformly from ``[0, P]``.
            - ``"shapley"`` — subset size drawn ∝ ``1/(k·(P−k))``.
        random_state: NumPy ``RandomState`` for reproducibility.  Defaults to
            the global random state.

    Returns:
        Integer array of shape ``(num_mask_samples, P)`` with values in
        ``{0, 1}``, or shape ``(P,)`` when ``num_mask_samples`` is ``None``.
        ``1`` = visible, ``0`` = masked.

    Raises:
        ValueError: If ``paired_mask_samples=True`` and ``num_mask_samples`` is
            odd, or if ``mode`` is not recognised.
    """
    rng = random_state if random_state is not None else np.random

    # When generating a single mask (num_mask_samples=None), paired logic is
    # meaningless; treat as unpaired so no even-number constraint applies.
    num_samples_ = num_mask_samples if num_mask_samples is not None else 1
    use_paired = paired_mask_samples and (num_mask_samples is not None)

    if use_paired:
        if num_samples_ % 2 != 0:
            raise ValueError(
                "'num_mask_samples' must be even when 'paired_mask_samples' is True; "
                f"got {num_samples_}"
            )
        base = num_samples_ // 2
    else:
        base = num_samples_

    if mode == "uniform":
        # Each entry drawn independently; threshold from a per-row random scalar.
        masks = (rng.rand(base, num_players) > rng.rand(base, 1)).astype(int)
    elif mode == "shapley":
        # Draw cardinality k ∝ 1/(k·(P−k)) then keep k patches.
        ks = np.arange(1, num_players)  # possible sizes {1, …, P-1}
        probs = 1.0 / (ks * (num_players - ks))
        probs = probs / probs.sum()
        # Vectorised: threshold per row differs per sample.
        chosen_k = rng.choice(ks, size=(base, 1), p=probs)  # (base, 1)
        thresholds = chosen_k / num_players  # fraction to keep
        masks = (rng.rand(base, num_players) > 1 - thresholds).astype(int)
    else:
        raise ValueError(f"'mode' must be 'uniform' or 'shapley'; got {mode!r}")

    if use_paired:
        masks = np.stack([masks, 1 - masks], axis=1).reshape(num_samples_, num_players)

    if num_mask_samples is None:
        return masks.squeeze(0)  # (P,)
    return masks  # (N, P)


def get_random_explanation(
    num_players: int,
    num_samples: Optional[int] = None,
    random_seed: int = 42,
) -> np.ndarray:
    """Return a near-zero random ordering baseline explanation.

    All values are extremely small (< 1e-40) so that the ranking induced by
    this explanation is essentially random, while still being non-zero to allow
    ``get_relative_value`` to break ties consistently.

    Args:
        num_players: Number of patches ``P``.
        num_samples: If given, returns shape ``(num_samples, P)``; otherwise
            shape ``(P,)``.
        random_seed: Seed for the ``RandomState`` so results are reproducible.

    Returns:
        Float array of shape ``(P,)`` or ``(num_samples, P)``.
    """
    rng = np.random.RandomState(random_seed)
    if num_samples is None:
        return rng.uniform(low=0.0, high=1e-40, size=(num_players,))
    return rng.uniform(low=0.0, high=1e-40, size=(num_samples, num_players))


def get_relative_value(
    x: np.ndarray,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Convert a 1-D attribution array to ordinal ranks with random tie-breaking.

    The rank of patch ``i`` is the number of patches with strictly smaller
    attribution values (0 = least important).  Ties are broken randomly.

    Args:
        x: 1-D float array of shape ``(P,)``.
        random_seed: Integer seed for reproducible tie-breaking.  ``None``
            uses the global random state.

    Returns:
        Integer array of shape ``(P,)`` with ordinal ranks in ``[0, P−1]``.
    """
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D; got shape {x.shape}")

    if isinstance(random_seed, int):
        rng = np.random.default_rng(random_seed)
        perm = rng.permutation(np.arange(len(x)))
    else:
        perm = np.random.permutation(np.arange(len(x)))

    # Apply permutation to break ties, then double-argsort to get ranks.
    argsorted = np.arange(len(x))[perm][np.argsort(x[perm])]
    relative_value = np.argsort(argsorted)
    return relative_value


def explanation_to_mask(
    explanation: np.ndarray,
    mode: str = "insertion",
) -> np.ndarray:
    """Convert per-patch attribution scores to a sequence of binary masks.

    Produces ``P+1`` masks (one per step of an insertion or deletion curve).
    Each step adds/removes one more patch according to the explanation ranking.

    Insertion mode (``mode="insertion"``):
        - Step 0: all patches masked (empty image).
        - Step P: all patches visible (full image).
        - Highest-attributed patches are inserted first.

    Deletion mode (``mode="deletion"``):
        - Step 0: all patches visible (full image).
        - Step P: all patches masked (empty image).
        - Highest-attributed patches are deleted first.

    Args:
        explanation: Float array of shape ``(B, P)`` — per-sample, per-patch
            attribution scores.
        mode: ``"insertion"`` or ``"deletion"``.

    Returns:
        Boolean array of shape ``(B, P+1, P)`` — ``True`` = patch visible.
        ``axis=1`` indexes the evaluation step (0 = beginning, P = end).

    Raises:
        ValueError: If ``mode`` is not ``"insertion"`` or ``"deletion"``.
    """
    B, P = explanation.shape

    # Broadcast explanation against sorted thresholds.
    # exp_broad: (B, P, P) — copy of explanation per threshold step.
    exp_broad = np.repeat(explanation[:, np.newaxis, :], P, axis=1)

    # sorted_desc: (B, P) — per-sample descending sort.
    sorted_desc = np.sort(explanation, axis=-1)[:, ::-1]  # (B, P)

    if mode == "insertion":
        # At step k, patches with attribution strictly greater than the k-th
        # largest threshold are visible.  Step 0 (threshold = largest value)
        # has 0 visible patches; the final appended row is all-visible.
        # Compare (B, P, P) against thresholds (B, P, 1).
        visible = exp_broad > sorted_desc[:, :, np.newaxis]  # (B, P, P)
        # Append all-visible step → shape (B, P+1, P).
        all_visible = np.ones((B, 1, P), dtype=bool)
        result = np.concatenate([visible, all_visible], axis=1)
    elif mode == "deletion":
        # At step k, patches with attribution strictly less than the k-th
        # largest threshold are still visible.  Step 0 (prepended all-visible)
        # has all patches; the last comparison row has 0 visible patches.
        visible = exp_broad < sorted_desc[:, :, np.newaxis]  # (B, P, P)
        # Prepend all-visible step → shape (B, P+1, P).
        all_visible = np.ones((B, 1, P), dtype=bool)
        result = np.concatenate([all_visible, visible], axis=1)
    else:
        raise ValueError(f"'mode' must be 'insertion' or 'deletion'; got {mode!r}")

    return result  # (B, P+1, P)
