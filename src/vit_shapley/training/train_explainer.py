"""Explainer model training loop for ViT-Shapley Stage 3.

Implements:
  - ``shapley_kernel_weights``: precompute Shapley kernel weights by cardinality
  - ``sample_shapley_masks``: sample masks from the Shapley distribution
  - ``train_one_epoch_explainer``: single epoch with MSE loss and multi-mask sampling
  - ``evaluate_explainer``: validation loop with MSE loss and efficiency gap
  - ``train_explainer``: full training with checkpointing and warmup LR schedule
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from vit_shapley.training.train_surrogate import (
    _cosine_schedule_with_warmup,
    sample_subset_masks,
)


def shapley_kernel_weights(num_patches: int) -> list[float]:
    """Precompute Shapley kernel weights for each cardinality.

    The Shapley kernel weight for a coalition of size ``k`` out of ``n``
    patches is (Eq. 3 of the ViT-Shapley paper / KernelSHAP):

        w(k) = (n−1) / (C(n,k) · k · (n−k))

    Boundary cases ``w(0) = w(n) = 0`` (those terms are ill-defined / handled
    separately via the null value).

    Computation is done in log-space using ``math.lgamma`` to avoid integer
    overflow for large ``n`` (e.g. 196 patches).

    Args:
        num_patches: Total number of patches ``n``.

    Returns:
        Python list of length ``n+1`` with ``weights[k]`` = ``w(k)``.
        ``weights[0] = weights[n] = 0.0`` by definition.
    """
    n = num_patches
    weights = [0.0] * (n + 1)
    log_n_minus_1 = math.log(n - 1) if n > 1 else 0.0

    for k in range(1, n):
        # log C(n,k) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
        log_binom = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        log_w = log_n_minus_1 - log_binom - math.log(k) - math.log(n - k)
        weights[k] = math.exp(log_w)

    return weights


def sample_shapley_masks(
    batch_size: int,
    num_patches: int,
    num_mask_samples: int,
    paired: bool = True,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample binary patch masks from the Shapley distribution.

    A cardinality index ``c`` is sampled from the Shapley distribution
    ∝ ``1/(k·(n−k))`` for ``k ∈ {1, …, n−1}``, then each patch is
    independently included with probability ``1 − c/n`` (Bernoulli threshold
    ``rand() > c/n``).

    When ``paired=True`` the second half of each sample's masks are the
    bitwise complements of the first half, providing variance reduction via
    antithetic sampling (S and 1−S have the same cardinality weight).

    Args:
        batch_size: Number of images in the batch ``B``.
        num_patches: Total number of patches ``n``.
        num_mask_samples: Masks per image ``M``.  Must be even when
                          ``paired=True``.
        paired: Generate paired complementary masks (S and 1−S).
        device: Target device for the returned tensor.
        generator: Optional :class:`torch.Generator` for reproducible sampling.

    Returns:
        Float tensor ``(B, M, n)`` — ``1.0`` = visible, ``0.0`` = masked.
    """
    if device is None:
        device = torch.device("cpu")
    n = num_patches
    if paired and num_mask_samples % 2 != 0:
        raise ValueError(
            f"num_mask_samples must be even when paired=True; got {num_mask_samples}"
        )

    # Number of "base" masks to generate before optionally complementing.
    base = num_mask_samples // 2 if paired else num_mask_samples
    num_total = batch_size * base

    # Shapley weights: w[k] ∝ 1/(k·(n−k)) for k = 1..n-1
    ks = torch.arange(1, n, dtype=torch.float32, device=device)
    weights = 1.0 / (ks * (n - ks))
    weights = weights / weights.sum()

    # Sample cardinality indices: values in [0, n-2]
    cardinality_idx = torch.multinomial(
        weights.unsqueeze(0).expand(num_total, -1),
        num_samples=1,
        generator=generator,
    ).squeeze(-1)  # (num_total,)

    # Bernoulli threshold: each patch included with prob 1 - c/n
    thresholds = cardinality_idx.float() / n  # (num_total,)
    thresholds = thresholds.unsqueeze(1)  # (num_total, 1)

    rand_vals = torch.rand(
        num_total,
        n,
        device=device,
        generator=generator,
    )
    masks_flat = (rand_vals > thresholds).float()  # (num_total, n)

    masks = masks_flat.view(batch_size, base, n)

    if paired:
        masks = torch.cat([masks, 1.0 - masks], dim=1)

    return masks


def train_one_epoch_explainer(
    explainer: nn.Module,
    surrogate: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    num_mask_samples: int = 32,
    paired: bool = True,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    surrogate_device: Optional[torch.device] = None,
    gradient_accumulation_steps: int = 1,
    target_type: str = "multiclass",
) -> dict[str, float]:
    """Run one explainer training epoch.

    Minimises the scaled MSE objective (matching reference ViT-Shapley):

        L(φ) = n · E_{x,S} [ ‖v(∅;x) + Σ_{i∈S} φ'_i(x) − v(S;x)‖² ]

    Masks ``S`` are sampled from the Shapley distribution ∝ 1/(k(n−k)).
    Optional paired sampling (S and 1−S) gives antithetic variance reduction.

    The surrogate is frozen; only the explainer parameters are updated.
    The LR scheduler, if provided, is stepped once per optimizer step (not
    per micro-batch).

    When ``gradient_accumulation_steps > 1`` the loss from each micro-batch
    is divided by that factor before calling ``.backward()``, so the
    effective batch size is ``loader.batch_size × gradient_accumulation_steps``
    while peak GPU memory stays proportional to ``loader.batch_size``.

    Args:
        explainer: The :class:`~vit_shapley.models.ExplainerViT` being trained.
        surrogate: Frozen :class:`~vit_shapley.models.SurrogateViT`.
        loader: Training DataLoader.
        optimizer: Optimizer for explainer parameters.
        device: Target device for the explainer.
        num_mask_samples: Number of masks to sample per image (paper default: 32).
        paired: Use paired (S, 1−S) masks (paper default: True).
        scheduler: Per-step LR scheduler, or ``None``.
        scaler: :class:`torch.amp.GradScaler` for AMP, or ``None``.
        surrogate_device: Device for the frozen surrogate. When ``None``, the
            surrogate is assumed to be on the same device as the explainer.
        gradient_accumulation_steps: Number of micro-batches to accumulate
            before each optimizer step (default: 1, i.e. no accumulation).

    Returns:
        Dict with ``"loss"`` (mean scaled MSE loss per sample).
    """
    if surrogate_device is None:
        surrogate_device = device
    explainer.train()
    surrogate.eval()
    total_loss = 0.0
    total_samples = 0
    num_patches = surrogate.vit.patch_embed.num_patches
    accum = gradient_accumulation_steps

    optimizer.zero_grad()

    for step_idx, (images, _) in enumerate(tqdm(loader, desc="Train", leave=False)):
        images = images.to(device, non_blocking=True)
        B = images.size(0)

        use_amp = scaler is not None
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            with torch.no_grad():
                surr_autocast = surrogate_device.type == "cuda"
                with torch.amp.autocast(
                    device_type=surrogate_device.type, enabled=surr_autocast
                ):
                    surr_images = images.to(surrogate_device)

                    # Null value v(∅; x): all patches masked out
                    null_mask = torch.zeros(B, num_patches, device=surrogate_device)
                    null_logits = surrogate(surr_images, patch_mask=null_mask)
                    if target_type == "binary":
                        null_probs = torch.sigmoid(null_logits).to(device)
                    else:
                        null_probs = null_logits.softmax(dim=-1).to(device)

                    # Grand value v(1; x): all patches visible
                    grand_mask = torch.ones(B, num_patches, device=surrogate_device)
                    grand_logits = surrogate(surr_images, patch_mask=grand_mask)
                    if target_type == "binary":
                        grand_probs = torch.sigmoid(grand_logits).to(device)
                    else:
                        grand_probs = grand_logits.softmax(dim=-1).to(device)

                    # Sample masks (B, M, n) from the Shapley distribution
                    masks = sample_shapley_masks(
                        B, num_patches, num_mask_samples, paired=paired, device=device
                    )

                    # Surrogate values for each mask: v(S; x) for all S
                    # Flatten to (B*M, n), repeat images to (B*M, C, H, W)
                    masks_flat = masks.flatten(0, 1)  # (B*M, n)
                    images_rep = surr_images.repeat_interleave(num_mask_samples, dim=0)
                    surr_logits_flat = surrogate(
                        images_rep, patch_mask=masks_flat.to(surrogate_device)
                    )
                    if target_type == "binary":
                        surr_flat = torch.sigmoid(surr_logits_flat)
                    else:
                        surr_flat = surr_logits_flat.softmax(dim=-1)
                    surrogate_values = surr_flat.view(B, num_mask_samples, -1).to(
                        device
                    )  # (B, M, C)

            # Shapley predictions φ'(x): (B, n, C)
            # Explainer normalises internally given grand and null.
            phi = explainer(images, grand=grand_probs, null=null_probs)

            # Approximate v(S; x) ≈ v(∅) + Σ_{i∈S} φ'_i
            # null_probs: (B, C)  →  unsqueeze → (B, 1, C)
            # masks:      (B, M, n), phi: (B, n, C)
            v_approx = null_probs.unsqueeze(1) + masks.float() @ phi  # (B, M, C)

            # Scaled MSE loss (factor n matches reference)
            loss = num_patches * F.mse_loss(v_approx, surrogate_values)
            # Scale by accumulation factor so gradients average correctly
            scaled_loss = loss / accum

        if use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # Step optimizer every `accum` micro-batches or at the last batch
        if (step_idx + 1) % accum == 0 or (step_idx + 1) == len(loader):
            if use_amp:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(explainer.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(explainer.parameters(), max_norm=1.0)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

        # Track the unscaled loss for reporting
        total_loss += loss.item() * B
        total_samples += B

    return {"loss": total_loss / total_samples}


@torch.no_grad()
def evaluate_explainer(
    explainer: nn.Module,
    surrogate: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    num_mask_samples: int = 32,
    paired: bool = True,
    surrogate_device: Optional[torch.device] = None,
    target_type: str = "multiclass",
) -> dict[str, float]:
    """Evaluate the explainer on the validation set.

    Computes the scaled MSE loss and the **efficiency gap** — how closely the
    Shapley values satisfy the efficiency axiom:

        Σ_i φ'_i(x) ≈ v(grand; x) − v(∅; x)

    Args:
        explainer: The :class:`~vit_shapley.models.ExplainerViT` to evaluate.
        surrogate: Frozen :class:`~vit_shapley.models.SurrogateViT`.
        loader: Validation DataLoader.
        device: Target device for the explainer.
        num_mask_samples: Number of masks to sample per image (paper default: 32).
        paired: Use paired (S, 1−S) masks (paper default: True).
        surrogate_device: Device for the frozen surrogate. When ``None``, the
            surrogate is assumed to be on the same device as the explainer.

    Returns:
        Dict with ``"loss"`` (mean scaled MSE) and ``"efficiency_gap"``
        (mean absolute deviation from the efficiency axiom).
    """
    if surrogate_device is None:
        surrogate_device = device
    explainer.eval()
    surrogate.eval()
    total_loss = 0.0
    total_efficiency_gap = 0.0
    total_samples = 0
    num_patches = surrogate.vit.patch_embed.num_patches

    for images, _ in tqdm(loader, desc="Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        B = images.size(0)

        surr_images = images.to(surrogate_device)

        # Null and grand values
        null_mask = torch.zeros(B, num_patches, device=surrogate_device)
        null_logits = surrogate(surr_images, patch_mask=null_mask)
        if target_type == "binary":
            null_probs = torch.sigmoid(null_logits).to(device)
        else:
            null_probs = null_logits.softmax(dim=-1).to(device)

        grand_mask = torch.ones(B, num_patches, device=surrogate_device)
        grand_logits = surrogate(surr_images, patch_mask=grand_mask)
        if target_type == "binary":
            grand_probs = torch.sigmoid(grand_logits).to(device)
        else:
            grand_probs = grand_logits.softmax(dim=-1).to(device)

        # Sample masks and compute surrogate values
        masks = sample_shapley_masks(
            B, num_patches, num_mask_samples, paired=paired, device=device
        )
        masks_flat = masks.flatten(0, 1)
        images_rep = surr_images.repeat_interleave(num_mask_samples, dim=0)
        surr_logits_flat = surrogate(
            images_rep, patch_mask=masks_flat.to(surrogate_device)
        )
        if target_type == "binary":
            surr_flat = torch.sigmoid(surr_logits_flat)
        else:
            surr_flat = surr_logits_flat.softmax(dim=-1)
        surrogate_values = surr_flat.view(B, num_mask_samples, -1).to(
            device
        )  # (B, M, C)

        # Shapley predictions
        phi = explainer(images, grand=grand_probs, null=null_probs)  # (B, n, C)

        # Scaled MSE loss
        v_approx = null_probs.unsqueeze(1) + masks.float() @ phi  # (B, M, C)
        loss = num_patches * F.mse_loss(v_approx, surrogate_values)

        # Efficiency gap: |Σ_i φ'_i(x) − (v(grand) − v(null))|
        phi_sum = phi.sum(dim=1)  # (B, C)
        efficiency_gap = (phi_sum - (grand_probs - null_probs)).abs().mean()

        total_loss += loss.item() * B
        total_efficiency_gap += efficiency_gap.item() * B
        total_samples += B

    return {
        "loss": total_loss / total_samples,
        "efficiency_gap": total_efficiency_gap / total_samples,
    }


def train_explainer(
    explainer: nn.Module,
    surrogate: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    warmup_steps: int = 500,
    num_mask_samples: int = 32,
    paired: bool = True,
    device: torch.device | str = "cpu",
    surrogate_device: Optional[torch.device | str] = None,
    save_dir: Optional[str | os.PathLike] = None,
    use_amp: bool = True,
    gradient_accumulation_steps: int = 1,
    target_type: str = "multiclass",
) -> dict[str, Any]:
    """Full explainer training loop with checkpointing.

    Freezes the surrogate and trains the explainer with AdamW and a cosine LR
    schedule with linear warmup (stepped per gradient update).  Saves the best
    checkpoint (by minimum validation MSE loss) to
    ``<save_dir>/best_explainer.pth``.

    Args:
        explainer: :class:`~vit_shapley.models.ExplainerViT` to train.
        surrogate: Frozen :class:`~vit_shapley.models.SurrogateViT`.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        epochs: Number of training epochs (paper default: 100).
        lr: Peak learning rate for AdamW (paper default: 1e-4).
        weight_decay: L2 regularisation coefficient (paper default: 1e-5).
        warmup_steps: Linear LR warm-up steps (paper default: 500).
        num_mask_samples: Masks sampled per image per step (paper default: 2).
        paired: Use paired (S, 1−S) masks (paper default: True).
        device: Target compute device for the explainer.
        surrogate_device: Device for the frozen surrogate. When ``None``,
            the surrogate is placed on the same device as the explainer.
        save_dir: Directory for checkpoints. Skipped if ``None``.
        use_amp: Enable mixed-precision training (CUDA only).
        gradient_accumulation_steps: Number of micro-batches to accumulate
            before each optimizer step (default: 1).  Set > 1 to reduce peak
            GPU memory while keeping the effective batch size unchanged.

    Returns:
        History dict::

            {
                "train_loss":         [...],   # per-epoch scaled MSE
                "val_loss":           [...],
                "val_efficiency_gap": [...],
                "best_val_loss":      float,
                "best_epoch":         int,
            }
    """
    device = torch.device(device) if isinstance(device, str) else device
    if surrogate_device is None:
        surrogate_device = device
    else:
        surrogate_device = (
            torch.device(surrogate_device)
            if isinstance(surrogate_device, str)
            else surrogate_device
        )
    explainer = explainer.to(device)
    surrogate = surrogate.to(surrogate_device)

    # Freeze the surrogate.
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        explainer.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Scheduler steps once per optimizer step, not per micro-batch
    steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    total_steps = epochs * steps_per_epoch
    scheduler = _cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    history: dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "val_efficiency_gap": [],
        "best_val_loss": float("inf"),
        "best_epoch": -1,
    }

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch_explainer(
            explainer,
            surrogate,
            train_loader,
            optimizer,
            device,
            num_mask_samples=num_mask_samples,
            paired=paired,
            scheduler=scheduler,
            scaler=scaler,
            surrogate_device=surrogate_device,
            gradient_accumulation_steps=gradient_accumulation_steps,
            target_type=target_type,
        )
        val_metrics = evaluate_explainer(
            explainer,
            surrogate,
            val_loader,
            device,
            num_mask_samples=num_mask_samples,
            paired=paired,
            surrogate_device=surrogate_device,
            target_type=target_type,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_efficiency_gap"].append(val_metrics["efficiency_gap"])

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_metrics['loss']:.6f}  "
            f"val_loss={val_metrics['loss']:.6f}  "
            f"eff_gap={val_metrics['efficiency_gap']:.6f}"
        )

        if val_metrics["loss"] < history["best_val_loss"]:
            history["best_val_loss"] = val_metrics["loss"]
            history["best_epoch"] = epoch

            if save_dir is not None:
                ckpt_path = save_dir / "best_explainer.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": explainer.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_metrics["loss"],
                        "val_efficiency_gap": val_metrics["efficiency_gap"],
                    },
                    ckpt_path,
                )
                print(f"  -> Saved best checkpoint to {ckpt_path}")

    return history
