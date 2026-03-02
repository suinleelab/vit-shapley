"""Surrogate model training loop for ViT-Shapley Stage 2.

Implements:
  - ``sample_subset_masks``: sample patch masks for surrogate training
  - ``train_one_epoch_surrogate``: single epoch with KL divergence loss
  - ``evaluate_surrogate``: validation loop
  - ``train_surrogate``: full training with checkpointing
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


def _cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine LR decay with linear warmup (mirrors HuggingFace get_cosine_schedule_with_warmup).

    During warmup, LR rises linearly from 0 to peak.  After warmup it follows
    a cosine curve from peak down to 0.  The schedule is stepped once per
    gradient update (not per epoch).

    Args:
        optimizer: The optimizer whose LR will be scheduled.
        warmup_steps: Number of warm-up steps (reference default: 500).
        total_steps: Total number of gradient updates across all epochs.

    Returns:
        :class:`torch.optim.lr_scheduler.LambdaLR` ready to be stepped after
        each optimizer step.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def sample_subset_masks(
    batch_size: int,
    num_patches: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample random binary patch masks for surrogate training.

    For each sample a random threshold ``t ~ U(0,1)`` is drawn and each patch
    is independently included when ``rand() > t``.  This produces a uniform
    distribution over cardinalities ``{0, 1, ..., num_patches}`` with
    independent Bernoulli within-cardinality sampling.

    Args:
        batch_size: Number of masks to generate.
        num_patches: Total number of patches (e.g. 196 for a 14×14 grid).
        device: Target device for the returned tensor.
        generator: Optional :class:`torch.Generator` for reproducible sampling.
            Pass a seeded generator to get deterministic masks (used for
            validation); ``None`` uses the global RNG.

    Returns:
        Float tensor ``(batch_size, num_patches)`` — ``1.0`` = visible,
        ``0.0`` = masked.
    """
    rand_vals = torch.rand(
        batch_size, num_patches, device=device, generator=generator,
    )
    thresholds = torch.rand(
        batch_size, 1, device=device, generator=generator,
    )
    masks = (rand_vals > thresholds).float()
    return masks


def train_one_epoch_surrogate(
    surrogate: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    classifier_device: Optional[torch.device] = None,
) -> dict[str, float]:
    """Run one surrogate training epoch.

    Minimises ``DKL(f(x;η) || g(x_s;β))`` where ``f`` is the frozen
    classifier (teacher) and ``g`` is the surrogate, evaluated on
    randomly sampled patch subsets ``s``.

    Args:
        surrogate: The :class:`~vit_shapley.models.SurrogateViT` being trained.
        classifier: Frozen teacher classifier (frozen outside this function).
        loader: Training DataLoader.
        optimizer: Optimizer for surrogate parameters.
        device: Target device for the surrogate.
        scaler: :class:`torch.cuda.amp.GradScaler` for AMP, or ``None``.
        scheduler: Step-level LR scheduler (stepped once per gradient update).
            ``None`` disables LR scheduling within the epoch.
        classifier_device: Device for the classifier. When ``None``, the
            classifier is assumed to be on the same device as the surrogate.

    Returns:
        Dict with ``"loss"`` (mean KL divergence per sample) and ``"acc"``
        (top-1 accuracy of the surrogate on randomly masked inputs).
    """
    if classifier_device is None:
        classifier_device = device
    surrogate.train()
    classifier.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_patches = surrogate.vit.patch_embed.num_patches

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.size(0)

        patch_mask = sample_subset_masks(B, num_patches, device)

        optimizer.zero_grad()

        use_amp = scaler is not None
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            with torch.no_grad():
                clf_autocast = classifier_device.type == "cuda"
                with torch.amp.autocast(device_type=classifier_device.type, enabled=clf_autocast):
                    teacher_logits = classifier(images.to(classifier_device))
            teacher_probs = teacher_logits.to(device).softmax(dim=-1)

            surrogate_logits = surrogate(images, patch_mask=patch_mask)
            surrogate_log_probs = surrogate_logits.log_softmax(dim=-1)

            # DKL(teacher || surrogate) — Eq. 2 / 14 of the paper
            loss = F.kl_div(surrogate_log_probs, teacher_probs, reduction="batchmean")

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(surrogate.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(surrogate.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * B
        total_correct += (surrogate_logits.detach().argmax(dim=1) == labels).sum().item()
        total_samples += B

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
    }


@torch.no_grad()
def evaluate_surrogate(
    surrogate: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
    val_seed: int = 0,
    classifier_device: Optional[torch.device] = None,
) -> dict[str, float]:
    """Evaluate the surrogate on the validation set.

    Computes the mean KL divergence between the frozen teacher and the
    surrogate over deterministically sampled masks (seeded by ``val_seed``),
    plus the top-1 accuracy of the surrogate on fully visible inputs.

    Using a fixed ``val_seed`` ensures that the same masks are applied to the
    same images on every call, making the validation metric fully reproducible
    across epochs (matching the reference implementation's cached val masks).

    Args:
        surrogate: The :class:`~vit_shapley.models.SurrogateViT` to evaluate.
        classifier: Frozen teacher classifier.
        loader: Validation DataLoader (should have ``shuffle=False``).
        device: Target device for the surrogate.
        val_seed: Seed for the validation mask generator.  All calls with the
            same seed produce identical masks, giving reproducible val KL.
        classifier_device: Device for the classifier. When ``None``, the
            classifier is assumed to be on the same device as the surrogate.

    Returns:
        Dict with ``"loss"`` (mean KL divergence) and ``"acc"`` (top-1
        accuracy on full-image surrogate predictions).
    """
    if classifier_device is None:
        classifier_device = device
    surrogate.eval()
    classifier.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_patches = surrogate.vit.patch_embed.num_patches

    # Seeded generator so the same masks are used every call (reproducible val).
    gen = torch.Generator(device=device)
    gen.manual_seed(val_seed)

    for images, labels in tqdm(loader, desc="Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.size(0)

        patch_mask = sample_subset_masks(B, num_patches, device, generator=gen)

        teacher_probs = classifier(images.to(classifier_device)).to(device).softmax(dim=-1)
        surrogate_log_probs = surrogate(images, patch_mask=patch_mask).log_softmax(dim=-1)
        loss = F.kl_div(surrogate_log_probs, teacher_probs, reduction="batchmean")

        # Accuracy on full-image surrogate predictions (mask=None)
        full_logits = surrogate(images, patch_mask=None)
        total_correct += (full_logits.argmax(dim=1) == labels).sum().item()

        total_loss += loss.item() * B
        total_samples += B

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
    }


def train_surrogate(
    surrogate: nn.Module,
    classifier: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 50,
    lr: float = 1e-5,
    weight_decay: float = 1e-5,
    warmup_steps: int = 500,
    device: torch.device | str = "cpu",
    classifier_device: Optional[torch.device | str] = None,
    save_dir: Optional[str | os.PathLike] = None,
    use_amp: bool = True,
) -> dict[str, Any]:
    """Full surrogate training loop with checkpointing.

    Freezes the classifier and fine-tunes the surrogate with AdamW and a
    cosine schedule with linear warmup (matching the reference).  Saves the
    best checkpoint (by minimum validation KL) to
    ``<save_dir>/best_surrogate.pth``.

    Args:
        surrogate: :class:`~vit_shapley.models.SurrogateViT` to fine-tune.
        classifier: Frozen teacher classifier.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        epochs: Number of fine-tuning epochs (paper default: 50).
        lr: Peak learning rate for AdamW (paper default: 1e-5).
        weight_decay: AdamW weight decay (paper default: 1e-5).
        warmup_steps: Number of gradient steps for linear LR warm-up
            (paper default: 500).
        device: Target compute device for the surrogate.
        classifier_device: Device for the frozen classifier. When ``None``,
            the classifier is placed on the same device as the surrogate.
        save_dir: Directory for checkpoints. Skipped if ``None``.
        use_amp: Enable mixed-precision training (CUDA only).

    Returns:
        History dict::

            {
                "train_loss": [...],   # per-epoch KL divergence
                "val_loss":   [...],
                "val_acc":    [...],
                "best_val_loss": float,
                "best_epoch":    int,
            }
    """
    device = torch.device(device) if isinstance(device, str) else device
    if classifier_device is None:
        classifier_device = device
    else:
        classifier_device = (
            torch.device(classifier_device)
            if isinstance(classifier_device, str)
            else classifier_device
        )
    surrogate = surrogate.to(device)
    classifier = classifier.to(classifier_device)

    # Freeze the teacher.
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    total_steps = len(train_loader) * epochs
    optimizer = torch.optim.AdamW(
        surrogate.parameters(), lr=lr, weight_decay=weight_decay
    )
    # Step-level cosine schedule with linear warmup (reference uses HuggingFace
    # get_cosine_schedule_with_warmup with warmup_steps=500 and step interval).
    scheduler = _cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    history: dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "best_val_loss": float("inf"),
        "best_epoch": -1,
    }

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch_surrogate(
            surrogate, classifier, train_loader, optimizer, device, scaler, scheduler,
            classifier_device=classifier_device,
        )
        val_metrics = evaluate_surrogate(
            surrogate, classifier, val_loader, device, val_seed=0,
            classifier_device=classifier_device,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_kl={train_metrics['loss']:.4f}  "
            f"val_kl={val_metrics['loss']:.4f}  val_acc={val_metrics['acc']:.4f}"
        )

        if val_metrics["loss"] < history["best_val_loss"]:
            history["best_val_loss"] = val_metrics["loss"]
            history["best_epoch"] = epoch

            if save_dir is not None:
                ckpt_path = save_dir / "best_surrogate.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": surrogate.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_metrics["loss"],
                        "val_acc": val_metrics["acc"],
                    },
                    ckpt_path,
                )
                print(f"  -> Saved best checkpoint to {ckpt_path}")

    return history
