"""Classifier training loop for ViT-Shapley Stage 1.

Implements:
  - ``train_one_epoch``: single epoch training with AMP + grad clipping +
    optional step-level LR scheduling
  - ``evaluate``: validation loop
  - ``train_classifier``: full training orchestration with checkpointing
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def _cosine_with_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    """LR multiplier: linear warmup then cosine decay to 0.

    Matches ``transformers.get_cosine_schedule_with_warmup`` exactly:
    - step < warmup_steps  →  step / warmup_steps  (linear ramp)
    - step >= warmup_steps →  0.5 * (1 + cos(π * progress))  (cosine decay)

    Args:
        step: Current global step (0-indexed, incremented after each batch).
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.

    Returns:
        Multiplicative factor for the base learning rate.
    """
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    target_type: str = "multiclass",
) -> dict[str, float]:
    """Run one training epoch.

    Args:
        model: The classifier (already on *device*).
        loader: Training DataLoader.
        optimizer: Optimizer (e.g. AdamW).
        criterion: Loss function (e.g. CrossEntropyLoss).
        device: Target device.
        scaler: :class:`torch.amp.GradScaler` for mixed-precision training,
                or ``None`` to disable AMP.
        scheduler: Step-level LR scheduler stepped after every optimizer update,
                   or ``None`` to skip scheduling.

    Returns:
        Dict with keys ``"loss"`` (mean) and ``"acc"`` (top-1 accuracy 0-1).
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        use_amp = scaler is not None
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            if target_type == "binary":
                loss = criterion(logits, labels.float().unsqueeze(1))
            else:
                loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        if target_type == "binary":
            total_correct += (
                (logits.squeeze(-1) > 0).long() == labels
            ).sum().item()
        else:
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += bs

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_type: str = "multiclass",
) -> dict[str, float]:
    """Evaluate the model on *loader*.

    Args:
        model: The classifier (already on *device*).
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Target device.

    Returns:
        Dict with keys ``"loss"`` (mean) and ``"acc"`` (top-1 accuracy 0-1).
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(loader, desc="Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        if target_type == "binary":
            loss = criterion(logits, labels.float().unsqueeze(1))
        else:
            loss = criterion(logits, labels)

        bs = images.size(0)
        total_loss += loss.item() * bs
        if target_type == "binary":
            total_correct += (
                (logits.squeeze(-1) > 0).long() == labels
            ).sum().item()
        else:
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += bs

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
    }


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 25,
    lr: float = 1e-5,
    weight_decay: float = 1e-5,
    warmup_steps: int = 500,
    device: torch.device | str = "cpu",
    save_dir: str | os.PathLike | None = None,
    use_amp: bool = True,
    target_type: str = "multiclass",
) -> dict[str, Any]:
    """Full training loop with checkpointing.

    Uses AdamW with a cosine decay schedule with linear warmup (step-level),
    AMP, and gradient clipping. Saves the best checkpoint (by validation
    accuracy) to ``<save_dir>/best_classifier.pth``.

    Hyperparameter defaults match the original ViT-Shapley paper:
    ``lr=1e-5``, ``weight_decay=1e-5``, ``warmup_steps=500``, ``epochs=25``.

    Args:
        model: Initialised classifier (moved to *device* inside this function).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        epochs: Number of training epochs.
        lr: Peak learning rate for AdamW.
        weight_decay: L2 regularisation for AdamW.
        warmup_steps: Steps of linear LR warmup before cosine decay.
        device: Device string or :class:`torch.device`.
        save_dir: Directory for saving checkpoints. Skipped if ``None``.
        use_amp: Enable mixed-precision training (requires CUDA).

    Returns:
        History dict::

            {
                "train_loss": [...],   # per-epoch
                "train_acc":  [...],
                "val_loss":   [...],
                "val_acc":    [...],
                "best_val_acc": float,
                "best_epoch":   int,
            }
    """
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)

    if target_type == "binary":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_with_warmup(step, warmup_steps, total_steps),
    )

    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    history: dict[str, Any] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_val_acc": -1.0,
        "best_epoch": -1,
    }

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, scheduler,
            target_type=target_type,
        )
        val_metrics = evaluate(model, val_loader, criterion, device,
                               target_type=target_type)
        # scheduler is stepped per batch inside train_one_epoch; no epoch step here

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_metrics['loss']:.4f}  train_acc={train_metrics['acc']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['acc']:.4f}"
        )

        if val_metrics["acc"] > history["best_val_acc"]:
            history["best_val_acc"] = val_metrics["acc"]
            history["best_epoch"] = epoch

            if save_dir is not None:
                ckpt_path = save_dir / "best_classifier.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": val_metrics["acc"],
                        "val_loss": val_metrics["loss"],
                    },
                    ckpt_path,
                )
                print(f"  -> Saved best checkpoint to {ckpt_path}")

    return history
