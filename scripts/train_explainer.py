#!/usr/bin/env python
"""CLI entry point for Stage 3: train a ViT explainer on ImageNette.

The explainer is initialised from a Stage 2 surrogate checkpoint and then
trained to produce per-patch Shapley value estimates in a single forward pass.
Training minimises a scaled MSE objective with masks sampled from the Shapley
distribution (ViT-Shapley paper, ICLR 2023).

Example
-------
    python scripts/train_explainer.py --config configs/explainer.yaml

    # Quick override:
    python scripts/train_explainer.py --config configs/explainer.yaml \\
        --set model_name=vit_tiny_patch16_224 epochs=100 lr=1e-4
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vit_shapley.configs import ExplainerConfig, load_config
from vit_shapley.data import get_imagenette_dataset
from vit_shapley.models import build_vit_explainer, build_vit_surrogate
from vit_shapley.training import train_explainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a ViT explainer on ImageNette (Stage 3 of ViT-Shapley).",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file."
    )
    parser.add_argument(
        "--env",
        type=str,
        default=".env",
        help="Path to .env file for $variable resolution (default: .env).",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. --set lr=1e-4 epochs=100",
    )
    args = parser.parse_args()
    cfg = load_config(ExplainerConfig, args.config, args.set, env_path=args.env)

    device = (
        torch.device(cfg.device)
        if cfg.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    surrogate_device = (
        torch.device(cfg.surrogate_device) if cfg.surrogate_device else device
    )
    print(f"Using device: {device} (surrogate: {surrogate_device})")

    print("Loading datasets …")
    train_dataset = get_imagenette_dataset(
        root=cfg.data_root, split="train", image_size=cfg.image_size, download=True
    )
    val_dataset = get_imagenette_dataset(
        root=cfg.data_root, split="val", image_size=cfg.image_size, download=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    num_classes = len(train_dataset.classes)
    print(f"Classes ({num_classes}): {train_dataset.classes}")

    # Build frozen surrogate from the Stage 2 checkpoint.
    print(f"Loading surrogate from: {cfg.surrogate_ckpt}")
    surrogate = build_vit_surrogate(
        model_name=cfg.model_name,
        num_classes=num_classes,
        classifier_ckpt_path=None,
        masking_strategy=cfg.masking_strategy,
    )
    ckpt = torch.load(cfg.surrogate_ckpt, map_location="cpu", weights_only=True)
    sd = ckpt.get("model_state_dict", ckpt)
    surrogate.load_state_dict(sd)

    # Build explainer initialised from surrogate backbone.
    print(f"Building explainer: {cfg.model_name}")
    explainer = build_vit_explainer(
        model_name=cfg.model_name,
        num_classes=num_classes,
        surrogate_ckpt_path=cfg.surrogate_ckpt,
    )

    history = train_explainer(
        explainer=explainer,
        surrogate=surrogate,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        num_mask_samples=cfg.num_mask_samples,
        paired=cfg.paired_masks,
        device=device,
        surrogate_device=surrogate_device,
        save_dir=cfg.save_dir,
        use_amp=cfg.use_amp,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    )

    # Save config and training history as JSON
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w") as f:
        json.dump(cfg.model_dump(), f, indent=2)
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(
        f"\nTraining complete. "
        f"Best val loss: {history['best_val_loss']:.6f} at epoch {history['best_epoch']}."
    )


if __name__ == "__main__":
    main()
