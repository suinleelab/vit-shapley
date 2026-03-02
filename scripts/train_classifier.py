#!/usr/bin/env python
"""CLI entry point for Stage 1: train a ViT classifier on ImageNette.

Example
-------
    python scripts/train_classifier.py --config configs/classifier.yaml

    # Quick override:
    python scripts/train_classifier.py --config configs/classifier.yaml \\
        --set model_name=vit_tiny_patch16_224 epochs=10
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Allow running as `python scripts/train_classifier.py` without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vit_shapley.configs import ClassifierConfig, load_config
from vit_shapley.data import get_imagenette_dataset
from vit_shapley.models import build_vit_classifier
from vit_shapley.training import train_classifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a ViT classifier on ImageNette (Stage 1 of ViT-Shapley).",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
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
        help="Override config values, e.g. --set lr=1e-3 epochs=10",
    )
    args = parser.parse_args()
    cfg = load_config(ClassifierConfig, args.config, args.set, env_path=args.env)

    # Device selection
    if cfg.device:
        device = torch.device(cfg.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    print("Loading datasets …")
    train_dataset = get_imagenette_dataset(
        root=cfg.data_root,
        split="train",
        image_size=cfg.image_size,
        download=True,
    )
    val_dataset = get_imagenette_dataset(
        root=cfg.data_root,
        split="val",
        image_size=cfg.image_size,
        download=False,
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

    # Model
    print(f"Building model: {cfg.model_name} (pretrained={cfg.pretrained})")
    model = build_vit_classifier(
        model_name=cfg.model_name,
        num_classes=num_classes,
        pretrained=cfg.pretrained,
    )

    # Train
    history = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        device=device,
        save_dir=cfg.save_dir,
        use_amp=cfg.use_amp,
    )

    print(
        f"\nTraining complete. "
        f"Best val acc: {history['best_val_acc']:.4f} at epoch {history['best_epoch']}."
    )


if __name__ == "__main__":
    main()
