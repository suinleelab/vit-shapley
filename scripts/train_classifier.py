#!/usr/bin/env python
"""CLI entry point for Stage 1: train a ViT classifier on ImageNette.

Example
-------
    python scripts/train_classifier.py \\
        --model-name vit_tiny_patch16_224 \\
        --epochs 10 \\
        --batch-size 64 \\
        --save-dir checkpoints/classifier
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Allow running as `python scripts/train_classifier.py` without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vit_shapley.data import get_imagenette_dataset
from vit_shapley.models import build_vit_classifier
from vit_shapley.training import train_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ViT classifier on ImageNette (Stage 1 of ViT-Shapley).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/local-b/chanwkim/vit-shapley-data",
        help="Root directory for ImageNette data. Dataset is downloaded if absent.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="vit_base_patch16_224",
        help="timm model name (e.g. vit_tiny_patch16_224, vit_base_patch16_224).",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load ImageNet-pretrained weights from timm hub.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Peak learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Steps of linear LR warmup before cosine decay.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image spatial resolution.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints/classifier",
        help="Directory where the best checkpoint is saved.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        default=False,
        help="Disable automatic mixed precision (AMP).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device string (e.g. 'cuda', 'cpu'). Auto-detected if empty.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    print("Loading datasets …")
    train_dataset = get_imagenette_dataset(
        root=args.data_root,
        split="train",
        image_size=args.image_size,
        download=True,
    )
    val_dataset = get_imagenette_dataset(
        root=args.data_root,
        split="val",
        image_size=args.image_size,
        download=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    num_classes = len(train_dataset.classes)
    print(f"Classes ({num_classes}): {train_dataset.classes}")

    # Model
    print(f"Building model: {args.model_name} (pretrained={args.pretrained})")
    model = build_vit_classifier(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=args.pretrained,
    )

    # Train
    use_amp = not args.no_amp
    history = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        device=device,
        save_dir=args.save_dir,
        use_amp=use_amp,
    )

    print(
        f"\nTraining complete. "
        f"Best val acc: {history['best_val_acc']:.4f} at epoch {history['best_epoch']}."
    )


if __name__ == "__main__":
    main()
