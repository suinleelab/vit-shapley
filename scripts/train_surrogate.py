#!/usr/bin/env python
"""CLI entry point for Stage 2: fine-tune a ViT surrogate on ImageNette.

The surrogate is initialised from a Stage 1 classifier checkpoint and then
fine-tuned to handle randomly masked image patches via attention masking,
minimising the KL divergence between the classifier and surrogate outputs
(Eq. 2 of the ViT-Shapley paper, ICLR 2023).

Example
-------
    python scripts/train_surrogate.py \\
        --model-name vit_tiny_patch16_224 \\
        --classifier-ckpt checkpoints/classifier/best_classifier.pth \\
        --epochs 50 \\
        --batch-size 64 \\
        --save-dir checkpoints/surrogate
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vit_shapley.data import get_imagenette_dataset
from vit_shapley.models import build_vit_classifier, build_vit_surrogate
from vit_shapley.training import train_surrogate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ViT surrogate on ImageNette (Stage 2 of ViT-Shapley).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/local-b/chanwkim/vit-shapley-data",
        help="Root directory for ImageNette data.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="vit_base_patch16_224",
        help="timm model name (must match the classifier checkpoint).",
    )
    parser.add_argument(
        "--classifier-ckpt",
        type=str,
        required=True,
        help="Path to best_classifier.pth from Stage 1.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of fine-tuning epochs.",
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
        help="Number of gradient steps for linear LR warm-up.",
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
        default="checkpoints/surrogate",
        help="Directory where the best surrogate checkpoint is saved.",
    )
    parser.add_argument(
        "--masking-strategy",
        type=str,
        default="attn_mask",
        choices=["attn_mask", "zero_input"],
        help=(
            "Patch masking strategy: 'attn_mask' (attention bias, default) or "
            "'zero_input' (zero out masked patch pixels)."
        ),
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

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    print("Loading datasets …")
    train_dataset = get_imagenette_dataset(
        root=args.data_root, split="train", image_size=args.image_size, download=True
    )
    val_dataset = get_imagenette_dataset(
        root=args.data_root, split="val", image_size=args.image_size, download=False
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
    )

    num_classes = len(train_dataset.classes)
    print(f"Classes ({num_classes}): {train_dataset.classes}")

    # Build surrogate (initialised from classifier checkpoint).
    print(f"Building surrogate: {args.model_name} (masking={args.masking_strategy})")
    surrogate = build_vit_surrogate(
        model_name=args.model_name,
        num_classes=num_classes,
        classifier_ckpt_path=args.classifier_ckpt,
        masking_strategy=args.masking_strategy,
    )

    # Build a separate frozen classifier instance to serve as the teacher.
    print(f"Loading classifier teacher from: {args.classifier_ckpt}")
    classifier = build_vit_classifier(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=False,
    )
    ckpt = torch.load(args.classifier_ckpt, map_location="cpu", weights_only=True)
    classifier.load_state_dict(ckpt.get("model_state_dict", ckpt))

    history = train_surrogate(
        surrogate=surrogate,
        classifier=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        device=device,
        save_dir=args.save_dir,
        use_amp=not args.no_amp,
    )

    print(
        f"\nTraining complete. "
        f"Best val KL: {history['best_val_loss']:.4f} at epoch {history['best_epoch']}."
    )


if __name__ == "__main__":
    main()
