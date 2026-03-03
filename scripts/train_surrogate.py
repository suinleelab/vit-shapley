#!/usr/bin/env python
"""CLI entry point for Stage 2: fine-tune a ViT surrogate on ImageNette.

The surrogate is initialised from a Stage 1 classifier checkpoint and then
fine-tuned to handle randomly masked image patches via attention masking,
minimising the KL divergence between the classifier and surrogate outputs
(Eq. 2 of the ViT-Shapley paper, ICLR 2023).

Example
-------
    python scripts/train_surrogate.py --config configs/surrogate.yaml

    # Quick override:
    python scripts/train_surrogate.py --config configs/surrogate.yaml \\
        --set model_name=vit_tiny_patch16_224 epochs=50 save_dir=checkpoints/surrogate_attn
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vit_shapley.configs import SurrogateConfig, load_config
from vit_shapley.data import get_dataset, resolve_num_classes
from vit_shapley.models import build_vit_classifier, build_vit_surrogate
from vit_shapley.training import train_surrogate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a ViT surrogate on ImageNette (Stage 2 of ViT-Shapley).",
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
        help="Override config values, e.g. --set lr=1e-3 epochs=50",
    )
    args = parser.parse_args()
    cfg = load_config(SurrogateConfig, args.config, args.set, env_path=args.env)

    device = (
        torch.device(cfg.device)
        if cfg.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    classifier_device = (
        torch.device(cfg.classifier_device) if cfg.classifier_device else device
    )
    print(f"Using device: {device} (classifier: {classifier_device})")

    print("Loading datasets …")
    train_dataset = get_dataset(
        cfg.dataset, root=cfg.data_root, split="train", image_size=cfg.image_size, download=True
    )
    val_dataset = get_dataset(
        cfg.dataset, root=cfg.data_root, split="val", image_size=cfg.image_size, download=False
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

    num_classes = resolve_num_classes(train_dataset, cfg.target_type)
    print(f"Classes ({len(train_dataset.classes)}): {train_dataset.classes}")

    # Build surrogate (initialised from classifier checkpoint).
    print(f"Building surrogate: {cfg.model_name} (masking={cfg.masking_strategy})")
    surrogate = build_vit_surrogate(
        model_name=cfg.model_name,
        num_classes=num_classes,
        classifier_ckpt_path=cfg.classifier_ckpt,
        masking_strategy=cfg.masking_strategy,
    )

    # Build a separate frozen classifier instance to serve as the teacher.
    print(f"Loading classifier teacher from: {cfg.classifier_ckpt}")
    classifier = build_vit_classifier(
        model_name=cfg.model_name,
        num_classes=num_classes,
        pretrained=False,
    )
    ckpt = torch.load(cfg.classifier_ckpt, map_location="cpu", weights_only=True)
    classifier.load_state_dict(ckpt.get("model_state_dict", ckpt))

    history = train_surrogate(
        surrogate=surrogate,
        classifier=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        device=device,
        classifier_device=classifier_device,
        save_dir=cfg.save_dir,
        use_amp=cfg.use_amp,
        target_type=cfg.target_type,
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
        f"Best val KL: {history['best_val_loss']:.4f} at epoch {history['best_epoch']}."
    )


if __name__ == "__main__":
    main()
