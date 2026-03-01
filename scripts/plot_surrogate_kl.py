#!/usr/bin/env python
"""CLI: plot KL divergence vs. mask cardinality (Figure 2 of ViT-Shapley).

Evaluates four model variants against a frozen classifier across a range of
masked-patch cardinalities and saves a publication-quality figure showing four
lines:

  - Blue solid:   attn_mask strategy, fine-tuned surrogate weights
  - Blue dotted:  attn_mask strategy, original classifier weights (un-finetuned)
  - Red solid:    zero_input strategy, fine-tuned surrogate weights
  - Red dotted:   zero_input strategy, original classifier weights (un-finetuned)

Each line shows mean KL(classifier(full_image) || surrogate(masked_image))
with a shaded 95% confidence interval, as a function of the number of visible
patches.

Example
-------
    python scripts/plot_surrogate_kl.py \\
        --classifier-ckpt  checkpoints/classifier/best_classifier.pth \\
        --attn-surrogate-ckpt checkpoints/surrogate_attn/best_surrogate.pth \\
        --zero-surrogate-ckpt checkpoints/surrogate_zero/best_surrogate.pth \\
        --model-name vit_tiny_patch16_224 \\
        --num-images 50 --num-masks 50 --step 10 \\
        --output figures/surrogate_kl.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless server — must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vit_shapley.data import get_imagenette_dataset
from vit_shapley.evaluation import compute_kl_vs_cardinality
from vit_shapley.models import build_vit_classifier, build_vit_surrogate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot surrogate KL divergence vs. mask cardinality (Figure 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--classifier-ckpt",
        type=str,
        required=True,
        help="Path to best_classifier.pth (Stage 1 output).",
    )
    parser.add_argument(
        "--attn-surrogate-ckpt",
        type=str,
        required=True,
        help="Path to best_surrogate.pth trained with attn_mask strategy.",
    )
    parser.add_argument(
        "--zero-surrogate-ckpt",
        type=str,
        required=True,
        help="Path to best_surrogate.pth trained with zero_input strategy.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="vit_base_patch16_224",
        help="timm model name (must match all checkpoints).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/local-b/chanwkim/vit-shapley-data",
        help="Root directory for ImageNette data.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=50,
        help="Number of validation images to use.",
    )
    parser.add_argument(
        "--num-masks",
        type=int,
        default=50,
        help="Number of random masks to sample per cardinality.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Cardinality step size (patches).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/surrogate_kl.png",
        help="Output path for the saved figure.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device string (e.g. 'cuda', 'cpu'). Auto-detected if empty.",
    )
    return parser.parse_args()


def _load_surrogate_ckpt(surrogate, ckpt_path: str) -> None:
    """Load a surrogate checkpoint into a SurrogateViT model.

    Handles both full SurrogateViT state dicts (keys prefixed "vit.") and
    bare inner-vit state dicts.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    if all(k.startswith("vit.") for k in state_dict):
        surrogate.load_state_dict(state_dict)
    else:
        surrogate.vit.load_state_dict(state_dict)


def _load_classifier_into_surrogate(surrogate, ckpt_path: str) -> None:
    """Load a classifier checkpoint into the inner .vit of a SurrogateViT."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    surrogate.vit.load_state_dict(state_dict)


def main() -> None:
    args = parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ data --
    print("Loading validation dataset …")
    val_dataset = get_imagenette_dataset(
        root=args.data_root, split="val", image_size=224, download=False
    )
    indices = list(range(min(args.num_images, len(val_dataset))))
    subset = Subset(val_dataset, indices)
    loader = DataLoader(subset, batch_size=args.num_images, shuffle=False, num_workers=2)
    images, _ = next(iter(loader))
    images = images.to(device)
    print(f"  {images.shape[0]} images loaded.")

    # --------------------------------------------------------------- models --
    num_classes = len(val_dataset.classes)

    print(f"Loading classifier from {args.classifier_ckpt} …")
    classifier = build_vit_classifier(
        model_name=args.model_name, num_classes=num_classes, pretrained=False
    )
    ckpt = torch.load(args.classifier_ckpt, map_location="cpu", weights_only=True)
    classifier.load_state_dict(ckpt.get("model_state_dict", ckpt))
    classifier.to(device).eval()

    # Line 1 (blue solid): attn_mask + fine-tuned surrogate weights
    print(f"Loading attn_mask surrogate from {args.attn_surrogate_ckpt} …")
    surr_attn = build_vit_surrogate(
        model_name=args.model_name, num_classes=num_classes, masking_strategy="attn_mask"
    )
    _load_surrogate_ckpt(surr_attn, args.attn_surrogate_ckpt)
    surr_attn.to(device).eval()

    # Line 2 (blue dotted): attn_mask + classifier weights (un-finetuned baseline)
    print(f"Loading attn_mask baseline (classifier weights) …")
    base_attn = build_vit_surrogate(
        model_name=args.model_name, num_classes=num_classes, masking_strategy="attn_mask"
    )
    _load_classifier_into_surrogate(base_attn, args.classifier_ckpt)
    base_attn.to(device).eval()

    # Line 3 (red solid): zero_input + fine-tuned surrogate weights
    print(f"Loading zero_input surrogate from {args.zero_surrogate_ckpt} …")
    surr_zero = build_vit_surrogate(
        model_name=args.model_name, num_classes=num_classes, masking_strategy="zero_input"
    )
    _load_surrogate_ckpt(surr_zero, args.zero_surrogate_ckpt)
    surr_zero.to(device).eval()

    # Line 4 (red dotted): zero_input + classifier weights (un-finetuned baseline)
    print(f"Loading zero_input baseline (classifier weights) …")
    base_zero = build_vit_surrogate(
        model_name=args.model_name, num_classes=num_classes, masking_strategy="zero_input"
    )
    _load_classifier_into_surrogate(base_zero, args.classifier_ckpt)
    base_zero.to(device).eval()

    # Infer num_patches from any of the surrogate models.
    num_patches: int = surr_attn.vit.patch_embed.num_patches
    print(f"  num_patches = {num_patches}")

    # ----------------------------------------------------------- evaluation --
    series = [
        (surr_attn, "Attn mask – surrogate",   "steelblue", "-"),
        (base_attn, "Attn mask – classifier",  "steelblue", "--"),
        (surr_zero, "Zero input – surrogate",  "firebrick",  "-"),
        (base_zero, "Zero input – classifier", "firebrick",  "--"),
    ]

    print(
        f"Computing KL vs. cardinality "
        f"(step={args.step}, masks_per_cardinality={args.num_masks}) …"
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for model, label, color, ls in series:
        print(f"  Evaluating: {label} …")
        results = compute_kl_vs_cardinality(
            surrogate=model,
            classifier=classifier,
            images=images,
            num_patches=num_patches,
            num_masks_per_cardinality=args.num_masks,
            cardinality_step=args.step,
            device=device,
        )
        cardinalities = sorted(results.keys())
        means = np.array([np.mean(results[m]) for m in cardinalities])
        stds = np.array([np.std(results[m], ddof=1) for m in cardinalities])
        ns = np.array([len(results[m]) for m in cardinalities])
        ci95 = 1.96 * stds / np.sqrt(ns)

        ax.plot(cardinalities, means, color=color, linestyle=ls, linewidth=2, label=label)
        ax.fill_between(
            cardinalities,
            means - ci95,
            means + ci95,
            alpha=0.2,
            color=color,
        )

    # ----------------------------------------------------------------- plot --
    ax.set_xlabel("Number of visible patches", fontsize=13)
    ax.set_ylabel("KL divergence", fontsize=13)
    ax.set_title(
        f"KL Divergence vs. Mask Cardinality\n"
        f"({args.model_name}, {images.shape[0]} images, {args.num_masks} masks)",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()
