#!/usr/bin/env python
"""CLI: plot KL divergence vs. mask cardinality (Figure 2 of ViT-Shapley).

Evaluates four model variants against a frozen classifier across a range of
masked-patch cardinalities and saves a publication-quality figure showing four
lines:

  - Blue dotted:  attn_mask strategy, fine-tuned surrogate weights
  - Blue solid:   attn_mask strategy, original classifier weights (un-finetuned)
  - Red dotted:   zero_input strategy, fine-tuned surrogate weights
  - Red solid:    zero_input strategy, original classifier weights (un-finetuned)

Each line shows mean KL(classifier(full_image) || surrogate(masked_image))
with a shaded 95% confidence interval, as a function of the number of visible
patches.

Example
-------
    python scripts/plot_surrogate_kl.py --config configs/plot_surrogate_kl.yaml

    # Quick override:
    python scripts/plot_surrogate_kl.py --config configs/plot_surrogate_kl.yaml \\
        --set model_name=vit_tiny_patch16_224 num_images=50 num_masks=50
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless server — must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from torch.utils.data import DataLoader, Subset

# ── Paired-palette colours (RGB / 256) ──────────────────────────────────────
_COLOR_ATTN = np.array([31, 120, 180]) / 256  # Paired index 1
_COLOR_ZERO = np.array([227, 26, 28]) / 256  # Paired index 5

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vit_shapley.configs import PlotConfig, load_config
from vit_shapley.data import get_dataset
from vit_shapley.evaluation import compute_kl_vs_cardinality
from vit_shapley.models import build_vit_classifier, build_vit_surrogate


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
    parser = argparse.ArgumentParser(
        description="Plot surrogate KL divergence vs. mask cardinality (Figure 2).",
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
        help="Override config values, e.g. --set num_images=100 step=5",
    )
    args = parser.parse_args()
    cfg = load_config(PlotConfig, args.config, args.set, env_path=args.env)

    device = (
        torch.device(cfg.device)
        if cfg.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ data --
    print("Loading validation dataset …")
    val_dataset = get_dataset(
        cfg.dataset, root=cfg.data_root, split="val", image_size=224, download=False
    )
    indices = list(range(min(cfg.num_images, len(val_dataset))))
    subset = Subset(val_dataset, indices)
    loader = DataLoader(subset, batch_size=cfg.num_images, shuffle=False, num_workers=2)
    images, _ = next(iter(loader))
    images = images.to(device)
    print(f"  {images.shape[0]} images loaded.")

    # --------------------------------------------------------------- models --
    num_classes = len(val_dataset.classes)

    print(f"Loading classifier from {cfg.classifier_ckpt} …")
    classifier = build_vit_classifier(
        model_name=cfg.model_name, num_classes=num_classes, pretrained=False
    )
    ckpt = torch.load(cfg.classifier_ckpt, map_location="cpu", weights_only=True)
    classifier.load_state_dict(ckpt.get("model_state_dict", ckpt))
    classifier.to(device).eval()

    # Line 1 (blue dotted): attn_mask + fine-tuned surrogate weights
    print(f"Loading attn_mask surrogate from {cfg.attn_surrogate_ckpt} …")
    surr_attn = build_vit_surrogate(
        model_name=cfg.model_name, num_classes=num_classes, masking_strategy="attn_mask"
    )
    _load_surrogate_ckpt(surr_attn, cfg.attn_surrogate_ckpt)
    surr_attn.to(device).eval()

    # Line 2 (blue solid): attn_mask + classifier weights (un-finetuned baseline)
    print(f"Loading attn_mask baseline (classifier weights) …")
    base_attn = build_vit_surrogate(
        model_name=cfg.model_name, num_classes=num_classes, masking_strategy="attn_mask"
    )
    _load_classifier_into_surrogate(base_attn, cfg.classifier_ckpt)
    base_attn.to(device).eval()

    # Line 3 (red dotted): zero_input + fine-tuned surrogate weights
    print(f"Loading zero_input surrogate from {cfg.zero_surrogate_ckpt} …")
    surr_zero = build_vit_surrogate(
        model_name=cfg.model_name,
        num_classes=num_classes,
        masking_strategy="zero_input",
    )
    _load_surrogate_ckpt(surr_zero, cfg.zero_surrogate_ckpt)
    surr_zero.to(device).eval()

    # Line 4 (red solid): zero_input + classifier weights (un-finetuned baseline)
    print(f"Loading zero_input baseline (classifier weights) …")
    base_zero = build_vit_surrogate(
        model_name=cfg.model_name,
        num_classes=num_classes,
        masking_strategy="zero_input",
    )
    _load_classifier_into_surrogate(base_zero, cfg.classifier_ckpt)
    base_zero.to(device).eval()

    # Infer num_patches from any of the surrogate models.
    num_patches: int = surr_attn.vit.patch_embed.num_patches
    print(f"  num_patches = {num_patches}")

    # ----------------------------------------------------------- evaluation --
    series = [
        (surr_attn, "Attn mask (finetuned)", _COLOR_ATTN, "--"),
        (base_attn, "Attn mask", _COLOR_ATTN, "-"),
        (surr_zero, "Zero input (finetuned)", _COLOR_ZERO, "--"),
        (base_zero, "Zero input", _COLOR_ZERO, "-"),
    ]

    mask_seed = cfg.seed

    print(
        f"Computing KL vs. cardinality "
        f"(step={cfg.step}, masks_per_cardinality={cfg.num_masks}, "
        f"mask_seed={mask_seed}) …"
    )

    # ── rcParams (reference style) ─────────────────────────────────────────
    plt.rcParams["font.family"] = "PT Sans"
    plt.rcParams["font.size"] = 18
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.edgecolor"] = "1.0"
    plt.rcParams["legend.framealpha"] = 0

    fig, ax = plt.subplots(figsize=(11, 6))

    legend_elements = []

    for model, label, color, ls in series:
        print(f"  Evaluating: {label} …")
        results = compute_kl_vs_cardinality(
            surrogate=model,
            classifier=classifier,
            images=images,
            num_patches=num_patches,
            num_masks_per_cardinality=cfg.num_masks,
            cardinality_step=cfg.step,
            device=device,
            seed=mask_seed,
        )
        cardinalities = sorted(results.keys())
        num_deleted = np.array(cardinalities)
        means = np.array([np.mean(results[m]) for m in cardinalities])
        stds = np.array([np.std(results[m], ddof=1) for m in cardinalities])
        ns = np.array([len(results[m]) for m in cardinalities])
        ci95 = 1.96 * stds / np.sqrt(ns)

        ax.plot(num_deleted, means, color=color, linestyle=ls, linewidth=3)
        ax.fill_between(
            num_deleted,
            means - ci95,
            means + ci95,
            alpha=0.2,
            color=color,
        )
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=5, linestyle=ls, label=label)
        )

    # ----------------------------------------------------------------- plot --
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)

    ax.xaxis.set_major_locator(MultipleLocator(28))
    ax.xaxis.set_minor_locator(MultipleLocator(14))
    ax.xaxis.grid(True, which="major", linewidth=0.4, alpha=0.6)
    ax.xaxis.grid(True, which="minor", linewidth=0.4, alpha=0.2)

    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.yaxis.grid(True, which="major", linewidth=0.4, alpha=0.6)
    ax.yaxis.grid(True, which="minor", linewidth=0.4, alpha=0.2)

    ax.set_xlabel("# of Deleted Patches", labelpad=10)
    ax.set_ylabel("KL divergence")
    ax.set_title(cfg.dataset.capitalize(), pad=10)
    ax.set_xlim(-2, num_patches + 2)

    fig.legend(
        handles=legend_elements,
        ncol=4,
        handletextpad=0.6,
        columnspacing=1,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
    ).set_zorder(100)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)

    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()
