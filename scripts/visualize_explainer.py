#!/usr/bin/env python
"""CLI: visualize Shapley value heatmaps from the trained ViT explainer.

For each selected image, the figure shows:
  - The original (denormalized) image.
  - One Shapley heatmap column per class (default: unique GT classes of the
    selected samples in first-appearance order).

Red = positive Shapley (patch increases class probability).
Blue = negative Shapley (patch suppresses class probability).

Example
-------
    python scripts/visualize_explainer.py --config configs/visualize_explainer.yaml

    # Quick override:
    python scripts/visualize_explainer.py --config configs/visualize_explainer.yaml \\
        --set model_name=vit_tiny_patch16_224 sample_indices=[0,5,10,15]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless server — must be before pyplot import

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vit_shapley.configs import VisualizeConfig, load_config
from vit_shapley.data import get_dataset, resolve_num_classes
from vit_shapley.models import build_vit_explainer, build_vit_surrogate
from vit_shapley.visualization import compute_shapley_values, plot_shapley_heatmaps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize Shapley value heatmaps from the ViT explainer.",
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
        help="Override config values, e.g. --set output=figures/my_heatmap.png",
    )
    args = parser.parse_args()
    cfg = load_config(VisualizeConfig, args.config, args.set, env_path=args.env)

    device = (
        torch.device(cfg.device)
        if cfg.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ data --
    print("Loading dataset …")
    dataset = get_dataset(
        cfg.dataset,
        root=cfg.data_root,
        split=cfg.split,
        image_size=cfg.image_size,
        download=False,
    )
    num_classes = resolve_num_classes(dataset, cfg.target_type)

    images_list, labels_list = [], []
    for idx in cfg.sample_indices:
        img, label = dataset[
            idx % len(dataset)
        ]  # wrap around if idx exceeds dataset size
        images_list.append(img.numpy())  # CHW numpy
        labels_list.append(label)

    # Determine heatmap columns
    if cfg.class_indices is not None:
        class_cols = list(cfg.class_indices)
    else:
        seen, class_cols = set(), []
        for lbl in labels_list:
            if lbl not in seen:
                seen.add(lbl)
                class_cols.append(lbl)

    print(f"Sample indices : {cfg.sample_indices}")
    class_names = dataset.classes
    print(f"GT labels      : {[class_names[l] for l in labels_list]}")
    print(f"Heatmap classes: {[class_names[c] for c in class_cols]}")

    # --------------------------------------------------------------- models --
    print(f"Loading surrogate  from {cfg.surrogate_ckpt} …")
    surrogate = build_vit_surrogate(
        model_name=cfg.model_name,
        num_classes=num_classes,
        masking_strategy=cfg.masking_strategy,
    )
    ckpt = torch.load(cfg.surrogate_ckpt, map_location="cpu", weights_only=True)
    surrogate.load_state_dict(ckpt.get("model_state_dict", ckpt))
    surrogate.to(device).eval()

    print(f"Loading explainer  from {cfg.explainer_ckpt} …")
    explainer = build_vit_explainer(
        model_name=cfg.model_name,
        num_classes=num_classes,
    )
    ckpt = torch.load(cfg.explainer_ckpt, map_location="cpu", weights_only=True)
    explainer.load_state_dict(ckpt.get("model_state_dict", ckpt))
    explainer.to(device).eval()

    # -------------------------------------------------------- Shapley values --
    print("Computing Shapley values …")
    images_tensor = torch.stack([torch.from_numpy(img) for img in images_list])
    phi, grand_probs = compute_shapley_values(
        explainer,
        surrogate,
        images_tensor,
        device,
        target_type=cfg.target_type,
    )

    # ----------------------------------------------------------------- plot --
    print("Rendering figure …")
    fig = plot_shapley_heatmaps(
        images_list=images_list,
        labels_list=labels_list,
        phi=phi,
        class_cols=class_cols,
        class_names=class_names,
        image_size=cfg.image_size,
        grand_probs=grand_probs,
    )

    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()
