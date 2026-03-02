# Evaluation

## Figure 2: KL Divergence vs. Mask Cardinality

Reproduces Figure 2 of the ViT-Shapley paper. Plots four lines:

Each line shows mean KL(classifier(full_image) ∥ surrogate(masked_image)) with
a shaded 95% confidence interval. Requires Stage 1 and both Stage 2 checkpoints.

Edit `configs/plot_surrogate_kl.yaml` to set the three required checkpoint paths,
then run:

```bash
# Generate figure
python scripts/plot_surrogate_kl.py --config configs/plot_surrogate_kl.yaml

# Quick override
python scripts/plot_surrogate_kl.py --config configs/plot_surrogate_kl.yaml \
    --set model_name=vit_tiny_patch16_224 num_images=50 num_masks=50 step=10
```

The figure is saved to `output` (default: `figures/surrogate_kl.png`).

## Shapley Heatmap Visualisation

Visualise per-patch Shapley values produced by the trained explainer. For
each selected image the script renders the original image alongside heatmap
overlays (one column per class). Red patches increase the predicted probability
for that class; blue patches suppress it.

Requires Stage 2 (surrogate) and Stage 3 (explainer) checkpoints.

Edit `configs/visualize_explainer.yaml` to set the surrogate and explainer
checkpoint paths, then run:

```bash
python scripts/visualize_explainer.py --config configs/visualize_explainer.yaml

# Quick override: select different samples and a specific model
python scripts/visualize_explainer.py --config configs/visualize_explainer.yaml \
    --set model_name=vit_tiny_patch16_224 output=figures/shapley_heatmaps.png
```

By default the heatmap columns correspond to the unique ground-truth classes of
the selected samples (in first-appearance order). Set `class_indices` in the
YAML (or via `--set`) to show specific classes instead.

The reusable functions (`compute_shapley_values`, `denormalize_imagenet`,
`shapley_to_heatmap`, `plot_shapley_heatmaps`) live in
`src/vit_shapley/visualization/` and can be imported directly for use in
notebooks or custom scripts.
