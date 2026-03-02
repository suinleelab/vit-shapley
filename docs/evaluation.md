# Evaluation

## KL Divergence vs. Mask Cardinality (Figure 2)

Plots mean KL(classifier ∥ surrogate) as a function of how many patches are visible, with 95% CI bands. Compares both surrogate masking strategies against a random-masking baseline. Requires Stage 1 + both Stage 2 checkpoints.

Output: `figures/surrogate_kl_${dataset}.png`

## Shapley Heatmap Visualisation

For each selected image, renders the original alongside per-class heatmap overlays. Red = increases predicted probability; blue = suppresses it. Requires Stage 2 + Stage 3 checkpoints.

By default, heatmap columns correspond to ground-truth classes of the selected samples. Set `class_indices` to show specific classes instead.

Output: `figures/shapley_heatmaps_${dataset}.png`

The reusable functions (`compute_shapley_values`, `denormalize_imagenet`, `shapley_to_heatmap`, `plot_shapley_heatmaps`) live in `src/vit_shapley/visualization/` for use in notebooks.
