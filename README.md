# ViT-Shapley

Shapley values are a theoretically grounded model explanation approach, but their exponential computational cost makes them difficult to use with large deep learning models. **ViT-Shapley** makes Shapley values practical for vision transformer (ViT) models by learning an _amortized explainer model_ that generates explanations in a single forward pass.

Please see [our paper (arXiv:2206.05282)](https://arxiv.org/abs/2206.05282?context=cs.LG) for more details, as well as the work that ViT-Shapley builds on ([KernelSHAP](https://arxiv.org/abs/1705.07874), [FastSHAP](https://openreview.net/forum?id=Zq2G_VTV53T)).

## Overview

```text
Stage 1: Classifier          Obtain your initial ViT image classifier
              |
Stage 2: Surrogate           Train a surrogate that mimics the classifier on masked inputs
              |
Stage 3: Explainer            Train an explainer that produces Shapley values in one forward pass
```

Each stage produces a checkpoint that feeds into the next. The final explainer generates per-patch importance scores (196 values for a 14x14 patch grid) without the exponential cost of exact Shapley computation.

## Installation

```bash
git clone https://github.com/chanwkimlab/vit-shapley.git
cd vit-shapley

# Create and activate conda environment (Python 3.10 required)
conda create -n vit-shapley python=3.10 -y
conda activate vit-shapley

# Install PyTorch with the CUDA variant that matches your driver.
# Example for CUDA 12.x (driver >= 525):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install the package and remaining dependencies (timm, numpy, tqdm, pydantic, pyyaml)
pip install -e .

# (Optional) Install development tools (jupyterlab, pytest, ruff, etc.)
pip install -e ".[dev]"
```

## Quick Start

```bash
# Stage 1: Train classifier
python scripts/train_classifier.py --config configs/classifier.yaml

# Stage 2: Train surrogates (one per masking strategy)
# The surrogate model supports two masking strategies: `attn_mask` (attention masking, default) and `zero_input` (zero out masked patch embeddings). Set `masking_strategy` in the surrogate, explainer, and visualize configs to control which strategy is used:
python scripts/train_surrogate.py --config configs/surrogate_attn.yaml
python scripts/train_surrogate.py --config configs/surrogate_zero.yaml

# Stage 3: Train explainer
python scripts/train_explainer.py --config configs/explainer.yaml

# Evaluate: KL divergence plot
python scripts/plot_surrogate_kl.py --config configs/plot_surrogate_kl.yaml

# Visualise: Shapley heatmaps
python scripts/visualize_explainer.py --config configs/visualize_explainer.yaml
```

Config YAML files use `$data_dir` and `$checkpoint_dir` variables (shell-style syntax). These are resolved automatically from a `.env` file in the project root.

### Masking strategy

The surrogate model supports two masking strategies: `attn_mask` (attention masking, default) and `zero_input` (zero out masked patch embeddings). Set `masking_strategy` in the surrogate, explainer, and visualize configs to control which strategy is used:

```bash
# Train surrogate with zero-input masking
python scripts/train_surrogate.py --config configs/surrogate_zero.yaml

# Train explainer with matching strategy
python scripts/train_explainer.py --config configs/explainer.yaml \
    --set masking_strategy=zero_input surrogate_ckpt=checkpoints/surrogate_zero/best_surrogate.pth
```

## Testing

```bash
python -m pytest tests/ -v
```

## Documentation

- [Config System](docs/config.md) — YAML configs, `--set` overrides, multi-GPU device placement
- [Training Pipeline](docs/training.md) — detailed Stage 1/2/3 instructions and architecture notes
- [Evaluation](docs/evaluation.md) — KL divergence plots and Shapley heatmap visualisation
- [Baselines](docs/baselines.md) — attention, perturbation, and gradient explanation baselines
