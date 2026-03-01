# ViT-Shapley

Shapley values are a theoretically grounded model explanation approach, but their exponential computational cost makes them difficult to use with large deep learning models. **ViT-Shapley** makes Shapley values practical for vision transformer (ViT) models by learning an _amortized explainer model_ that generates explanations in a single forward pass.

Please see [our paper (arXiv:2206.05282)](https://arxiv.org/abs/2206.05282?context=cs.LG) for more details, as well as the work that ViT-Shapley builds on ([KernelSHAP](https://arxiv.org/abs/1705.07874), [FastSHAP](https://openreview.net/forum?id=Zq2G_VTV53T)).

## Overview

ViT-Shapley follows a multi-stage training pipeline:

```text
Stage 1: Classifier          Train a ViT image classifier
              |
Stage 2: Surrogate           Train a surrogate that mimics the classifier on masked inputs
              |
Stage 3: Explainer            Train an explainer that produces Shapley values in one forward pass
              |
Stage 4: Classifier Masked   (Optional) Fine-tune classifier to handle masked patches
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

# Install the package and remaining dependencies (timm, numpy, tqdm)
pip install -e .

# (Optional) Install development tools (jupyterlab, pytest, ruff, etc.)
pip install -e ".[dev]"
```

## Testing

Run the full test suite:

```bash
python -m pytest tests/ -v
```

Run a specific test file:

```bash
python -m pytest tests/test_config.py -v
python -m pytest tests/modules/test_explainer.py -v
```

## Training Pipeline

### Stage 1: Train the Classifier

Train a ViT image classifier on ImageNette. The dataset is downloaded automatically
on first run into `--data-root` (default: `/local-b/chanwkim/vit-shapley-data`).

```bash
# Quick test with the smallest ViT
python scripts/train_classifier.py \
    --model-name vit_tiny_patch16_224 \
    --epochs 10 \
    --batch-size 64 \
    --save-dir checkpoints/classifier

# Full run with ViT-Base
python scripts/train_classifier.py \
    --model-name vit_base_patch16_224 \
    --pretrained \
    --epochs 10 \
    --batch-size 64 \
    --lr 1e-3 \
    --weight-decay 1e-2 \
    --num-workers 4 \
    --image-size 224 \
    --save-dir checkpoints/classifier
```

The best checkpoint (by validation accuracy) is saved to
`<save-dir>/best_classifier.pth`. Run `python scripts/train_classifier.py --help`
for all options.

### Stage 2: Train the Surrogate

Fine-tune a surrogate model that learns to mimic the classifier on randomly
masked subsets of patches. Requires the Stage 1 classifier checkpoint.

The surrogate is initialised from the classifier checkpoint and trained by
minimising `DKL(f(x) || g(x_s))` over random patch subsets `s` (Eq. 2 of the
paper). timm's native `attn_mask` support is used — no custom attention modules
are required.

Train **two** surrogates — one per masking strategy. They are independent and
can be run in parallel (e.g. in separate tmux panes or on separate GPUs).

```bash
# attn_mask surrogate (attention-bias masking)
python scripts/train_surrogate.py \
    --model-name vit_tiny_patch16_224 \
    --classifier-ckpt checkpoints/classifier/best_classifier.pth \
    --masking-strategy attn_mask \
    --epochs 50 \
    --batch-size 64 \
    --save-dir checkpoints/surrogate_attn

# zero_input surrogate (zero-pixel masking)
python scripts/train_surrogate.py \
    --model-name vit_tiny_patch16_224 \
    --classifier-ckpt checkpoints/classifier/best_classifier.pth \
    --masking-strategy zero_input \
    --epochs 50 \
    --batch-size 64 \
    --save-dir checkpoints/surrogate_zero
```

Full run with ViT-Base (paper defaults: `lr=1e-5`, 50 epochs):

```bash
python scripts/train_surrogate.py \
    --model-name vit_base_patch16_224 \
    --classifier-ckpt checkpoints/classifier/best_classifier.pth \
    --masking-strategy attn_mask \
    --epochs 50 --batch-size 64 --lr 1e-5 --weight-decay 1e-2 --num-workers 4 \
    --save-dir checkpoints/surrogate_attn

python scripts/train_surrogate.py \
    --model-name vit_base_patch16_224 \
    --classifier-ckpt checkpoints/classifier/best_classifier.pth \
    --masking-strategy zero_input \
    --epochs 50 --batch-size 64 --lr 1e-5 --weight-decay 1e-2 --num-workers 4 \
    --save-dir checkpoints/surrogate_zero
```

The best checkpoint (by minimum validation KL divergence) is saved to
`<save-dir>/best_surrogate.pth`.

### Stage 3: Train the Explainer

Train an explainer that produces per-patch Shapley value estimates in a single
forward pass, without the exponential cost of exact Shapley computation.

The explainer is initialised from a Stage 2 surrogate checkpoint and trained
by minimising the weighted least-squares (WLS) objective (Eq. 3 of the
ViT-Shapley paper):

    L(φ) = E_{x,S} [ w(|S|) · ‖v(S;x) − v(∅;x) − Σ_{i∈S} φ_i(x)‖² ]

where `v(S;x)` is the frozen surrogate's softmax output on masked subset `S`,
`w(|S|)` is the Shapley kernel weight, and `φ_i(x)` are the per-patch
Shapley value predictions.

```bash
# Quick test with the smallest ViT
python scripts/train_explainer.py \
    --surrogate-ckpt checkpoints/surrogate_attn/best_surrogate.pth \
    --model-name vit_tiny_patch16_224 \
    --epochs 50 --batch-size 64 \
    --save-dir checkpoints/explainer

# Full run with ViT-Base (paper defaults: lr=1e-5, 50 epochs)
python scripts/train_explainer.py \
    --surrogate-ckpt checkpoints/surrogate_attn/best_surrogate.pth \
    --model-name vit_base_patch16_224 \
    --epochs 50 --batch-size 64 --lr 1e-5 --weight-decay 1e-2 --num-workers 4 \
    --save-dir checkpoints/explainer
```

The best checkpoint (by minimum validation WLS loss) is saved to
`<save-dir>/best_explainer.pth`.
Run `python scripts/train_explainer.py --help` for all options.

## Evaluation

### Figure 2: KL Divergence vs. Mask Cardinality

Reproduces Figure 2 of the ViT-Shapley paper. Plots four lines:

| Line | Strategy | Weights |
|---|---|---|
| Blue solid | `attn_mask` | fine-tuned surrogate |
| Blue dotted | `attn_mask` | original classifier (un-finetuned baseline) |
| Red solid | `zero_input` | fine-tuned surrogate |
| Red dotted | `zero_input` | original classifier (un-finetuned baseline) |

Each line shows mean KL(classifier(full_image) ∥ surrogate(masked_image)) with
a shaded 95% confidence interval. Requires Stage 1 and both Stage 2 checkpoints.

```bash
# Install matplotlib (if not already installed via [dev])
pip install -e ".[dev]"

# Generate figure
python scripts/plot_surrogate_kl.py \
    --classifier-ckpt     checkpoints/classifier/best_classifier.pth \
    --attn-surrogate-ckpt checkpoints/surrogate_attn/best_surrogate.pth \
    --zero-surrogate-ckpt checkpoints/surrogate_zero/best_surrogate.pth \
    --model-name vit_tiny_patch16_224 \
    --num-images 50 \
    --num-masks 50 \
    --step 10 \
    --output figures/surrogate_kl.png
```

The figure is saved to `--output` (default: `figures/surrogate_kl.png`).
Run `python scripts/plot_surrogate_kl.py --help` for all options.
