# ViT-Shapley

Shapley values are a theoretically grounded model explanation approach, but their exponential computational cost makes them difficult to use with large deep learning models. **ViT-Shapley** makes Shapley values practical for vision transformer (ViT) models by learning an _amortized explainer model_ that generates explanations in a single forward pass.

Please see [our paper (arXiv:2206.05282)](https://arxiv.org/abs/2206.05282?context=cs.LG) for more details, as well as the work that ViT-Shapley builds on ([KernelSHAP](https://arxiv.org/abs/1705.07874), [FastSHAP](https://openreview.net/forum?id=Zq2G_VTV53T)).

## Installation

```bash
git clone https://github.com/chanwkimlab/vit-shapley.git && cd vit-shapley
conda create -n vit-shapley python=3.10 -y
conda activate vit-shapley
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .            # core deps: timm, pydantic, pyyaml, tqdm
pip install -e ".[dev]"     # optional: pytest, ruff, jupyterlab
pip install captum           # optional: gradient baselines
```

## Workflow

```text
Stage 1: Classifier          Obtain your initial ViT image classifier to explain
              |
Stage 2: Surrogate           If your model was not trained to acommodate held-out image patches, fine-tune it with random masking
              |
Stage 3: Explainer            Train an explainer that produces Shapley values in one forward pass
```

Each stage produces a checkpoint that feeds into the next. The final explainer generates per-patch importance scores (196 values for a 14x14 patch grid) without the exponential cost of exact Shapley computation.

```bash
# 1. Classifier
python scripts/train_classifier.py --config configs/classifier.yaml

# 2. Surrogates (one per masking strategy)
python scripts/train_surrogate.py --config configs/surrogate_attn.yaml
python scripts/train_surrogate.py --config configs/surrogate_zero.yaml

# 3. Explainer
python scripts/train_explainer.py --config configs/explainer.yaml

# Evaluate / visualise
python scripts/plot_surrogate_kl.py   --config configs/plot_surrogate_kl.yaml
python scripts/visualize_explainer.py --config configs/visualize_explainer.yaml
```

Switch dataset with `--set dataset=pet` on any command. All checkpoint paths and output filenames update automatically via `${dataset}` self-referencing in the YAML. See [Config System](docs/config.md) for details.

For binary classification datasets like MURA, add `--set target_type=binary` to all pipeline commands. This switches the loss function (BCEWithLogitsLoss), KL divergence computation (sigmoid-based), and link function (sigmoid instead of softmax).

## Datasets

| Name                   | Classes | Source                     |
| ---------------------- | ------- | -------------------------- |
| `imagenette` (default) | 10      | fast.ai subset of ImageNet |
| `mura`                 | 2       | Stanford MURA X-rays       |
| `pet`                  | 37      | Oxford-IIIT Pet            |

MURA (Musculoskeletal Radiographs) is a binary classification dataset of bone X-ray images (negative/positive) across 7 body parts. It requires a Stanford AIMI license — download from [Stanford AIMI](https://stanfordaimi.azurewebsites.net/datasets/3e00d84b-d86e-4fed-b2a4-bfe3effd661b) and place at `<data_root>/MURA-v1.1/`. Use with `--set dataset=mura`.

## Testing

```bash
python -m pytest tests/ -v
```

## Documentation

- [Config](docs/config.md) — YAML variables, `--set` overrides, `.env`, multi-GPU
- [Evaluation](docs/evaluation.md) — KL divergence plots and Shapley heatmaps
    <!-- - [Baselines](docs/baselines.md) — attention, perturbation, and gradient explanation methods -->
