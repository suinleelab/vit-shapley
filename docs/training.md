# Training Pipeline

## Stage 1: Train the Classifier

Train a ViT image classifier on ImageNette. The dataset is downloaded automatically
on first run into `data_root` (default: `/local-b/chanwkim/vit-shapley-data`).

Edit `configs/classifier.yaml` to adjust hyperparameters, then run:

```bash
python scripts/train_classifier.py --config configs/classifier.yaml

# Quick test with the smallest ViT
python scripts/train_classifier.py --config configs/classifier.yaml \
    --set model_name=vit_tiny_patch16_224 epochs=10 batch_size=64
```

The best checkpoint (by validation accuracy) is saved to
`<save_dir>/best_classifier.pth`.

## Stage 2: Train the Surrogate

Fine-tune a surrogate model that learns to mimic the classifier on randomly
masked subsets of patches. Requires the Stage 1 classifier checkpoint.

The surrogate is initialised from the classifier checkpoint and trained by
minimising `DKL(f(x) || g(x_s))` over random patch subsets `s` (Eq. 2 of the
paper). timm's native `attn_mask` support is used — no custom attention modules
are required.

Train **two** surrogates independently — one per masking strategy.

```bash
# attn_mask surrogate (attention-bias masking)
python scripts/train_surrogate.py --config configs/surrogate.yaml \
    --set masking_strategy=attn_mask save_dir=checkpoints/surrogate_attn

# zero_input surrogate (zero-pixel masking)
python scripts/train_surrogate.py --config configs/surrogate.yaml \
    --set masking_strategy=zero_input save_dir=checkpoints/surrogate_zero
```

The best checkpoint (by minimum validation KL divergence) is saved to
`<save_dir>/best_surrogate.pth`.

## Stage 3: Train the Explainer

Train an explainer that produces per-patch Shapley value estimates in a single
forward pass, without the exponential cost of exact Shapley computation.

**Architecture** (paper default):

- ViT backbone initialised from the Stage 2 surrogate checkpoint
- 1 extra ViT attention block (CLS + patch tokens)
- LayerNorm + 3-layer MLP head: `D → 4D → 4D → C` with GELU activations
- Tanh output activation
- **Additive normalisation** to hard-enforce the efficiency axiom:
  `φ'_i = φ_i + (v(grand) − v(null) − Σ_j φ_j) / n`

Edit `configs/explainer.yaml` to set `surrogate_ckpt`, then run:

```bash
python scripts/train_explainer.py --config configs/explainer.yaml

# Quick override for a small model
python scripts/train_explainer.py --config configs/explainer.yaml \
    --set model_name=vit_tiny_patch16_224 epochs=100 batch_size=32
```

The best checkpoint (by minimum validation MSE loss) is saved to
`<save_dir>/best_explainer.pth`.
