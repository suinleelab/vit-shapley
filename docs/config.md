# Config System

All scripts use `--config <yaml>` with optional `--set KEY=VALUE` overrides.

## Variable resolution

YAML values can contain `$var` or `${var}` placeholders. Resolution order:

1. **Config self-references** — a value can reference another key in the same file (e.g. `${dataset}`)
2. **`.env` file** — loaded automatically from the project root (or via `--env path`)
3. **Shell environment variables**

`--set` overrides are applied **before** resolution, so `--set dataset=pet` propagates into `${dataset}` everywhere:

```yaml
# configs/classifier.yaml
dataset: imagenette
save_dir: $checkpoint_dir/classifier_${dataset}   # → checkpoints/classifier_imagenette
```

```bash
python scripts/train_classifier.py --config configs/classifier.yaml --set dataset=pet
# save_dir resolves to: checkpoints/classifier_pet
```

Transitive chains (A → B → C) are resolved automatically via multi-pass evaluation.

## `.env` file

For machine-specific paths, create a `.env` in the project root:

```bash
data_dir=/sdata/chanwkim/vit-shapley-data
checkpoint_dir=checkpoints
```

## Multi-GPU device placement

When two models don't fit on one GPU, place the frozen model on a separate device:

```bash
# Surrogate training: classifier on a different GPU
python scripts/train_surrogate.py --config configs/surrogate_attn.yaml \
    --set device=cuda:0 classifier_device=cuda:1

# Explainer training: surrogate on a different GPU
python scripts/train_explainer.py --config configs/explainer.yaml \
    --set device=cuda:0 surrogate_device=cuda:1
```

When `classifier_device` / `surrogate_device` is omitted, the frozen model shares the training device.
