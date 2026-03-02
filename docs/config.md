# Config System

Config YAML files use `$data_dir` and `$checkpoint_dir` variables (shell-style syntax). These are resolved automatically from a `.env` file in the project root:

```bash
# .env
data_dir=/sdata/chanwkim/vit-shapley-data
checkpoint_dir=checkpoints
```

Scripts load `.env` by default — no extra flags needed. You can also point to a different file with `--env path/to/other.env`, or use shell environment variables instead (`.env` values take priority over shell env vars).

Override any config value with `--set KEY=VALUE` (e.g. `--set epochs=10 lr=1e-4`). `--set` overrides apply last and take final precedence.

All scripts are driven by YAML config files. Default configs are in `configs/`.
Override any value with `--set KEY=VALUE`:

```bash
# Use default config
python scripts/train_classifier.py --config configs/classifier.yaml

# Override individual fields
python scripts/train_classifier.py --config configs/classifier.yaml \
    --set model_name=vit_tiny_patch16_224 epochs=10 lr=1e-4
```

Edit the relevant YAML file to set your `data_root` and checkpoint paths before
running.

## Multi-GPU Device Placement

By default all models run on a single device (auto-detected or set via `device`).
When two models don't fit on one GPU, you can place the frozen model on a
separate device:

```bash
# Surrogate training: classifier on a different GPU
python scripts/train_surrogate.py --config configs/surrogate.yaml \
    --set device=cuda:0 classifier_device=cuda:1

# Explainer training: surrogate on a different GPU
python scripts/train_explainer.py --config configs/explainer.yaml \
    --set device=cuda:0 surrogate_device=cuda:1
```

When `classifier_device` / `surrogate_device` is left empty (default), the
frozen model is placed on the same device as the model being trained.
