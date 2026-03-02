# Baselines

`src/vit_shapley/baselines/` implements ~10 explanation baseline methods for
comparing against ViT-Shapley Shapley values. All public functions return
numpy arrays of shape `(num_classes, P)` or `(B, P)` depending on the method.

## Submodules

| Module         | Contents                                                                                                |
| -------------- | ------------------------------------------------------------------------------------------------------- |
| `utils`        | `generate_mask`, `get_random_explanation`, `get_relative_value`, `explanation_to_mask`                  |
| `attention`    | `compute_joint_attention`, `attentions_to_explanation`, `extract_attention_maps`                        |
| `perturbation` | `leave_one_out`, `rise`                                                                                 |
| `gradient`     | `get_vanilla_gradient`, `get_smoothgrad`, `get_vargrad`, `get_integrated_gradients` (requires `captum`) |

## Quick usage

```python
import torch
import timm
from vit_shapley.models.surrogate import build_vit_surrogate
from vit_shapley.baselines.attention import extract_attention_maps, attentions_to_explanation
from vit_shapley.baselines.perturbation import leave_one_out, rise
from vit_shapley.baselines.gradient import get_vanilla_gradient, get_integrated_gradients

# Load models
vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
surrogate = build_vit_surrogate("vit_tiny_patch16_224", num_classes=10,
                                 classifier_ckpt_path="checkpoints/classifier/best_classifier.pth")
image = torch.randn(3, 224, 224)

# Attention rollout: (num_classes, P) — shape is actually (B, P) from rollout
attn_maps = extract_attention_maps(vit, image.unsqueeze(0))   # (1, L, H, N, N)
rollout = attentions_to_explanation(attn_maps, mode="rollout") # (1, P)

# Leave-one-out
loo = leave_one_out(image, surrogate)   # (num_classes, P)

# RISE
rise_scores = rise(image, surrogate, N=2000, batch_size=100)  # (num_classes, P)

# Gradient methods (requires: pip install captum)
grad = get_vanilla_gradient(image, vit, output_dim=10, space="embedding")  # (10, P)
ig   = get_integrated_gradients(image, vit, output_dim=10, space="embedding", n_steps=50)
```

## Attention modes

`attentions_to_explanation` accepts three `mode` values:

| Mode        | Description                                                             |
| ----------- | ----------------------------------------------------------------------- |
| `"rollout"` | Attention rollout (Abnar & Zuidema 2020): matrix-multiply across layers |
| `"raw"`     | Last-layer CLS→patch attention (after residual + re-norm)               |
| `int`       | Layer at index `mode` (0-indexed)                                       |

## Mask utilities

```python
from vit_shapley.baselines.utils import (
    generate_mask, get_random_explanation, get_relative_value, explanation_to_mask
)

# Generate 100 Shapley-distributed masks over 196 patches
masks = generate_mask(196, num_mask_samples=100, mode="shapley")  # (100, 196)

# Convert attribution scores to insertion/deletion curves
explanation = loo[0]  # (P,) — attributions for class 0
from vit_shapley.baselines.utils import explanation_to_mask
seq = explanation_to_mask(explanation[None], mode="insertion")  # (1, P+1, P)
```

## Optional dependency

Gradient methods require `captum`:

```bash
pip install captum
```

If `captum` is not installed, the gradient module raises a clear `ImportError`
when called. The test file `tests/baselines/test_gradient.py` is automatically
skipped via `pytest.importorskip("captum")`.
