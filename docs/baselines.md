# Baselines

`src/vit_shapley/baselines/` implements explanation baselines for comparison. All return numpy arrays.

| Module | Methods |
|--------|---------|
| `attention` | `extract_attention_maps`, `attentions_to_explanation` (`"rollout"` / `"raw"` / layer index), `compute_joint_attention` |
| `perturbation` | `leave_one_out`, `rise` |
| `gradient` | `get_vanilla_gradient`, `get_smoothgrad`, `get_vargrad`, `get_integrated_gradients` |
| `utils` | `generate_mask`, `get_random_explanation`, `get_relative_value`, `explanation_to_mask` |

## Quick usage

```python
from vit_shapley.baselines.attention import extract_attention_maps, attentions_to_explanation
from vit_shapley.baselines.perturbation import leave_one_out, rise
from vit_shapley.baselines.gradient import get_vanilla_gradient, get_integrated_gradients

attn_maps = extract_attention_maps(vit, image.unsqueeze(0))      # (1, L, H, N, N)
rollout = attentions_to_explanation(attn_maps, mode="rollout")    # (1, P)

loo = leave_one_out(image, surrogate)                             # (C, P)
rise_scores = rise(image, surrogate, N=2000, batch_size=100)      # (C, P)

grad = get_vanilla_gradient(image, vit, output_dim=10, space="embedding")  # (10, P)
ig = get_integrated_gradients(image, vit, output_dim=10, space="embedding") # (10, P)
```

Gradient methods require `captum` (`pip install captum`). Tests are auto-skipped if not installed.
