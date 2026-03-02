"""Pydantic config for visualize_explainer."""

from typing import Optional

from pydantic import BaseModel


class VisualizeConfig(BaseModel):
    # Required — no defaults. Must be provided in YAML or via --set.
    surrogate_ckpt: str
    explainer_ckpt: str
    model_name: str = "vit_base_patch16_224"
    data_root: str = "/local-b/chanwkim/vit-shapley-data"
    split: str = "val"
    sample_indices: list[int] = [0, 1, 2, 3]
    class_indices: Optional[list[int]] = None
    masking_strategy: str = "attn_mask"
    output: str = "figures/shapley_heatmaps.png"
    image_size: int = 224
    device: str = ""
