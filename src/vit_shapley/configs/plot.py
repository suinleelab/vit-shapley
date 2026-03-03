"""Pydantic config for plot_surrogate_kl."""

from pydantic import BaseModel, ConfigDict


class PlotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Required — no defaults. Must be provided in YAML or via --set.
    classifier_ckpt: str
    attn_surrogate_ckpt: str
    zero_surrogate_ckpt: str
    target_type: str = "multiclass"
    dataset: str = "imagenette"
    model_name: str = "vit_base_patch16_224"
    data_root: str = "/local-b/chanwkim/vit-shapley-data"
    num_images: int = 50
    num_masks: int = 50
    step: int = 10
    seed: int = 42
    output: str = "figures/surrogate_kl.png"
    device: str = ""
