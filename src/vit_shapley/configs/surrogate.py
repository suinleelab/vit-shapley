"""Pydantic config for Stage 2: train_surrogate."""

from pydantic import BaseModel, ConfigDict


class SurrogateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Required — no default. Must be provided in YAML or via --set.
    classifier_ckpt: str
    target_type: str = "multiclass"
    dataset: str = "imagenette"
    data_root: str = "/local-b/chanwkim/vit-shapley-data"
    model_name: str = "vit_base_patch16_224"
    masking_strategy: str = "attn_mask"
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-5
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    num_workers: int = 4
    image_size: int = 224
    save_dir: str = "checkpoints/surrogate"
    use_amp: bool = True
    device: str = ""
    classifier_device: str = ""
