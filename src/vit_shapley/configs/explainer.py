"""Pydantic config for Stage 3: train_explainer."""

from pydantic import BaseModel


class ExplainerConfig(BaseModel):
    # Required — no default. Must be provided in YAML or via --set.
    surrogate_ckpt: str
    data_root: str = "/local-b/chanwkim/vit-shapley-data"
    model_name: str = "vit_base_patch16_224"
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    num_mask_samples: int = 32
    paired_masks: bool = True
    num_workers: int = 4
    image_size: int = 224
    save_dir: str = "checkpoints/explainer"
    use_amp: bool = True
    device: str = ""
    surrogate_device: str = ""
