"""Pydantic config for Stage 1: train_classifier."""

from pydantic import BaseModel


class ClassifierConfig(BaseModel):
    data_root: str = "/local-b/chanwkim/vit-shapley-data"
    model_name: str = "vit_base_patch16_224"
    pretrained: bool = True
    epochs: int = 25
    batch_size: int = 64
    lr: float = 1e-5
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    num_workers: int = 4
    image_size: int = 224
    save_dir: str = "checkpoints/classifier"
    use_amp: bool = True
    device: str = ""
