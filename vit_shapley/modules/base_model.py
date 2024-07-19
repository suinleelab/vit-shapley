import logging
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models as cnn_models
from transformers import get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup


class BaseModel(pl.LightningModule):
    """
    Args:
        backbone_type: should be the class name defined in `timm.models.vision_transformer`
        download_weight: whether to initialize backbone with the pretrained weights
        load_path: If not None. loads the weights saved in the checkpoint to the model
        target_type: `binary` or `multi-class` or `multi-label`
        output_dim: the dimension of output
        checkpoint_metric: the metric used to determine whether to save the current status as checkpoints during the validation phase
        optim_type: type of optimizer for optimizing parameters
        learning_rate: learning rate of optimizer
        weight_decay: weight decay of optimizer
        decay_power: only `cosine` annealing scheduler is supported currently
        warmup_steps: parameter for the `cosine` annealing scheduler
    """

    def __init__(self,
                 backbone_type: str = "vit_base_patch16_224",
                 download_weight: bool = False,
                 load_path: Union[str, None] = None,
                 target_type: str = "multiclass",
                 output_dim: int = 10,
                 checkpoint_metric: str = "accuracy",
                 optim_type: str = "Adamw",
                 learning_rate: float = 1e-5,
                 learning_rate_min: float = 0.0,
                 layer_decay_rate: float = 1.0,
                 weight_decay: float = 1e-5,
                 decay_power: str = "cosine",
                 warmup_steps: int = 500):
        super().__init__()

        self.save_hyperparameters()

        self.logger_ = logging.getLogger(self.__class__.__name__)

        assert not (self.hparams.download_weight and self.hparams.load_path is not None), \
            "'download_weight' and 'load_path' cannot be activated at the same time as the downloaded weight will be overwritten by weights in 'load_path'."

        # Backbone initialization. (currently support only vit)
        if self.__class__.__name__ == "Classifier":
            import vit_shapley.modules.vision_transformer_verbose as vit
        else:
            import vit_shapley.modules.vision_transformer as vit

        if hasattr(vit, self.hparams.backbone_type):
            self.backbone = getattr(vit, self.hparams.backbone_type)(pretrained=self.hparams.download_weight)
        elif hasattr(cnn_models, self.hparams.backbone_type):
            self.backbone = getattr(cnn_models, self.hparams.backbone_type)(pretrained=self.hparams.download_weight)
        else:
            raise NotImplementedError("Not supported backbone type")
        if self.hparams.download_weight:
            self.logger_.info("The backbone parameters were initialized with the downloaded pretrained weights.")

        # Nullify classification head built in the backbone module and rebuild.
        if self.backbone.__class__.__name__ == 'VisionTransformer':
            self.head_in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        elif self.backbone.__class__.__name__ == 'ResNet':
            self.head_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif self.backbone.__class__.__name__ == 'DenseNet':
            self.head_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise NotImplementedError("Not supported backbone type")

    def load_checkpoint(self):
        # Load checkpoints
        if self.hparams.load_path is not None:
            checkpoint = torch.load(self.hparams.load_path, map_location="cpu")
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint          
            ret = self.load_state_dict(state_dict, strict=False)
            self.logger_.info(f"Model parameters were updated from a checkpoint file {self.hparams.load_path}")
            self.logger_.info(f"Unmatched parameters - missing_keys:    {ret.missing_keys}")
            self.logger_.info(f"Unmatched parameters - unexpected_keys: {ret.unexpected_keys}")
        elif not self.hparams.download_weight:
            self.logger_.info("The backbone parameters were randomly initialized.")

    def configure_optimizers(self):
        if self.hparams.optim_type == "Adamw":
            optimizer = optim.AdamW(self.parameters(),
                                    lr=self.hparams.learning_rate,
                                    weight_decay=self.hparams.weight_decay)
        elif self.hparams.optim_type == "Adam":
            optimizer = optim.Adam(self.parameters(),
                                lr=self.hparams.learning_rate,
                                weight_decay=self.hparams.weight_decay)
        elif self.hparams.optim_type == "SGD":
            optimizer = optim.SGD(self.parameters(),
                                lr=self.hparams.learning_rate,
                                momentum=0.9,
                                weight_decay=self.hparams.weight_decay)
        else:
            optimizer = None

        # setup scheduler
        if self.trainer.max_steps is None or self.trainer.max_steps == -1:
            max_steps = (
                    len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs // self.trainer.accumulate_grad_batches)
        else:
            max_steps = self.trainer.max_steps

        if self.hparams.decay_power == "cosine":
            scheduler = {"scheduler": get_cosine_schedule_with_warmup(optimizer,
                                                                    num_warmup_steps=self.hparams.warmup_steps,
                                                                    num_training_steps=max_steps),
                        "interval": "step"}
            return ([optimizer], [scheduler])
        elif self.hparams.decay_power == "polynomial":
            scheduler = {"scheduler": get_polynomial_decay_schedule_with_warmup(optimizer,
                                                                                num_warmup_steps=self.hparams.warmup_steps,
                                                                                num_training_steps=max_steps,
                                                                                lr_end=self.hparams.learning_rate_min,
                                                                                power=0.9),
                        "interval": "step"}
            return ([optimizer], [scheduler])
        elif self.hparams.decay_power is None:
            return optimizer
        else:
            NotImplementedError("Unsupported scheduler!")
