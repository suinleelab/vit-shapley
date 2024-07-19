import torch
import torch.nn as nn
from torchvision import models as cnn_models

from vit_shapley.modules.base_model import BaseModel
import vit_shapley.modules.vision_transformer as vit
from vit_shapley.modules import classifier_utils


class ClassifierMasked(BaseModel):
    """
    `pytorch_lightning` module for surrogate

    Args:
        mask_location: how the mask is applied to the input. ("pre-softmax" or "zero-input")
    """

    def __init__(self,
                 mask_location: str,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.head = nn.Linear(self.head_in_features, self.hparams.output_dim)
        self.load_checkpoint()

        # Check the validity of 'mask_location` parameter
        if hasattr(vit, self.hparams.backbone_type):
            assert self.hparams.mask_location in ["pre-softmax", "post-softmax",
                                                  "zero-input", "zero-embedding",
                                                  "random-sampling"], f"'mask_location' for ViT models must be 'pre-softmax', 'post-softmax', 'zero-input', 'zero-embedding', or 'random-sampling', but {self.hparams.mask_location}"
        elif hasattr(cnn_models, self.hparams.backbone_type):
            assert self.hparams.mask_location == "zero-input", "'mask_location' for CNN models must be 'zero-input'"
        else:
            raise NotImplementedError("Not supported backbone type")
        # Set `num_players` variable.
        if hasattr(self.backbone, 'patch_embed'):
            self.num_players = self.backbone.patch_embed.num_patches
        else:
            self.num_players = 196

        # Set up modules for calculating metric
        classifier_utils.set_metrics(self)

    def forward(self, images, masks, mask_location=None):
        assert masks.shape[-1] == self.num_players
        mask_location = self.hparams.mask_location if mask_location is None else mask_location

        if self.backbone.__class__.__name__ == 'VisionTransformer':
            if mask_location in ['pre-softmax', 'post-softmax', 'zero-input', 'zero-embedding', 'random-sampling']:
                output = self.backbone(x=images, mask=masks, mask_location=mask_location)
                embedding_cls, embedding_tokens = output['x'], output['x_others']
                logits = self.head(embedding_cls)
                output.update({'logits': logits})
            else:
                raise ValueError(
                    "'mask_location' should be 'pre-softmax', 'post-softmax', 'zero-out', 'zero-embedding', 'random-sampling'")
        elif self.backbone.__class__.__name__ == 'ResNet':
            if mask_location == 'zero-input':
                if images.shape[2:4] == (224, 224) and masks.shape[1] == 196:
                    masks = masks.reshape(-1, 14, 14)
                    masks = torch.repeat_interleave(torch.repeat_interleave(masks, 16, dim=2), 16, dim=1)
                else:
                    raise NotImplementedError
                images_masked = images * masks.unsqueeze(1)
                out = self.backbone(images_masked)
                logits = self.head(out)
                output = {'logits': logits}
            else:
                raise ValueError("'mask_location' should be 'zero-out'")
        else:
            raise NotImplementedError("Not supported backbone type")

        return output

    def training_step(self, batch, batch_idx):
        images, masks, labels = batch["images"], batch["masks"], batch["labels"]
        logits = self(images, masks)['logits']
        loss = classifier_utils.compute_metrics(self, logits=logits, labels=labels, phase='train')
        return loss

    def training_epoch_end(self, outs):
        classifier_utils.epoch_wrapup(self, phase='train')

    def validation_step(self, batch, batch_idx):
        images, masks, labels = batch["images"], batch["masks"], batch["labels"]
        logits = self(images, masks)['logits']
        loss = classifier_utils.compute_metrics(self, logits=logits, labels=labels, phase='val')

    def validation_epoch_end(self, outs):
        classifier_utils.epoch_wrapup(self, phase='val')

    def test_step(self, batch, batch_idx):
        images, masks, labels = batch["images"], batch["masks"], batch["labels"]
        logits = self(images, masks)['logits']
        loss = classifier_utils.compute_metrics(self, logits=logits, labels=labels, phase='test')

    def test_epoch_end(self, outs):
        classifier_utils.epoch_wrapup(self, phase='test')
