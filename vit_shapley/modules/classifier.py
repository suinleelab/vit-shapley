from typing import Union

import torch.nn as nn

from vit_shapley.modules.base_model import BaseModel
from vit_shapley.modules import classifier_utils


class Classifier(BaseModel):
    """
    `pytorch_lightning` module for image classifier

    Args:
        enable_pos_embed: wether to add positional embeddings to patch embeddings
        loss_weight: weighting of classes in cross-entropy loss
    """

    def __init__(self,
                 enable_pos_embed: bool,
                 loss_weight: Union[float, None] = None,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.head = nn.Linear(self.head_in_features, self.hparams.output_dim)
        self.load_checkpoint()

        if not self.hparams.enable_pos_embed:
            self.backbone.pos_embed.requires_grad = False
            self.backbone.pos_embed[:] = 0
        # Set up modules for calculating metric
        classifier_utils.set_metrics(self)

    def forward(self, images, output_attentions=False, output_hidden_states=False):
        if self.backbone.__class__.__name__ == 'VisionTransformer':
            output = self.backbone(images, output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states)
            embedding_cls, embedding_tokens = output['x'], output['x_others']
            # embedding_cls, embedding_tokens = self.backbone(images)
            logits = self.head(embedding_cls)
            output.update({'logits': logits})
        elif self.backbone.__class__.__name__ == 'ResNet':
            out = self.backbone(images)
            logits = self.head(out)
            output = {'logits': logits}
        else:
            raise NotImplementedError("Not supported backbone type")

        return output

    def training_step(self, batch, batch_idx):
        images, labels = batch["images"], batch["labels"]
        logits = self(batch["images"])['logits']
        loss = classifier_utils.compute_metrics(self, logits=logits, labels=labels, phase='train')
        return loss

    def training_epoch_end(self, outs):
        classifier_utils.epoch_wrapup(self, phase='train')

    def validation_step(self, batch, batch_idx):
        images, labels = batch["images"], batch["labels"]
        logits = self(batch["images"])['logits']
        loss = classifier_utils.compute_metrics(self, logits=logits, labels=labels, phase='val')

    def validation_epoch_end(self, outs):
        classifier_utils.epoch_wrapup(self, phase='val')

    def test_step(self, batch, batch_idx):
        images, labels = batch["images"], batch["labels"]
        logits = self(batch["images"])['logits']
        loss = classifier_utils.compute_metrics(self, logits=logits, labels=labels, phase='test')

    def test_epoch_end(self, outs):
        classifier_utils.epoch_wrapup(self, phase='test')
