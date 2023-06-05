import copy
import ipdb
import logging
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models as cnn_models

import vit_shapley.modules.vision_transformer as vit
from vit_shapley.modules import explainer_utils


class Explainer(pl.LightningModule):
    """
    `pytorch_lightning` module for surrogate

    Args:
        normalization: 'additive' or 'multiplicative'
        normalization_class: 'additive',
        activation:
        backbone_type: should be the class name defined in `torchvision.models.cnn_models` or `timm.models.vision_transformer`
        download_weight: whether to initialize backbone with the pretrained weights
        load_path: If not None. loads the weights saved in the checkpoint to the model
        target_type: `binary` or `multi-class` or `multi-label`
        output_dim: the dimension of output,

        explainer_head_num_attention_blocks:
        explainer_head_include_cls:
        explainer_head_num_mlp_layers:
        explainer_head_mlp_layer_ratio:
        explainer_norm:

        surrogate: 'surrogate' is a model takes masks as input
        link: link function for surrogate outputs (e.g., nn.Softmax).
        efficiency_lambda: lambda hyperparameter for efficiency penalty.
        efficiency_class_lambda: lambda hyperparameter for efficiency penalty.
        freeze_backbone: whether to freeze the backbone while training
        checkpoint_metric: the metric used to determine whether to save the current status as checkpoints during the validation phase
        optim_type: type of optimizer for optimizing parameters
        learning_rate: learning rate of optimizer
        weight_decay: weight decay of optimizer
        decay_power: only `cosine` annealing scheduler is supported currently
        warmup_steps: parameter for the `cosine` annealing scheduler
    """

    def __init__(self, normalization, normalization_class, activation, backbone_type: str, download_weight: bool,
                 load_path: str or None,
                 residual: list,
                 target_type: str, output_dim: int,
                 explainer_head_num_attention_blocks: int, explainer_head_include_cls: bool,
                 explainer_head_num_mlp_layers: int, explainer_head_mlp_layer_ratio: bool, explainer_norm: bool,
                 surrogate: pl.LightningModule, link: pl.LightningModule or nn.Module or None, efficiency_lambda,
                 efficiency_class_lambda,
                 freeze_backbone: bool, checkpoint_metric: str or None,
                 optim_type: str or None, learning_rate: float or None, weight_decay: float or None,
                 decay_power: str or None, warmup_steps: int or None, load_path_state_dict=False):

        super().__init__()
        self.save_hyperparameters()
        self.__null = None

        self.logger_ = logging.getLogger(__name__)

        assert not (self.hparams.download_weight and self.hparams.load_path is not None), \
            "'download_weight' and 'load_path' cannot be activated at the same time as the downloaded weight will be overwritten by weights in 'load_path'."

        # Backbone initialization. (currently support only vit and cnn)
        if hasattr(vit, self.hparams.backbone_type):
            self.backbone = getattr(vit, self.hparams.backbone_type)(pretrained=self.hparams.download_weight)
        elif hasattr(cnn_models, self.hparams.backbone_type):
            self.backbone = getattr(cnn_models, self.hparams.backbone_type)(pretrained=self.hparams.download_weight)
        else:
            raise NotImplementedError("Not supported backbone type")
        if self.hparams.download_weight:
            self.logger_.info("The backbone parameters were initialized with the downloaded pretrained weights.")
        else:
            self.logger_.info("The backbone parameters were randomly initialized.")

        # Nullify classification head built in the backbone module and rebuild.
        if self.backbone.__class__.__name__ == 'VisionTransformer':
            head_in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        elif self.backbone.__class__.__name__ == 'ResNet':
            head_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif self.backbone.__class__.__name__ == 'DenseNet':
            head_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise NotImplementedError("Not supported backbone type")

        if self.backbone.__class__.__name__ == 'VisionTransformer':
            # attention_blocks
            if self.hparams.explainer_head_num_attention_blocks == 0:
                self.attention_blocks = nn.Identity()
            else:
                self.attention_blocks = nn.ModuleList([vit.Block(dim=self.backbone.blocks[-1].mlp.fc1.in_features,
                                                                 num_heads=self.backbone.blocks[-1].attn.num_heads,
                                                                 mlp_ratio=(self.backbone.blocks[
                                                                                -1].mlp.fc1.out_features /
                                                                            self.backbone.blocks[
                                                                                -1].mlp.fc1.in_features),
                                                                 qkv_bias=self.backbone.blocks[
                                                                              -1].attn.qkv.bias is not None,
                                                                 drop=self.backbone.blocks[-1].mlp.drop1.p,
                                                                 attn_drop=self.backbone.blocks[-1].attn.attn_drop.p,
                                                                 drop_path=0 if isinstance(
                                                                     self.backbone.blocks[-1].drop_path,
                                                                     nn.Identity) else
                                                                 self.backbone.blocks[-1].drop_path.p,
                                                                 act_layer=self.backbone.blocks[-1].mlp.act.__class__,
                                                                 norm_layer=self.backbone.blocks[-1].norm1.__class__)
                                                       for
                                                       i in range(self.hparams.explainer_head_num_attention_blocks)])

                # self.backbone.norm = nn.Identity()
                # self.backbone.blocks[-1]=nn.Identity()

                self.attention_blocks[0].norm1 = nn.Identity()

                # self.attention_blocks[0].
                # self.backbone.blocks[-1].mlp.act.__class__()

            # mlps
            mlps_list = []
            if self.hparams.explainer_norm and self.hparams.explainer_head_num_attention_blocks > 0:
                mlps_list.append(nn.LayerNorm(self.backbone.num_features))

            if self.hparams.explainer_head_num_mlp_layers == 1:
                mlps_list.append(
                    nn.Linear(in_features=self.backbone.num_features, out_features=self.hparams.output_dim))
            elif self.hparams.explainer_head_num_mlp_layers == 2:
                mlps_list.append(
                    nn.Linear(in_features=self.backbone.num_features, out_features=int(
                        self.hparams.explainer_head_mlp_layer_ratio * self.backbone.num_features)))
                mlps_list.append(self.backbone.blocks[0].mlp.act.__class__())
                mlps_list.append(
                    nn.Linear(in_features=int(self.hparams.explainer_head_mlp_layer_ratio * self.backbone.num_features),
                              out_features=self.hparams.output_dim))
            elif self.hparams.explainer_head_num_mlp_layers == 3:
                mlps_list.append(
                    nn.Linear(in_features=self.backbone.num_features, out_features=int(
                        self.hparams.explainer_head_mlp_layer_ratio * self.backbone.num_features)))
                mlps_list.append(self.backbone.blocks[0].mlp.act.__class__())
                mlps_list.append(
                    nn.Linear(in_features=int(self.hparams.explainer_head_mlp_layer_ratio * self.backbone.num_features),
                              out_features=int(
                                  self.hparams.explainer_head_mlp_layer_ratio * self.backbone.num_features)))
                mlps_list.append(self.backbone.blocks[0].mlp.act.__class__())
                mlps_list.append(
                    nn.Linear(in_features=int(self.hparams.explainer_head_mlp_layer_ratio * self.backbone.num_features),
                              out_features=self.hparams.output_dim))
            elif self.hparams.explainer_head_num_mlp_layers == 4:
                mlps_list.append(
                    nn.Linear(in_features=self.backbone.num_features, out_features=int(
                        self.hparams.explainer_head_mlp_layer_ratio * self.backbone.num_features)))
                mlps_list.append(self.backbone.blocks[0].mlp.act.__class__())
                mlps_list.append(
                    nn.Linear(in_features=int(self.hparams.explainer_head_mlp_layer_ratio * self.backbone.num_features),
                              out_features=int(
                                  self.hparams.explainer_head_mlp_layer_ratio * self.backbone.num_features)))
                mlps_list.append(self.backbone.blocks[0].mlp.act.__class__())
                mlps_list.append(
                    nn.Linear(in_features=int(self.hparams.explainer_head_mlp_layer_ratio * self.backbone.num_features),
                              out_features=int(
                                  self.hparams.explainer_head_mlp_layer_ratio * self.backbone.num_features)))
                mlps_list.append(self.backbone.blocks[0].mlp.act.__class__())
                mlps_list.append(
                    nn.Linear(in_features=int(self.hparams.explainer_head_mlp_layer_ratio * self.backbone.num_features),
                              out_features=self.hparams.output_dim))

            self.mlps = nn.Sequential(*mlps_list)

        else:
            raise NotImplementedError("'explainer_head' is only implemented for VisionTransformer.")
        # Load checkpoints
        if self.hparams.load_path is not None:
            if load_path_state_dict:
                state_dict = torch.load(self.hparams.load_path, map_location="cpu")
            else:
                checkpoint = torch.load(self.hparams.load_path, map_location="cpu")
                state_dict = checkpoint["state_dict"]                
            ret = self.load_state_dict(state_dict, strict=False)
            self.logger_.info(f"Model parameters were updated from a checkpoint file {self.hparams.load_path}")
            self.logger_.info(f"Unmatched parameters - missing_keys:    {ret.missing_keys}")
            self.logger_.info(f"Unmatched parameters - unexpected_keys: {ret.unexpected_keys}")

        # Set up link function
        if self.hparams.link is None:
            self.link = lambda x: x
        elif self.hparams.link == 'logsoftmax':
            self.link = torch.nn.LogSoftmax(dim=1)
        elif self.hparams.link == 'softmax':
            self.link = torch.nn.Softmax(dim=1)
        elif self.hparams.link == 'sigmoid':
            self.link = torch.nn.Sigmoid()
        else:
            raise ValueError('unsupported link: {} function'.format(self.hparams.link))

        # Set up normalization.
        if self.hparams.normalization is None:
            self.normalization = None
        elif self.hparams.normalization == 'additive':
            # (batch, num_players, num_classes), (batch, 1, num_classes), (batch, 1, num_classes)
            self.normalization = lambda pred, grand, null: pred + ((grand - null) - torch.sum(pred, dim=1)).unsqueeze(
                1) / pred.shape[1]
        elif self.hparams.normalization == 'multiplicative':
            self.normalization = lambda pred, grand, null: pred * ((grand - null) / torch.sum(pred, dim=1)).unsqueeze(1)
        else:
            raise ValueError('unsupported normalization: {}'.format(self.hparams.normalization))

        # Set up normalization.
        if self.hparams.normalization_class is None:
            self.normalization_class = None
        elif self.hparams.normalization_class == 'additive':
            # (batch, num_players, num_classes)
            self.normalization_class = lambda pred: pred - torch.sum(pred, dim=2).unsqueeze(2) / pred.shape[2]
        else:
            raise ValueError('unsupported normalization: {}'.format(self.hparams.normalization_class))

        # freeze backbone
        if self.hparams.freeze_backbone == 'all':
            for key, value in self.backbone.named_parameters():
                value.requires_grad = False
        elif self.hparams.freeze_backbone == 'except_last':
            for key, value in self.backbone.named_parameters():
                value.requires_grad = False
            for key, value in self.backbone.blocks[-1].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.norm.named_parameters():
                value.requires_grad = True
        elif self.hparams.freeze_backbone == 'except_last_two':
            for key, value in self.backbone.named_parameters():
                value.requires_grad = False
            for key, value in self.backbone.blocks[-2].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-1].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.norm.named_parameters():
                value.requires_grad = True
        elif self.hparams.freeze_backbone == 'except_last_three':
            for key, value in self.backbone.named_parameters():
                value.requires_grad = False
            for key, value in self.backbone.blocks[-3].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-2].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-1].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.norm.named_parameters():
                value.requires_grad = True
        elif self.hparams.freeze_backbone == 'except_last_four':
            for key, value in self.backbone.named_parameters():
                value.requires_grad = False
            for key, value in self.backbone.blocks[-4].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-3].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-2].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-1].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.norm.named_parameters():
                value.requires_grad = True
        elif self.hparams.freeze_backbone == 'except_last_five':
            for key, value in self.backbone.named_parameters():
                value.requires_grad = False
            for key, value in self.backbone.blocks[-5].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-4].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-3].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-2].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-1].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.norm.named_parameters():
                value.requires_grad = True
        elif self.hparams.freeze_backbone == 'except_last_seven':
            for key, value in self.backbone.named_parameters():
                value.requires_grad = False
            for key, value in self.backbone.blocks[-7].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-6].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-5].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-4].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-3].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-2].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-1].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.norm.named_parameters():
                value.requires_grad = True
        elif self.hparams.freeze_backbone == 'except_last_nine':
            for key, value in self.backbone.named_parameters():
                value.requires_grad = False
            for key, value in self.backbone.blocks[-9].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-8].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-7].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-6].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-5].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-4].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-3].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-2].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-1].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.norm.named_parameters():
                value.requires_grad = True
        elif self.hparams.freeze_backbone == 'except_last_ten':
            for key, value in self.backbone.named_parameters():
                value.requires_grad = False
            for key, value in self.backbone.blocks[-10].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-9].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-8].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-7].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-6].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-5].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-4].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-3].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-2].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-1].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.norm.named_parameters():
                value.requires_grad = True
        elif self.hparams.freeze_backbone == 'except_last_eleven':
            for key, value in self.backbone.named_parameters():
                value.requires_grad = False
            for key, value in self.backbone.blocks[-11].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-10].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-9].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-8].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-7].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-6].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-5].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-4].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-3].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-2].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-1].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.norm.named_parameters():
                value.requires_grad = True
        elif self.hparams.freeze_backbone == 'except_last_twelve':
            for key, value in self.backbone.named_parameters():
                value.requires_grad = False
            for key, value in self.backbone.blocks[-12].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-11].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-10].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-9].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-8].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-7].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-6].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-5].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-4].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-3].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-2].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.blocks[-1].named_parameters():
                value.requires_grad = True
            for key, value in self.backbone.norm.named_parameters():
                value.requires_grad = True
        elif self.hparams.freeze_backbone == 'none':
            pass
        else:
            raise ValueError("unsupported freeze_backbone: {}".format(self.hparams.freeze_backbone))

        # Set up modules for calculating metric
        explainer_utils.set_metrics(self)

        # self.hparams.surrogate_ = copy.deepcopy(self.hparams.surrogate)
        # self.hparams.surrogate_ = self.hparams.surrogate

        # self.hparams.surrogate.backbone.norm = nn.Identity()

    def configure_optimizers(self):
        return explainer_utils.set_schedule(self)

    def null(self, images: torch.Tensor or None = None) -> torch.Tensor:
        """
        calculate or load cached null

        Args:
            images: torch.Tensor (batch, channel, height, width)
        Returns:
            values: torch.Tensor (1, num_classes)
        """
        if hasattr(self, '__null'):
            return self.__null
        else:
            if images is not None:
                self.hparams.surrogate.eval()
                with torch.no_grad():
                    self.__null = self.link(
                        self.hparams.surrogate(images[0:1].to(self.hparams.surrogate.device),
                                               torch.zeros(1, self.hparams.surrogate.num_players,
                                                           device=self.hparams.surrogate.device))['logits']).to(
                        self.device)  # (batch, channel, height, weight) -> (1, num_classes)
                return self.__null
            else:
                raise RuntimeError(
                    "You should call explainer.null(x) at least once to get null value. As 'x' is just used for guessing the shape of input, any dummy variable is okay.")

    def grand(self, images):
        self.hparams.surrogate.eval()
        with torch.no_grad():
            grand = self.link(self.hparams.surrogate(images=images.to(self.hparams.surrogate.device),
                                                     # (batch, channel, height, weight)
                                                     masks=torch.ones(images.shape[0],
                                                                      self.hparams.surrogate.num_players,
                                                                      device=self.hparams.surrogate.device)
                                                     # (batch, num_players)
                                                     )['logits']).to(self.device)  # (1, num_classes)
        return grand

    def surrogate_multiple_masks(self, images, multiple_masks=None):
        """
        forward pass for embedded surrogate model.
        Args:
            images: torch.Tensor (batch, channel, height, width)
            multiple_masks: torch.Tensor (batch, num_mask_samples, num_players)

        Returns:
            surrogate_values: torch.Tensor (batch, num_mask_samples, num_classes)

        """
        # evaluate surrogate
        self.hparams.surrogate.eval()
        with torch.no_grad():
            # mask
            assert len(multiple_masks.shape) == 3  # (batch, num_mask_samples, num_players)
            batch_size = multiple_masks.shape[0]
            assert multiple_masks.shape[0] == images.shape[0]
            num_mask_samples = multiple_masks.shape[1]
            assert self.hparams.surrogate.num_players == multiple_masks.shape[2]
            surrogate_values = self.link(self.hparams.surrogate(
                images=images.repeat_interleave(num_mask_samples, dim=0).to(self.hparams.surrogate.device),
                # (batch, channel, height, weight) -> (batch * num_mask_samples, channel, height, weight)
                masks=multiple_masks.flatten(0, 1).to(self.hparams.surrogate.device)
                # (batch, num_mask_samples, num_players) -> (batch * num_mask_samples, num_players)
            )['logits']).reshape(batch_size, num_mask_samples, -1).to(
                self.device)  # (batch, num_mask_samples, num_classes)

        return surrogate_values

    def forward(self, images, surrogate_grand=None, surrogate_null=None):
        """
        forward pass
        Args:
            residual:
            surrogate_grand:
            surrogate_null:
            images: torch.Tensor (batch, channel, height, width)

        Returns:
            pred: torch.Tensor (batch, num_players, num_classes)
            pred_sum: torch.Tensor (batch, num_classes)

        """
        output = self.backbone(x=images)
        embedding_cls, embedding_tokens = output['x'], output['x_others']

        if self.hparams.explainer_head_include_cls:
            embedding_all = torch.cat([embedding_cls.unsqueeze(dim=1), embedding_tokens], dim=1)
        else:
            embedding_all = embedding_tokens

        if self.attention_blocks.__class__.__name__ == 'Identity':
            pass
        else:
            for i, layer_module in enumerate(self.attention_blocks):
                layer_outputs = layer_module(x=embedding_all)
                embedding_all = layer_outputs[0]

        if self.hparams.explainer_head_include_cls:
            pred = self.mlps(embedding_all)[:, 1:]
        else:
            pred = self.mlps(embedding_all)

        if self.hparams.activation is None:
            pass
        elif self.hparams.activation == 'tanh':
            pred = pred.tanh()
        else:
            raise ValueError('unsupported activation: {}'.format(self.hparams.activation))

            # pred = pred.tanh()
            # pred = pred.exp()-1

            # pred = pred.exp()
            # pred = (pred.shape[-1] * pred - pred.sum(axis=-1).unsqueeze(2))

        if self.normalization:
            if surrogate_grand is None:
                surrogate_grand = self.grand(images).to(
                    self.device)  # (batch, channel, height, weight) -> (batch, num_classes)
            if surrogate_null is None:
                surrogate_null = self.null(images).to(
                    self.device)  # (batch, channel, height, weight) -> (1, num_classes)
            pred = self.normalization(pred=pred, grand=surrogate_grand, null=surrogate_null)

        if self.normalization_class:
            pred = self.normalization_class(pred=pred)

        pred_sum = pred.sum(dim=1)  # (batch, num_players, num_classes) -> (batch, num_classes)
        pred_sum_class = pred.sum(dim=2)  # (batch, num_players, num_classes) -> (batch, num_players)

        return pred, pred_sum, pred_sum_class

    def training_step(self, batch, batch_idx):
        images, masks = batch["images"], batch["masks"]

        # evaluate surrogate
        surrogate_values = self.surrogate_multiple_masks(images,
                                                         masks)  # (batch, channel, height, width), (batch, num_mask_samples, num_players) -> (batch, num_mask_samples, num_classes)
        surrogate_grand = self.grand(images)  # (batch, channel, height, weight) -> (batch, num_classes)
        surrogate_null = self.null(images)  # (batch, channel, height, weight) -> (1, num_classes)

        # evaluate explainer
        values_pred, value_pred_beforenorm_sum, value_pred_beforenorm_sum_class = self(images,
                                                                                       surrogate_grand=surrogate_grand,
                                                                                       surrogate_null=surrogate_null)  # (batch, channel, height, weight) -> (batch, num_players, num_classes), (batch, num_classes)

        value_pred_approx = surrogate_null + masks.float() @ values_pred  # (1, num_classes) + (batch, num_mask_samples, num_players) @ (batch, num_players, num_classes) -> (batch, num_mask_samples, num_classes)

        value_diff = self.hparams.surrogate.num_players * F.mse_loss(input=value_pred_approx, target=surrogate_values,
                                                                     reduction='mean')

        loss = explainer_utils.compute_metrics(self,
                                               num_players=self.hparams.surrogate.num_players,
                                               values_pred=value_pred_approx,
                                               values_target=surrogate_values,
                                               efficiency_lambda=self.hparams.efficiency_lambda,
                                               value_pred_beforenorm_sum=value_pred_beforenorm_sum,
                                               surrogate_grand=surrogate_grand,
                                               surrogate_null=surrogate_null,
                                               efficiency_class_lambda=self.hparams.efficiency_class_lambda,
                                               value_pred_beforenorm_sum_class=value_pred_beforenorm_sum_class,
                                               phase='train')

        return loss

    def training_epoch_end(self, outs):
        explainer_utils.epoch_wrapup(self, phase='train')

    def validation_step(self, batch, batch_idx):
        images, masks = batch["images"], batch["masks"]

        # evaluate surrogate
        surrogate_values = self.surrogate_multiple_masks(images, masks)
        surrogate_grand = self.grand(images)  # (batch, channel, height, weight) -> (batch, num_classes)
        surrogate_null = self.null(images)  # (batch, channel, height, weight) -> (1, num_classes)

        # evaluate explainer
        values_pred, value_pred_beforenorm_sum, value_pred_beforenorm_sum_class = self(images,
                                                                                       surrogate_grand=surrogate_grand,
                                                                                       surrogate_null=surrogate_null)  # (batch, channel, height, weight) -> (batch, num_players, num_classes), (batch, num_classes)

        value_pred_approx = surrogate_null + masks.float() @ values_pred  # (1, num_classes) + (batch, num_mask_samples, num_players) @ (batch, num_players, num_classes) -> (batch, num_mask_samples, num_classes)

        loss = explainer_utils.compute_metrics(self,
                                               num_players=self.hparams.surrogate.num_players,
                                               values_pred=value_pred_approx,
                                               values_target=surrogate_values,
                                               efficiency_lambda=self.hparams.efficiency_lambda,
                                               value_pred_beforenorm_sum=value_pred_beforenorm_sum,
                                               surrogate_grand=surrogate_grand,
                                               surrogate_null=surrogate_null,
                                               efficiency_class_lambda=self.hparams.efficiency_class_lambda,
                                               value_pred_beforenorm_sum_class=value_pred_beforenorm_sum_class,
                                               phase='val')

        return loss

    def validation_epoch_end(self, outs):
        explainer_utils.epoch_wrapup(self, phase='val')

    def test_step(self, batch, batch_idx):
        images, masks = batch["images"], batch["masks"]

        # evaluate surrogate
        surrogate_values = self.surrogate_multiple_masks(images, masks)
        surrogate_grand = self.grand(images)  # (batch, channel, height, weight) -> (batch, num_classes)
        surrogate_null = self.null(images)  # (batch, channel, height, weight) -> (1, num_classes)

        # evaluate explainer
        values_pred, value_pred_beforenorm_sum, value_pred_beforenorm_sum_class = self(images,
                                                                                       surrogate_grand=surrogate_grand,
                                                                                       surrogate_null=surrogate_null)  # (batch, channel, height, weight) -> (batch, num_players, num_classes), (batch, num_classes), (batch, num_players)

        value_pred_approx = surrogate_null + masks.float() @ values_pred  # (1, num_classes) + (batch, num_mask_samples, num_players) @ (batch, num_players, num_classes) -> (batch, num_mask_samples, num_classes)

        loss = explainer_utils.compute_metrics(self,
                                               num_players=self.hparams.surrogate.num_players,
                                               values_pred=value_pred_approx,
                                               values_target=surrogate_values,
                                               efficiency_lambda=self.hparams.efficiency_lambda,
                                               value_pred_beforenorm_sum=value_pred_beforenorm_sum,
                                               surrogate_grand=surrogate_grand,
                                               surrogate_null=surrogate_null,
                                               efficiency_class_lambda=self.hparams.efficiency_class_lambda,
                                               value_pred_beforenorm_sum_class=value_pred_beforenorm_sum_class,
                                               phase='test')
        return loss

    def test_epoch_end(self, outs):
        explainer_utils.epoch_wrapup(self, phase='test')


if __name__ == '__main__':
    from vit_shapley.modules.surrogate import Surrogate
    import os

    os.chdir("../../")

    surrogate = Surrogate(mask_location="pre-softmax",
                          backbone_type="deit_small_patch16_224",
                          download_weight=False,
                          load_path="results/vit_project/rqhwty12/checkpoints/epoch=23-step=3527.ckpt",
                          target_type="multiclass",
                          output_dim=10,

                          target_model=None,
                          checkpoint_metric=None,
                          optim_type=None,
                          learning_rate=None,
                          weight_decay=None,
                          decay_power=None,
                          warmup_steps=None)

    print(1)
    with torch.no_grad():
        out = surrogate(torch.rand(1, 3, 224, 224), torch.ones(1, 196))
    print(1)
