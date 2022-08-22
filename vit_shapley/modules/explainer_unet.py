import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_shapley.modules import explainer_utils


class MultiConv(nn.Module):
    '''(convolution => [BN] => ReLU) * n'''

    def __init__(self, in_channels, out_channels, mid_channels=None,
                 num_convs=2, batchnorm=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if batchnorm:
            # Input conv.
            module_list = [
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)]

            # Middle convs.
            for _ in range(num_convs - 2):
                module_list = module_list + [
                    nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                              padding=1),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True)]

            # Output conv.
            module_list = module_list + [
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)]

        else:
            # Input conv.
            module_list = [
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)]

            # Middle convs.
            for _ in range(num_convs - 2):
                module_list = module_list + [
                    nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                              padding=1),
                    nn.ReLU(inplace=True)]

            # Output conv.
            module_list = module_list + [
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)]

        # Set up sequential.
        self.multi_conv = nn.Sequential(*module_list)

    def forward(self, x):
        return self.multi_conv(x)


class Down(nn.Module):
    '''
    Downscaling with maxpool then multiconv.
    Adapted from https://github.com/milesial/Pytorch-UNet
    '''

    def __init__(self, in_channels, out_channels, num_convs=2, batchnorm=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            MultiConv(in_channels, out_channels, num_convs=num_convs,
                      batchnorm=batchnorm)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    '''
    Upscaling then multiconv.
    Adapted from https://github.com/milesial/Pytorch-UNet
    '''

    def __init__(self, in_channels, out_channels, num_convs=2, bilinear=True,
                 batchnorm=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                                  align_corners=True)
            self.conv = MultiConv(in_channels, out_channels, in_channels // 2,
                                  num_convs, batchnorm)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                         kernel_size=2, stride=2)
            self.conv = MultiConv(in_channels, out_channels,
                                  num_convs=num_convs, batchnorm=batchnorm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self,
                 n_classes,
                 num_down,
                 num_up,
                 in_channels=3,
                 base_channels=64,
                 num_convs=2,
                 batchnorm=True,
                 bilinear=True):
        super().__init__()
        assert num_down >= num_up

        # Input conv.
        self.inc = MultiConv(in_channels, base_channels, num_convs=num_convs,
                             batchnorm=batchnorm)

        # Downsampling layers.
        down_layers = []
        channels = base_channels
        out_channels = 2 * channels
        for _ in range(num_down - 1):
            down_layers.append(
                Down(channels, out_channels, num_convs, batchnorm))
            channels = out_channels
            out_channels *= 2

        # Last down layer.
        factor = 2 if bilinear else 1
        down_layers.append(
            Down(channels, out_channels // factor, num_convs, batchnorm))
        self.down_layers = nn.ModuleList(down_layers)

        # Upsampling layers.
        up_layers = []
        channels *= 2
        out_channels = channels // 2
        for _ in range(num_up - 1):
            up_layers.append(
                Up(channels, out_channels // factor, num_convs, bilinear,
                   batchnorm))
            channels = out_channels
            out_channels = channels // 2

        # Last up layer.
        up_layers.append(
            Up(channels, out_channels, num_convs, bilinear, batchnorm))
        self.up_layers = nn.ModuleList(up_layers)

        # Output layer.
        self.outc = OutConv(out_channels, n_classes)

    def forward(self, x):
        # Input conv.
        x = self.inc(x)

        # Apply downsampling layers.
        x_list = []
        for down in self.down_layers:
            x = down(x)
            x_list.append(x)

        # Apply upsampling layers.
        for i, up in enumerate(self.up_layers):
            residual_x = x_list[-(i + 2)]
            x = up(x, residual_x)

        # Output.
        logits = self.outc(x)
        return logits


class ExplainerUNet(pl.LightningModule):
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
                 decay_power: str or None, warmup_steps: int or None):

        super().__init__()
        self.save_hyperparameters()
        self.__null = None

        self.logger_ = logging.getLogger(__name__)

        assert not (self.hparams.download_weight and self.hparams.load_path is not None), \
            "'download_weight' and 'load_path' cannot be activated at the same time as the downloaded weight will be overwritten by weights in 'load_path'."

        # Backbone initialization. (currently support only vit and cnn)
        self.backbone = UNet(n_classes=self.hparams.output_dim,
                             num_down=5,
                             num_up=1,
                             in_channels=3,
                             base_channels=64,
                             num_convs=2,
                             batchnorm=True,
                             bilinear=True)

        if self.hparams.download_weight:
            self.logger_.info("The backbone parameters were initialized with the downloaded pretrained weights.")
        else:
            self.logger_.info("The backbone parameters were randomly initialized.")

        # Load checkpoints
        if self.hparams.load_path is not None:
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
        pred = output.flatten(2, 3).transpose(1, 2)

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
