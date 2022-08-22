import copy
import os
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from vit_shapley.config import ex
from vit_shapley.datamodules.ImageNette_datamodule import ImageNetteDataModule
from vit_shapley.datamodules.MURA_datamodule import MURADataModule
from vit_shapley.modules.classifier import Classifier
from vit_shapley.modules.classifier_masked import ClassifierMasked
from vit_shapley.modules.explainer import Explainer
from vit_shapley.modules.explainer_unet import ExplainerUNet
from vit_shapley.modules.surrogate import Surrogate


def set_datamodule(datasets,
                   dataset_location,
                   explanation_location_train,
                   explanation_mask_amount_train,
                   explanation_mask_ascending_train,

                   explanation_location_val,
                   explanation_mask_amount_val,
                   explanation_mask_ascending_val,

                   explanation_location_test,
                   explanation_mask_amount_test,
                   explanation_mask_ascending_test,

                   transforms_train,
                   transforms_val,
                   transforms_test,
                   num_workers,
                   per_gpu_batch_size,
                   test_data_split):
    dataset_parameters = {
        "dataset_location": dataset_location,
        "explanation_location_train": explanation_location_train,
        "explanation_mask_amount_train": explanation_mask_amount_train,
        "explanation_mask_ascending_train": explanation_mask_ascending_train,

        "explanation_location_val": explanation_location_val,
        "explanation_mask_amount_val": explanation_mask_amount_val,
        "explanation_mask_ascending_val": explanation_mask_ascending_val,

        "explanation_location_test": explanation_location_test,
        "explanation_mask_amount_test": explanation_mask_amount_test,
        "explanation_mask_ascending_test": explanation_mask_ascending_test,

        "transforms_train": transforms_train,
        "transforms_val": transforms_val,
        "transforms_test": transforms_test,
        "num_workers": num_workers,
        "per_gpu_batch_size": per_gpu_batch_size,
        "test_data_split": test_data_split
    }

    if datasets == "MURA":
        datamodule = MURADataModule(**dataset_parameters)
    elif datasets == "ImageNette":
        datamodule = ImageNetteDataModule(**dataset_parameters)
    else:
        ValueError("Invalid 'datasets' configuration")
    return datamodule


def set_wandb_logger(exp_name, wandb_project_name, log_dir, log_model):
    os.makedirs(log_dir, exist_ok=True)
    wandb_logger = WandbLogger(project=wandb_project_name, name=exp_name, save_dir=log_dir,
                               log_model=log_model)  # , name=exp_name, config=_config)
    return wandb_logger


def get_grad_steps(gpus, batch_size, per_gpu_batch_size, num_nodes):
    # Calculate the number of steps to accumulate gradients
    if isinstance(gpus, int):
        gpus = gpus
    elif isinstance(gpus, list):
        gpus = len(gpus)
    elif len(gpus) == 0:
        gpus = 1
    else:
        raise NotImplementedError
    grad_steps = batch_size // (per_gpu_batch_size * gpus * num_nodes)
    return grad_steps


def generate_mask(num_players: int, num_mask_samples: int or None = None, paired_mask_samples: bool = True,
                  mode: str = 'uniform', random_state: np.random.RandomState or None = None) -> np.array:
    """
    Args:
        num_players: the number of players in the coalitional game
        num_mask_samples: the number of masks to generate
        paired_mask_samples: if True, the generated masks are pairs of x and 1-x.
        mode: the distribution that the number of masked features follows. ('uniform' or 'shapley')
        random_state: random generator

    Returns:
        torch.Tensor of shape
        (num_masks, num_players) if num_masks is int
        (num_players) if num_masks is None

    """
    random_state = random_state or np.random

    num_samples_ = num_mask_samples or 1

    if paired_mask_samples:
        assert num_samples_ % 2 == 0, "'num_samples' must be a multiple of 2 if 'paired' is True"
        num_samples_ = num_samples_ // 2
    else:
        num_samples_ = num_samples_

    if mode == 'uniform':
        masks = (random_state.rand(num_samples_, num_players) > random_state.rand(num_samples_, 1)).astype('int')
    elif mode == 'shapley':
        probs = 1 / (np.arange(1, num_players) * (num_players - np.arange(1, num_players)))
        probs = probs / probs.sum()
        masks = (random_state.rand(num_samples_, num_players) > 1 / num_players * random_state.choice(
            np.arange(num_players - 1), p=probs, size=[num_samples_, 1])).astype('int')
    else:
        raise ValueError("'mode' must be 'random' or 'shapley'")

    if paired_mask_samples:
        masks = np.stack([masks, 1 - masks], axis=1).reshape(num_samples_ * 2, num_players)

    if num_mask_samples is None:
        masks = masks.squeeze(0)
        return masks  # (num_masks)
    else:
        return masks  # (num_samples, num_masks)


@ex.automain
def main(_config):
    print('-----------config------------\n', _config, '\n')
    _config = copy.deepcopy(_config)

    # (1) Set global seed
    pl.seed_everything(seed=_config["seed"])

    # (2) Initialize WandB logger
    # date and time is appended to the 'exp_name' of WandB
    _config["exp_name"] = datetime.now().strftime("%y%m%d_%H%M%S") + "_" + _config['exp_name']
    wandb_logger = set_wandb_logger(exp_name=_config["exp_name"],
                                    wandb_project_name=_config["wandb_project_name"],
                                    log_dir=_config["log_dir"],
                                    log_model=(_config["explanation_location_train"] is None))
    wandb_logger.experiment.config.update(_config)

    # (3) Initialize `pytorch_lightning` model
    if _config["stage"] == "classifier":
        classifier = Classifier(backbone_type=_config['classifier_backbone_type'],
                                download_weight=_config['classifier_download_weight'],
                                load_path=_config["classifier_load_path"],
                                target_type=_config["target_type"],
                                output_dim=_config["output_dim"],
                                enable_pos_embed=_config["classifier_enable_pos_embed"],

                                checkpoint_metric=_config["checkpoint_metric"],
                                optim_type=_config["optim_type"],
                                learning_rate=_config["learning_rate"],
                                loss_weight=_config["loss_weight"],
                                weight_decay=_config["weight_decay"],
                                decay_power=_config["decay_power"],
                                warmup_steps=_config["warmup_steps"])
        model_to_train = classifier
        gpus = _config["gpus_classifier"]
    elif _config["stage"] == "classifier_masked":
        classifier_masked = ClassifierMasked(mask_location=_config['classifier_masked_mask_location'],
                                             backbone_type=_config['classifier_backbone_type'],
                                             download_weight=_config['classifier_download_weight'],
                                             load_path=_config["classifier_load_path"],
                                             target_type=_config["target_type"],
                                             output_dim=_config["output_dim"],

                                             checkpoint_metric=_config["checkpoint_metric"],
                                             optim_type=_config["optim_type"],
                                             learning_rate=_config["learning_rate"],
                                             loss_weight=_config["loss_weight"],
                                             weight_decay=_config["weight_decay"],
                                             decay_power=_config["decay_power"],
                                             warmup_steps=_config["warmup_steps"])
        model_to_train = classifier_masked
        gpus = _config["gpus_classifier"]
    elif _config["stage"] == "surrogate":
        classifier = Classifier(backbone_type=_config['classifier_backbone_type'],
                                download_weight=_config['classifier_download_weight'],
                                load_path=_config["classifier_load_path"],
                                target_type=_config["target_type"],
                                output_dim=_config["output_dim"],
                                enable_pos_embed=_config["classifier_enable_pos_embed"],

                                checkpoint_metric=None,
                                optim_type=None,
                                learning_rate=None,
                                loss_weight=None,
                                weight_decay=None,
                                decay_power=None,
                                warmup_steps=None).to(_config["gpus_classifier"][0])

        surrogate = Surrogate(mask_location=_config["surrogate_mask_location"],
                              backbone_type=_config['surrogate_backbone_type'],
                              download_weight=_config['surrogate_download_weight'],
                              load_path=_config["surrogate_load_path"],
                              target_type=_config["target_type"],
                              output_dim=_config["output_dim"],

                              target_model=classifier,
                              checkpoint_metric=_config["checkpoint_metric"],
                              optim_type=_config["optim_type"],
                              learning_rate=_config["learning_rate"],
                              weight_decay=_config["weight_decay"],
                              decay_power=_config["decay_power"],
                              warmup_steps=_config["warmup_steps"])
        model_to_train = surrogate
        gpus = _config["gpus_surrogate"]
    elif _config["stage"] == "explainer":
        surrogate = Surrogate(mask_location=_config["surrogate_mask_location"],
                              backbone_type=_config['surrogate_backbone_type'],
                              download_weight=_config['surrogate_download_weight'],
                              load_path=_config["surrogate_load_path"],
                              target_type=_config["target_type"],
                              output_dim=_config["output_dim"],

                              target_model=None,
                              checkpoint_metric=None,
                              optim_type=None,
                              learning_rate=None,
                              weight_decay=None,
                              decay_power=None,
                              warmup_steps=None).to(_config["gpus_surrogate"][0])
        if _config["explainer_backbone_type"] == "unet":
            explainer = ExplainerUNet(normalization=_config["explainer_normalization"],
                                      normalization_class=_config["explainer_normalization_class"],
                                      activation=_config["explainer_activation"],
                                      surrogate=surrogate,
                                      link=_config["explainer_link"],
                                      backbone_type=_config["explainer_backbone_type"],
                                      download_weight=_config['explainer_download_weight'],
                                      load_path=_config["explainer_load_path"],
                                      residual=_config["explainer_residual"],
                                      target_type=_config["target_type"],
                                      output_dim=_config["output_dim"],

                                      explainer_head_num_attention_blocks=_config[
                                          "explainer_head_num_attention_blocks"],
                                      explainer_head_include_cls=_config["explainer_head_include_cls"],
                                      explainer_head_num_mlp_layers=_config["explainer_head_num_mlp_layers"],
                                      explainer_head_mlp_layer_ratio=_config["explainer_head_mlp_layer_ratio"],
                                      explainer_norm=_config["explainer_norm"],

                                      efficiency_lambda=_config["explainer_efficiency_lambda"],
                                      efficiency_class_lambda=_config["explainer_efficiency_class_lambda"],
                                      freeze_backbone=_config["explainer_freeze_backbone"],

                                      checkpoint_metric=_config["checkpoint_metric"],
                                      optim_type=_config["optim_type"],
                                      learning_rate=_config["learning_rate"],
                                      weight_decay=_config["weight_decay"],
                                      decay_power=_config["decay_power"],
                                      warmup_steps=_config["warmup_steps"])
        else:
            explainer = Explainer(normalization=_config["explainer_normalization"],
                                  normalization_class=_config["explainer_normalization_class"],
                                  activation=_config["explainer_activation"],
                                  surrogate=surrogate,
                                  link=_config["explainer_link"],
                                  backbone_type=_config["explainer_backbone_type"],
                                  download_weight=_config['explainer_download_weight'],
                                  load_path=_config["explainer_load_path"],
                                  residual=_config["explainer_residual"],
                                  target_type=_config["target_type"],
                                  output_dim=_config["output_dim"],

                                  explainer_head_num_attention_blocks=_config["explainer_head_num_attention_blocks"],
                                  explainer_head_include_cls=_config["explainer_head_include_cls"],
                                  explainer_head_num_mlp_layers=_config["explainer_head_num_mlp_layers"],
                                  explainer_head_mlp_layer_ratio=_config["explainer_head_mlp_layer_ratio"],
                                  explainer_norm=_config["explainer_norm"],

                                  efficiency_lambda=_config["explainer_efficiency_lambda"],
                                  efficiency_class_lambda=_config["explainer_efficiency_class_lambda"],
                                  freeze_backbone=_config["explainer_freeze_backbone"],

                                  checkpoint_metric=_config["checkpoint_metric"],
                                  optim_type=_config["optim_type"],
                                  learning_rate=_config["learning_rate"],
                                  weight_decay=_config["weight_decay"],
                                  decay_power=_config["decay_power"],
                                  warmup_steps=_config["warmup_steps"])
        model_to_train = explainer
        gpus = _config["gpus_explainer"]
    else:
        raise NotImplementedError

    # (4) Initialize datamodule
    datamodule = set_datamodule(datasets=_config["datasets"],
                                dataset_location=_config["dataset_location"],

                                explanation_location_train=_config["explanation_location_train"],
                                explanation_mask_amount_train=_config["explanation_mask_amount_train"],
                                explanation_mask_ascending_train=_config["explanation_mask_ascending_train"],

                                explanation_location_val=_config["explanation_location_val"],
                                explanation_mask_amount_val=_config["explanation_mask_amount_val"],
                                explanation_mask_ascending_val=_config["explanation_mask_ascending_val"],

                                explanation_location_test=_config["explanation_location_test"],
                                explanation_mask_amount_test=_config["explanation_mask_amount_test"],
                                explanation_mask_ascending_test=_config["explanation_mask_ascending_test"],

                                transforms_train=_config["transforms_train"],
                                transforms_val=_config["transforms_val"],
                                transforms_test=_config["transforms_test"],
                                num_workers=_config["num_workers"],
                                per_gpu_batch_size=_config["per_gpu_batch_size"],
                                test_data_split=_config["test_data_split"])

    if _config["stage"] == "classifier":
        pass
    elif _config["stage"] == "classifier_masked":
        # The batch for training classifier consists of images and labels, but the batch for training surrogate consists of images and masks.
        # The masks are generated to have evenly distributed cardinality (i.e., the number of masked features is evenly sampled between 0 and num_patches)
        original_getitem = copy.deepcopy(datamodule.dataset_cls.__getitem__)

        def __getitem__(self, idx):
            if self.split == 'train':
                masks = generate_mask(num_players=self.num_players,
                                      num_mask_samples=self.num_mask_samples,
                                      paired_mask_samples=self.paired_mask_samples,
                                      mode=self.mode,
                                      random_state=None)
            elif self.split == 'val' or self.split == 'test':
                # initialize cache if not initialized yet.
                if not hasattr(self, "masks_cached"):
                    self.masks_cached = {}
                # get cached if available
                random_state = np.random.RandomState(idx)
                masks = self.masks_cached.setdefault(idx, generate_mask(num_players=self.num_players,
                                                                        num_mask_samples=self.num_mask_samples,
                                                                        paired_mask_samples=self.paired_mask_samples,
                                                                        mode=self.mode,
                                                                        random_state=random_state))
            else:
                raise ValueError("'split' variable must be train, val or test.")
            original_item = original_getitem(self, idx)
            return {"images": original_item["images"],
                    "labels": original_item["labels"],
                    "masks": masks}

        datamodule.dataset_cls.__getitem__ = __getitem__
        datamodule.dataset_cls.num_players = classifier_masked.num_players
        datamodule.dataset_cls.num_mask_samples = None
        datamodule.dataset_cls.paired_mask_samples = False
        datamodule.dataset_cls.mode = 'uniform'
    elif _config["stage"] == "surrogate":
        # The batch for training classifier consists of images and labels, but the batch for training surrogate consists of images and masks.
        # The masks are generated to have evenly distributed cardinality (i.e., the number of masked features is evenly sampled between 0 and num_patches)
        original_getitem = copy.deepcopy(datamodule.dataset_cls.__getitem__)

        def __getitem__(self, idx):
            if self.split == 'train':
                masks = generate_mask(num_players=self.num_players,
                                      num_mask_samples=self.num_mask_samples,
                                      paired_mask_samples=self.paired_mask_samples,
                                      mode=self.mode,
                                      random_state=None)
            elif self.split == 'val' or self.split == 'test':
                # initialize cache if not initialized yet.
                if not hasattr(self, "masks_cached"):
                    self.masks_cached = {}
                # get cached if available
                random_state = np.random.RandomState(idx)
                masks = self.masks_cached.setdefault(idx, generate_mask(num_players=self.num_players,
                                                                        num_mask_samples=self.num_mask_samples,
                                                                        paired_mask_samples=self.paired_mask_samples,
                                                                        mode=self.mode,
                                                                        random_state=random_state))
            else:
                raise ValueError("'split' variable must be train, val or test.")
            return {"images": original_getitem(self, idx)["images"],
                    "masks": masks}

        datamodule.dataset_cls.__getitem__ = __getitem__
        datamodule.dataset_cls.num_players = surrogate.num_players
        datamodule.dataset_cls.num_mask_samples = None
        datamodule.dataset_cls.paired_mask_samples = False
        datamodule.dataset_cls.mode = 'uniform'

    elif _config["stage"] == "explainer":
        # The batch for training classifier consists of images and labels, but the batch for training explainer consists of images and masks.
        # The masks are generated to follow the Shapley distribution.
        original_getitem = copy.deepcopy(datamodule.dataset_cls.__getitem__)

        def __getitem__(self, idx):
            if self.split == 'train':
                masks = generate_mask(num_players=self.num_players,
                                      num_mask_samples=self.num_mask_samples,
                                      paired_mask_samples=self.paired_mask_samples,
                                      mode=self.mode,
                                      random_state=None)
            elif self.split == 'val' or self.split == 'test':
                # initialize cache if not initialized yet.
                if not hasattr(self, "masks_cached"):
                    self.masks_cached = {}
                # get cached if available
                random_state = np.random.RandomState(idx)
                masks = self.masks_cached.setdefault(idx, generate_mask(num_players=self.num_players,
                                                                        num_mask_samples=self.num_mask_samples,
                                                                        paired_mask_samples=self.paired_mask_samples,
                                                                        mode=self.mode,
                                                                        random_state=random_state))
                # if idx == 300:
                #     print(masks)
                #     print(masks.sum(axis=1))
            else:
                raise ValueError("'split' variable must be train, val or test.")
            return {"images": original_getitem(self, idx)["images"],
                    "masks": masks}

        datamodule.dataset_cls.__getitem__ = __getitem__
        datamodule.dataset_cls.num_players = surrogate.num_players
        datamodule.dataset_cls.num_mask_samples = 2
        datamodule.dataset_cls.paired_mask_samples = True
        datamodule.dataset_cls.mode = 'shapley'

    else:
        raise NotImplementedError

    # (5) Initialize `pytorch_lightning` Trainer
    grad_steps = get_grad_steps(gpus=gpus,
                                batch_size=_config["batch_size"],
                                per_gpu_batch_size=_config["per_gpu_batch_size"],
                                num_nodes=_config["num_nodes"])
    print(model_to_train)

    class NumMaskSampleControl(pl.callbacks.Callback):
        def __init__(self, epoch, num_mask_samples, paired_mask_samples):
            # assert epoch > 0, "'epoch' of 'NumMaskSampleControl' must be larger than 0"
            self.epoch = epoch
            self.num_mask_samples = num_mask_samples
            self.paired_mask_samples = paired_mask_samples

        def on_epoch_start(self, trainer, pl_module):
            if trainer.current_epoch >= self.epoch and trainer.train_dataloader is not None:
                if trainer.train_dataloader.dataset.datasets.num_mask_samples != self.num_mask_samples:
                    trainer.train_dataloader.dataset.datasets.num_mask_samples = self.num_mask_samples
                    trainer.reset_train_dataloader()
                    print(f"'num_samples' was changed to {self.num_mask_samples}")

    class FreezeControl(pl.callbacks.Callback):
        def __init__(self, epoch, gradual=False):
            assert epoch > 0, "'epoch' of 'FreezeControl' must be larger than 0"
            self.epoch = epoch
            self.gradual = gradual

        def on_epoch_start(self, trainer, pl_module):
            if trainer.current_epoch >= self.epoch:
                if not self.gradual:
                    for key, value in trainer.model.module.module.backbone.named_parameters():
                        if not value.requires_grad:
                            value.requires_grad = True
                            print(f"'{key} was unfrozen")
                else:
                    gradual = [trainer.model.module.module.backbone.norm] \
                              + [block for block in trainer.model.module.module.backbone.blocks][::-1] \
                              + [trainer.model.module.module.backbone]

                    for idx, module in enumerate(gradual):
                        for key, value in module.named_parameters():
                            if not value.requires_grad:
                                value.requires_grad = True
                                print(f"'{key} was unfrozen")
                        if idx >= trainer.current_epoch - self.epoch:
                            return

    print('Setting')
    print(_config["num_nodes"])
    print(grad_steps)
    trainer = pl.Trainer(
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        strategy="ddp",
        benchmark=True,
        deterministic=False,
        gpus=gpus,
        max_epochs=_config["max_epochs"],
        callbacks=[
                      pl.callbacks.ModelCheckpoint(verbose=True, monitor="val/checkpoint_metric", mode="max",
                                                   save_last=True,
                                                   save_top_k=1),
                      pl.callbacks.LearningRateMonitor(logging_interval="step")] +
                  ([NumMaskSampleControl(epoch=_config["explainer_num_mask_samples_epoch"],
                                         num_mask_samples=_config["explainer_num_mask_samples"],
                                         paired_mask_samples=_config["explainer_paired_mask_samples"])] if _config[
                                                                                                               "stage"] == "explainer" else []) +
                  ([FreezeControl(epoch=_config["unfreeze_after"], gradual=_config["unfreeze_after_gradual"])] if
                   _config["unfreeze_after"] is not None else [])
        ,
        logger=wandb_logger,
        accumulate_grad_batches=grad_steps,
        gradient_clip_val=_config["grad_clipping"],
        log_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    # (6) Finally start training or testing.
    if _config["test_only"]:
        trainer.test(model_to_train, datamodule=datamodule)
    else:

        trainer.fit(model_to_train, datamodule=datamodule)
        best_model_path = trainer.checkpoint_callbacks[0].best_model_path
        model_to_train.load_from_checkpoint(best_model_path)
        trainer.test(model_to_train, datamodule=datamodule)
