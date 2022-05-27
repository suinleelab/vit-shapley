import platform

from sacred import Experiment

ex = Experiment("ViT_shapley")


@ex.config
def config():
    # Stage Setting
    stage = "classifier"  ## (classifier, surrogate, explainer, classifier_masked)

    # Wandb setting
    wandb_project_name = "default_wandb_project_name"
    exp_name = "default_exp_name"

    # Environment Setting
    log_dir = "results"
    per_gpu_batch_size = 64  # you should define this manually with per_gpu_batch_size=#
    gpus_classifier = None
    gpus_surrogate = None
    gpus_explainer = None
    num_nodes = 1
    num_workers = 4
    precision = 16

    # Seed Setting
    seed = 0

    # Dataset Setting
    datasets = "dataset_name"  # (ImageNette, MURA)
    dataset_location = "path"
    explanation_location_train = None
    explanation_mask_amount_train = None
    explanation_mask_ascending_train = None
    explanation_location_val = None
    explanation_mask_amount_val = None
    explanation_mask_ascending_val = None
    explanation_location_test = None
    explanation_mask_amount_test = None
    explanation_mask_ascending_test = None

    output_dim = None
    target_type = None  # ("binary", "multiclass", "multilabel")
    img_channels = 3
    test_data_split = "test"  # (test, val, train)
    transforms_train = {
        "Resize": {
            "apply": False,
            "height": 256,
            "width": 256
        },
        "RandomResizedCrop": {
            "apply": False,
            "size": 224,
            "scale": [
                0.8,
                1.2
            ]
        },
        "CenterCrop": {
            "apply": False,
            "height": 224,
            "width": 224
        },
        "VerticalFlip": {
            "apply": False,
            "p": 0.5
        },
        "HorizontalFlip": {
            "apply": False,
            "p": 0.5
        },
        "ColorJitter": {
            "apply": False,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.1,
            "hue": 0.1,
            "p": 0.8
        },
        "Normalize": {
            "apply": True
        },
    }  # the final resize transform should be modified to fit the input dim of the model being trained.
    transforms_val = {
        "Resize": {
            "apply": False,
            "height": 256,
            "width": 256
        },
        "CenterCrop": {
            "apply": False,
            "height": 224,
            "width": 224
        },
        "Normalize": {
            "apply": False
        },

    }  # the final resize transform should be modified to fit the input dim of the model being trained.
    transforms_test = {
        "Resize": {
            "apply": False,
            "height": 256,
            "width": 256
        },
        "CenterCrop": {
            "apply": False,
            "height": 224,
            "width": 224
        },
        "Normalize": {
            "apply": False
        },

    }  # the final resize transform should be modified to fit the input dim of the model being trained.

    # Classifier Model Setting
    classifier_masked_mask_location = None
    classifier_backbone_type = None
    classifier_enable_pos_embed = True
    classifier_download_weight = True
    classifier_load_path = None

    # Surrogate Model Setting
    surrogate_mask_location = None
    surrogate_backbone_type = None  # "deit_small_patch16_224"
    surrogate_download_weight = True
    surrogate_load_path = None

    # Explainer Model Setting
    explainer_num_mask_samples_epoch = 0
    explainer_num_mask_samples = 32
    explainer_paired_mask_samples = True
    explainer_normalization = None
    explainer_normalization_class = None
    explainer_link = None
    explainer_residual = []
    explainer_activation = None

    explainer_efficiency_lambda = 0
    explainer_efficiency_class_lambda = 0
    explainer_freeze_backbone = True

    explainer_head_num_attention_blocks = 1
    explainer_head_num_mlp_layers = 3
    explainer_head_mlp_layer_ratio = 4
    explainer_head_include_cls = True
    explainer_norm = True

    explainer_backbone_type = None
    explainer_download_weight = True
    explainer_load_path = None

    # Dataloader Setting
    batch_size = 64  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Training Setting
    checkpoint_metric = None
    optim_type = None
    learning_rate = None
    weight_decay = None
    loss_weight = None
    decay_power = None
    grad_clipping = None
    max_epochs = None
    warmup_steps = None

    unfreeze_after = None
    unfreeze_after_gradual = False

    val_check_interval = 1.0
    test_only = False
    resume_from = None
    fast_dev_run = False


# Named configs for "environment"
@ex.named_config
def env_chanwkim():
    log_dir = "results"  # f'/homes/gws/chanwkim/network_drive/{platform.node()}/'
    per_gpu_batch_size = 64  # you should define this manually with per_gpu_batch_size=#
    gpus_classifier = [0]
    gpus_surrogate = [0]
    num_nodes = 1
    num_workers = 4
    precision = 16


# Named configs for "Train"
@ex.named_config
def training_hyperparameters_transformer():
    optim_type = "Adamw"
    learning_rate = 1e-4
    weight_decay = 1e-5
    decay_power = "cosine"
    grad_clipping = 1.0
    max_epochs = 25
    warmup_steps = 500

@ex.named_config
def dataset_ImageNette():
    datasets = "ImageNette"
    dataset_location = None
    output_dim = 10
    target_type = "multiclass"
    img_channels = 3
    test_data_split = "test"
    transforms_train = {
        "Resize": {
            "apply": True,
            "height": 256,
            "width": 256
        },
        "RandomResizedCrop": {
            "apply": True,
            "size": 224,
            "scale": [
                0.8,
                1.2
            ]
        },
        "VerticalFlip": {
            "apply": True,
            "p": 0.5
        },
        "HorizontalFlip": {
            "apply": True,
            "p": 0.5
        },
        "ColorJitter": {
            "apply": True,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.1,
            "hue": 0.1,
            "p": 0.8
        },
        "Normalize": {
            "apply": True
        },
    }
    transforms_val = {
        "Resize": {
            "apply": True,
            "height": 256,
            "width": 256
        },
        "CenterCrop": {
            "apply": True,
            "height": 224,
            "width": 224
        },
        "Normalize": {
            "apply": True
        },

    }
    transforms_test = {
        "Resize": {
            "apply": True,
            "height": 256,
            "width": 256
        },
        "CenterCrop": {
            "apply": True,
            "height": 224,
            "width": 224
        },
        "Normalize": {
            "apply": True
        },

    }


@ex.named_config
def dataset_ImageNette_ROAR():
    datasets = "ImageNette"
    log_dir = f'/homes/gws/chanwkim/network_drive/{platform.node()}/'
    dataset_location = None
    output_dim = 10
    target_type = "multiclass"
    img_channels = 3
    test_data_split = "test"
    transforms_train = {
        "Resize": {
            "apply": True,
            "height": 256,
            "width": 256
        },
        "CenterCrop": {
            "apply": True,
            "height": 224,
            "width": 224
        },
        "Normalize": {
            "apply": True
        },

    }
    transforms_val = {
        "Resize": {
            "apply": True,
            "height": 256,
            "width": 256
        },
        "CenterCrop": {
            "apply": True,
            "height": 224,
            "width": 224
        },
        "Normalize": {
            "apply": True
        },

    }
    transforms_test = {
        "Resize": {
            "apply": True,
            "height": 256,
            "width": 256
        },
        "CenterCrop": {
            "apply": True,
            "height": 224,
            "width": 224
        },
        "Normalize": {
            "apply": True
        },

    }


@ex.named_config
def dataset_MURA():
    datasets = "MURA"
    dataset_location = f'/homes/gws/chanwkim/network_drive/{platform.node()}/MURA-v1.1'
    output_dim = 1
    target_type = "binary"
    img_channels = 3
    test_data_split = "test"
    transforms_train = {
        "Resize": {
            "apply": True,
            "height": 256,
            "width": 256
        },
        "RandomResizedCrop": {
            "apply": True,
            "size": 224,
            "scale": [
                0.8,
                1.2
            ]
        },
        "VerticalFlip": {
            "apply": True,
            "p": 0.5
        },
        "HorizontalFlip": {
            "apply": True,
            "p": 0.5
        },
        "ColorJitter": {
            "apply": True,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.1,
            "hue": 0.1,
            "p": 0.8
        },
        "Normalize": {
            "apply": True
        },
    }
    transforms_val = {
        "Resize": {
            "apply": True,
            "height": 256,
            "width": 256
        },
        "CenterCrop": {
            "apply": True,
            "height": 224,
            "width": 224
        },
        "Normalize": {
            "apply": True
        },

    }
    transforms_test = {
        "Resize": {
            "apply": True,
            "height": 256,
            "width": 256
        },
        "CenterCrop": {
            "apply": True,
            "height": 224,
            "width": 224
        },
        "Normalize": {
            "apply": True
        },

    }


@ex.named_config
def dataset_MURA_ROAR():
    datasets = "MURA"
    dataset_location = f'/homes/gws/chanwkim/network_drive/{platform.node()}/MURA-v1.1'
    log_dir = f'/homes/gws/chanwkim/network_drive/{platform.node()}/'
    output_dim = 1
    target_type = "binary"
    img_channels = 3
    test_data_split = "test"
    transforms_train = {
        "Resize": {
            "apply": True,
            "height": 256,
            "width": 256
        },
        "CenterCrop": {
            "apply": True,
            "height": 224,
            "width": 224
        },
        "Normalize": {
            "apply": True
        },

    }
    transforms_val = {
        "Resize": {
            "apply": True,
            "height": 256,
            "width": 256
        },
        "CenterCrop": {
            "apply": True,
            "height": 224,
            "width": 224
        },
        "Normalize": {
            "apply": True
        },

    }
    transforms_test = {
        "Resize": {
            "apply": True,
            "height": 256,
            "width": 256
        },
        "CenterCrop": {
            "apply": True,
            "height": 224,
            "width": 224
        },
        "Normalize": {
            "apply": True
        },

    }
