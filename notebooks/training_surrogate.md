# Surrogate

## Pre-softmax

### ImageNette

```bash
python main.py with 'stage = "surrogate"' \
'wandb_project_name = "transformer_interpretability_project"' 'exp_name = "ImageNette_classifier_vit_base_patch16_224_lr1e-5_surrogate_lr1e-5_presoftmax"' \
env_chanwkim 'gpus_classifier=[3]' 'gpus_surrogate=[1]' \
dataset_ImageNette \
'classifier_backbone_type = "vit_base_patch16_224"' 'classifier_download_weight = False' 'classifier_load_path = "results/transformer_interpretability_project/2rq1issn/checkpoints/epoch=16-step=2498.ckpt"' \
'surrogate_mask_location = "pre-softmax"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/transformer_interpretability_project/2rq1issn/checkpoints/epoch=16-step=2498.ckpt"' \
training_hyperparameters_transformer 'checkpoint_metric = "loss"' 'learning_rate = 1e-5' 'max_epochs = 50'
```

### MURA

```bash
python main.py with 'stage = "surrogate"' \
'wandb_project_name = "transformer_interpretability_project"' 'exp_name = "MURA_classifier_vit_base_patch16_224_lr1e-5_surrogate_lr1e-5_presoftmax"' \
env_chanwkim 'gpus_classifier=[3]' 'gpus_surrogate=[2]' \
dataset_MURA \
'classifier_backbone_type = "vit_base_patch16_224"' 'classifier_download_weight = False' 'classifier_load_path = "results/transformer_interpretability_project/1u2xgwks/checkpoints/epoch=15-step=8255.ckpt"' \
'surrogate_mask_location = "pre-softmax"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/transformer_interpretability_project/1u2xgwks/checkpoints/epoch=15-step=8255.ckpt"' \
training_hyperparameters_transformer 'checkpoint_metric = "loss"' 'learning_rate = 1e-5' 'max_epochs = 50'
```

## Zero-input

### ImageNette

```bash
python main.py with 'stage = "surrogate"' \
'wandb_project_name = "transformer_interpretability_project"' 'exp_name = "ImageNette_classifier_vit_base_patch16_224_lr1e-5_surrogate_lr1e-5_zeroinput"' \
env_chanwkim 'gpus_classifier=[3]' 'gpus_surrogate=[0]' \
dataset_ImageNette \
'classifier_backbone_type = "vit_base_patch16_224"' 'classifier_download_weight = False' 'classifier_load_path = "results/transformer_interpretability_project/2rq1issn/checkpoints/epoch=16-step=2498.ckpt"' \
'surrogate_mask_location = "zero-input"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/transformer_interpretability_project/2rq1issn/checkpoints/epoch=16-step=2498.ckpt"' \
training_hyperparameters_transformer 'checkpoint_metric = "loss"' 'learning_rate = 1e-5' 'max_epochs = 50'
```

### MURA

```bash
python main.py with 'stage = "surrogate"' \
'wandb_project_name = "transformer_interpretability_project"' 'exp_name = "MURA_classifier_vit_base_patch16_224_lr1e-5_surrogate_lr1e-5_zeroinput"' \
env_chanwkim 'gpus_classifier=[0]' 'gpus_surrogate=[2]' \
dataset_MURA \
'classifier_backbone_type = "vit_base_patch16_224"' 'classifier_download_weight = False' 'classifier_load_path = "results/transformer_interpretability_project/1u2xgwks/checkpoints/epoch=15-step=8255.ckpt"' \
'surrogate_mask_location = "zero-input"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/transformer_interpretability_project/1u2xgwks/checkpoints/epoch=15-step=8255.ckpt"' \
training_hyperparameters_transformer 'checkpoint_metric = "loss"' 'learning_rate = 1e-5' 'max_epochs = 50'
```

## Post-softmax

### ImageNette

```bash
python main.py with 'stage = "surrogate"' \
'wandb_project_name = "transformer_interpretability_project"' 'exp_name = "ImageNette_classifier_vit_base_patch16_224_lr1e-5_surrogate_lr1e-5_postsoftmax"' \
env_chanwkim 'gpus_classifier=[0]' 'gpus_surrogate=[3]' \
dataset_ImageNette \
'classifier_backbone_type = "vit_base_patch16_224"' 'classifier_download_weight = False' 'classifier_load_path = "results/transformer_interpretability_project/2rq1issn/checkpoints/epoch=16-step=2498.ckpt"' \
'surrogate_mask_location = "post-softmax"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/transformer_interpretability_project/2rq1issn/checkpoints/epoch=16-step=2498.ckpt"' \
training_hyperparameters_transformer 'checkpoint_metric = "loss"' 'learning_rate = 1e-5' 'max_epochs = 50'
```

### MURA

```bash
python main.py with 'stage = "surrogate"' \
'wandb_project_name = "transformer_interpretability_project"' 'exp_name = "MURA_classifier_vit_base_patch16_224_lr1e-5_surrogate_lr1e-5_postsoftmax"' \
env_chanwkim 'gpus_classifier=[2]' 'gpus_surrogate=[3]' \
dataset_MURA \
'classifier_backbone_type = "vit_base_patch16_224"' 'classifier_download_weight = False' 'classifier_load_path = "results/transformer_interpretability_project/1u2xgwks/checkpoints/epoch=15-step=8255.ckpt"' \
'surrogate_mask_location = "post-softmax"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/transformer_interpretability_project/1u2xgwks/checkpoints/epoch=15-step=8255.ckpt"' \
training_hyperparameters_transformer 'checkpoint_metric = "loss"' 'learning_rate = 1e-5' 'max_epochs = 50'
```

## Zero-embedding

### ImageNette

```bash
python main.py with 'stage = "surrogate"' \
'wandb_project_name = "transformer_interpretability_project"' 'exp_name = "ImageNette_classifier_vit_base_patch16_224_lr1e-5_surrogate_lr1e-5_zeroembedding"' \
env_chanwkim 'gpus_classifier=[0]' 'gpus_surrogate=[1]' \
dataset_ImageNette \
'classifier_backbone_type = "vit_base_patch16_224"' 'classifier_download_weight = False' 'classifier_load_path = "results/transformer_interpretability_project/2rq1issn/checkpoints/epoch=16-step=2498.ckpt"' \
'surrogate_mask_location = "zero-embedding"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/transformer_interpretability_project/2rq1issn/checkpoints/epoch=16-step=2498.ckpt"' \
training_hyperparameters_transformer 'checkpoint_metric = "loss"' 'learning_rate = 1e-5' 'max_epochs = 50'
```

### MURA

```bash
python main.py with 'stage = "surrogate"' \
'wandb_project_name = "transformer_interpretability_project"' 'exp_name = "MURA_classifier_vit_base_patch16_224_lr1e-5_surrogate_lr1e-5_zeroembedding"' \
env_chanwkim 'gpus_classifier=[0]' 'gpus_surrogate=[3]' \
dataset_MURA \
'classifier_backbone_type = "vit_base_patch16_224"' 'classifier_download_weight = False' 'classifier_load_path = "results/transformer_interpretability_project/1u2xgwks/checkpoints/epoch=15-step=8255.ckpt"' \
'surrogate_mask_location = "zero-embedding"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/transformer_interpretability_project/1u2xgwks/checkpoints/epoch=15-step=8255.ckpt"' \
training_hyperparameters_transformer 'checkpoint_metric = "loss"' 'learning_rate = 1e-5' 'max_epochs = 50'
```

## Random-sampling

### ImageNette

```bash
python main.py with 'stage = "surrogate"' \
'wandb_project_name = "transformer_interpretability_project"' 'exp_name = "ImageNette_classifier_deit_base_patch16_224_lr1e-5_surrogate_lr1e-5_randomsampling"' \
env_chanwkim 'gpus_classifier=[1]' 'gpus_surrogate=[3]' \
dataset_ImageNette \
'classifier_backbone_type = "deit_base_patch16_224"' 'classifier_download_weight = False' 'classifier_load_path = "results/transformer_interpretability_project/3pysybi7/checkpoints/epoch=7-step=1175.ckpt"' \
'surrogate_mask_location = "random-sampling"' \
'surrogate_backbone_type = "deit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/transformer_interpretability_project/3pysybi7/checkpoints/epoch=7-step=1175.ckpt"' \
training_hyperparameters_transformer 'checkpoint_metric = "loss"' 'learning_rate = 1e-5' 'max_epochs = 50'
```

### MURA

```bash
python main.py with 'stage = "surrogate"' \
'wandb_project_name = "transformer_interpretability_project"' 'exp_name = "MURA_classifier_vit_base_patch16_224_lr1e-5_surrogate_lr1e-5_randomsampling"' \
env_chanwkim 'gpus_classifier=[1]' 'gpus_surrogate=[2]' \
dataset_MURA \
'classifier_backbone_type = "vit_base_patch16_224"' 'classifier_download_weight = False' 'classifier_load_path = "results/transformer_interpretability_project/1u2xgwks/checkpoints/epoch=15-step=8255.ckpt"' \
'surrogate_mask_location = "random-sampling"' \
'surrogate_backbone_type = "vit_base_patch16_224"' 'surrogate_download_weight = False' 'surrogate_load_path = "results/transformer_interpretability_project/1u2xgwks/checkpoints/epoch=15-step=8255.ckpt"' \
training_hyperparameters_transformer 'checkpoint_metric = "loss"' 'learning_rate = 1e-5' 'max_epochs = 50'
```


