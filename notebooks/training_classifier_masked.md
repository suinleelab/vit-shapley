# ClassifierMasked

## Masking method - Pre-softmax

### ImageNette

```bash
# train classifier
python main.py with 'stage = "classifier_masked"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "ImageNette_classifier_masked_vit_small_patch16_224_1e-5_train_presoftmax"' \
env_chanwkim 'gpus_classifier=[0]' \
dataset_ImageNette \
'classifier_backbone_type = "vit_small_patch16_224"' 'classifier_download_weight = True' 'classifier_load_path = None' \
'classifier_masked_mask_location = "pre-softmax"' \
training_hyperparameters_transformer 'checkpoint_metric = "accuracy"' 'learning_rate = 1e-5'
```

```bash
# train classifier
python main.py with 'stage = "classifier_masked"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "ImageNette_classifier_masked_vit_base_patch16_224_1e-5_train_presoftmax"' \
env_chanwkim 'gpus_classifier=[0]' \
dataset_ImageNette \
'classifier_backbone_type = "vit_base_patch16_224"' 'classifier_download_weight = True' 'classifier_load_path = None' \
'classifier_masked_mask_location = "pre-softmax"' \
training_hyperparameters_transformer 'checkpoint_metric = "accuracy"' 'learning_rate = 1e-5'
```

### MURA

```bash
# train classifier
python main.py with 'stage = "classifier_masked"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "MURA_classifier_masked_vit_base_patch16_224_1e-5_train_presoftmax"' \
env_chanwkim 'gpus_classifier=[0]' \
dataset_MURA \
'classifier_backbone_type = "vit_base_patch16_224"' 'classifier_download_weight = True' 'classifier_load_path = None' \
'classifier_masked_mask_location = "pre-softmax"' \
training_hyperparameters_transformer 'loss_weight = [21935, 14873]' 'checkpoint_metric = "CohenKappa"' 'learning_rate = 1e-5'
```

## Masking method - Zero-input

### ImageNette

```bash
# train classifier
python main.py with 'stage = "classifier_masked"' \
'wandb_project_name = "wandb_transformer_interpretability_project"' 'exp_name = "ImageNette_classifier_masked_vit_base_patch16_224_1e-5_train_zeroinput"' \
env_chanwkim 'gpus_classifier=[7]' \
dataset_ImageNette \
'classifier_backbone_type = "vit_base_patch16_224"' 'classifier_download_weight = True' 'classifier_load_path = None' \
'classifier_masked_mask_location = "zero-input"' \
training_hyperparameters_transformer 'checkpoint_metric = "accuracy"' 'learning_rate = 1e-5'
```