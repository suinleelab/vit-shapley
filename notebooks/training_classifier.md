# Classifier

## ImageNette

```bash
# train classifier
python main.py with 'stage = "classifier"' \
'wandb_project_name = "transformer_interpretability_project"' 'exp_name = "ImageNette_classifier_vit_small_patch16_224_1e-5_train"' \
env_chanwkim 'gpus_classifier=[6]' \
dataset_ImageNette \
'classifier_backbone_type = "vit_small_patch16_224"' 'classifier_download_weight = True' 'classifier_load_path = None' \
training_hyperparameters_transformer 'checkpoint_metric = "accuracy"' 'learning_rate = 1e-5'
```

```bash
# train classifier
python main.py with 'stage = "classifier"' \
'wandb_project_name = "transformer_interpretability_project"' 'exp_name = "ImageNette_classifier_vit_base_patch16_224_1e-5_train"' \
env_chanwkim 'gpus_classifier=[3]' \
dataset_ImageNette \
'classifier_backbone_type = "vit_base_patch16_224"' 'classifier_download_weight = True' 'classifier_load_path = None' \
training_hyperparameters_transformer 'checkpoint_metric = "accuracy"' 'learning_rate = 1e-5'
```

## MURA

```bash
# train classifier
python main.py with 'stage = "classifier"' \
'wandb_project_name = "transformer_interpretability_project"' 'exp_name = "MURA_classifier_vit_small_patch16_224_1e-5_train"' \
env_chanwkim 'gpus_classifier=[4]' \
dataset_MURA \
'classifier_backbone_type = "vit_small_patch16_224"' 'classifier_download_weight = True' 'classifier_load_path = None' \
training_hyperparameters_transformer 'loss_weight = [21935, 14873]' 'checkpoint_metric = "CohenKappa"' 'learning_rate = 1e-5'
```

```bash
# train classifier
python main.py with 'stage = "classifier"' \
'wandb_project_name = "transformer_interpretability_project"' 'exp_name = "MURA_classifier_vit_base_patch16_224_1e-5_train"' \
env_chanwkim 'gpus_classifier=[2]' \
dataset_MURA \
'classifier_backbone_type = "vit_base_patch16_224"' 'classifier_download_weight = True' 'classifier_load_path = None' \
training_hyperparameters_transformer 'loss_weight = [21935, 14873]' 'checkpoint_metric = "CohenKappa"' 'learning_rate = 1e-5'
```

```bash
python main.py with 'stage = "classifier"' \
'wandb_project_name = "transformer_interpretability_project"' 'exp_name = "MURA_classifier_vit_base_patch16_224_1e-5_test_classifierweight"' \
env_chanwkim 'gpus_classifier=[2]' \
dataset_MURA \
'classifier_backbone_type = "vit_base_patch16_224"' 'classifier_download_weight = False' 'classifier_load_path = "results/transformer_interpretability_project/1u2xgwks/checkpoints/epoch=15-step=8255.ckpt"' \
'test_only = True' 'checkpoint_metric = "CohenKappa"'
```

```bash
python main.py with 'stage = "classifier"' \
'wandb_project_name = "transformer_interpretability_project"' 'exp_name = "MURA_classifier_vit_base_patch16_224_1e-5_test_surrogateweight"' \
env_chanwkim 'gpus_classifier=[2]' \
dataset_MURA \
'classifier_backbone_type = "vit_base_patch16_224"' 'classifier_download_weight = False' 'classifier_load_path = "results/transformer_interpretability_project/22ompjqu/checkpoints/epoch=47-step=24767.ckpt"' \
'test_only = True' 'checkpoint_metric = "CohenKappa"'
```

* When you encounter an error, setting `'fast_dev_run = 1'` will be useful for debugging. The training will be done for
  only one batch.