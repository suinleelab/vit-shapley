# Classifier

## ImageNette

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
'wandb_project_name = "transformer_interpretability_project"' 'exp_name = "MURA_classifier_vit_base_patch16_224_1e-5_train"' \
env_chanwkim 'gpus_classifier=[2]' \
dataset_MURA \
'classifier_backbone_type = "vit_base_patch16_224"' 'classifier_download_weight = True' 'classifier_load_path = None' \
training_hyperparameters_transformer 'loss_weight = [21935, 14873]' 'checkpoint_metric = "CohenKappa"' 'learning_rate = 1e-5'
```

* When you encounter an error, setting `'fast_dev_run = 1'` will be useful for debugging. The training will be done for
  only one batch.