# ViT-Shapley

Shapley values are a theoretically grounded model explanation approach, but their exponential computational cost makes them difficult to use with large deep learning models. This package implements **ViT-Shapley**, an approach that makes Shapley values practical for vision transformer (ViT) models. The key idea is to learn an *amortized explainer model* that generates explanations in a single forward pass.

The high-level workflow for using ViT-Shapley is the following:

1. Obtain your initial ViT model
2. If your model was not trained to acommodate held-out image patches, fine-tune it with random masking
3. Train an explainer model using ViT-Shapley's custom loss function (often by fine-tuning parameters of the original ViT)

Please see our paper [here](https://arxiv.org/abs/2206.05282?context=cs.LG) for more details, as well as the work that ViT-Shapley builds on ([KernelSHAP](https://arxiv.org/abs/1705.07874), [FastSHAP](https://openreview.net/forum?id=Zq2G_VTV53T)).

## Installation

```bash
git clone https://github.com/chanwkimlab/vit-shapley.git
cd vit-shapley
pip install -r requirements.txt
```

## Training

Commands for training and testing the models are available in the files under `notebooks` directory.

* notebooks/training_classifier.md
* notebooks/training_surrogate.md
* notebooks/training_explainer.md
* notebooks/training_classifier_masked.md

## Benchmarking

1. Run `notebooks/2_1_benchmarking.ipynb` to obtain results.
2. Run `notebooks/2_2_ROAR.ipynb` to run retraining-based ROAR benchmarking.
3. Run `notebooks/3_plotting.ipynb` to plot the results.

## Datasets

- [ImageNette](https://github.com/fastai/imagenette)
- [MURA](https://stanfordmlgroup.github.io/competitions/mura/)

<!-- ## Download Pretrained Models

Download pretrained models from [here](). -->

## Citation

If you use any part of this code and pretrained weights for your own purpose, please cite
our [paper](https://arxiv.org/abs/2206.05282).

## Contact

- [Ian Covert](https://iancovert.com) (Paul G. Allen School of Computer Science and Engineering @ University of
  Washington)
- [Chanwoo Kim](https://chanwoo.kim) (Paul G. Allen School of Computer Science and Engineering @ University of
  Washington)
- [Su-In Lee](https://suinlee.cs.washington.edu/) (Paul G. Allen School of Computer Science and Engineering @ University
  of Washington)
