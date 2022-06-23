# ViT-Shapley (Shapley value estimation for Vision Transformers)

## Installation

```bash
git clone https://github.com/chanwkimlab/vit-shapley.git
cd vit-shapley
pip install -r requirements.txt
```

## Training

Command lines for training and testing the models are available in the files under `notebooks` directory.

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

## Download Pretrained Models

Download pretrained models from [here]().

## Citation

If you use any part of this code and pretrained weights for your own purpose, please cite our [paper]().

## Contact

- [Ian Covert](https://iancovert.com) (Paul G. Allen School of Computer Science and Engineering @ University of
  Washington)
- [Chanwoo Kim](https://chanwoo.kim) (Paul G. Allen School of Computer Science and Engineering @ University of
  Washington)
- [Su-In Lee](https://suinlee.cs.washington.edu/) (Paul G. Allen School of Computer Science and Engineering @ University
  of Washington)
