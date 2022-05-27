# Shapley value estimation for Vision Transformers

## Installation

```bash
git clone https://github.com/chanwkimlab/vit-shapley.git
cd vit-shapley
pip install -r requirements.txt
```

## Model Training

Command lines for training and testing the models are available under `notebooks` directory.

* notebooks/training_classifier.md
* notebooks/training_surrogate.md
* notebooks/training_explainer.md
* notebooks/training_classifier_masked.md

## Benchmarking

1. Run `notebooks/2_1_benchmarking.ipynb` to obtain results.
2. Run `notebooks/3_plotting.ipynb` to plot the results.

## Datasets

- ImageNette
- MURA

## Download Pretrained Models

download pretrained models from [here]().

## Citation

If you use any part of this code and pretrained weights for your own purpose, please cite our [paper]().

@InProceedings{xxx,xxx, title = {Learning to Estimate Shapley Values with Vision Transformers}, author = {Covert, Ian
and Kim, Chanwoo and Lee, Su-In}, booktitle = {Proceedings of the 38th International Conference on Machine Learning},
year = {2022}, }

## Contact

- [Ian Covert](https://iancovert.com) (Paul G. Allen School of Computer Science and Engineering @ University of
  Washington)
- [Chanwoo Kim](https://chanwoo.kim) (Paul G. Allen School of Computer Science and Engineering @ University of
  Washington)
- [Su-In Lee](https://suinlee.cs.washington.edu/) (Paul G. Allen School of Computer Science and Engineering @ University
  of Washington)
