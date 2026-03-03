"""MURA (Musculoskeletal Radiographs) dataset loader utilities.

MURA is a large dataset of bone X-rays containing 40,561 images across 7 body
parts (elbow, finger, forearm, hand, humerus, shoulder, wrist).  Each study is
labelled as ``negative`` (normal) or ``positive`` (abnormal).

The dataset must be obtained from the Stanford AIMI Center
(https://stanfordaimi.azurewebsites.net/datasets/4f1b406e-3a7b-43ac-8bb7-7d3434f6070e)
and placed at ``<root>/MURA-v1.1/``.

Splits replicate the original ViT-Shapley reference:

* **train / val** — read ``train_image_paths.csv``, extract unique patients,
  split 90/10 by patient using ``np.random.RandomState(44)`` (replicates
  ``sklearn.model_selection.train_test_split(random_state=44, test_size=0.1)``).
* **test** — read ``valid_image_paths.csv`` as-is.

Reference: https://arxiv.org/abs/2206.05282
"""

import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch.utils.data
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms import InterpolationMode

# ImageNet channel mean / std (MURA uses ImageNet-pretrained models)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_MURA_DIRNAME = "MURA-v1.1"

MURA_CLASSES = ["negative", "positive"]
MURA_CLASS_TO_IDX = {cls: i for i, cls in enumerate(MURA_CLASSES)}


def get_mura_transforms(
    train: bool = True,
    image_size: int = 224,
) -> T.Compose:
    """Return augmentation transforms matching the ImageNette/Pet pipeline.

    Training:
        Resize(256x256) -> RandomResizedCrop(scale=(0.8, 1.2)) ->
        RandomVerticalFlip(0.5) -> RandomHorizontalFlip(0.5) ->
        RandomApply(ColorJitter(br=0.2,co=0.2,sat=0.1,hue=0.1), p=0.8) ->
        ToTensor -> Normalize

    Validation:
        Resize(256x256) -> CenterCrop -> ToTensor -> Normalize

    Args:
        train: If ``True`` return the training transform, else the val transform.
        image_size: Spatial resolution fed to the model (default 224 for ViT).

    Returns:
        A :class:`torchvision.transforms.Compose` object.
    """
    normalize = T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    if train:
        return T.Compose(
            [
                T.Resize((256, 256)),
                T.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.2),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                T.RandomVerticalFlip(p=0.5),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(
                    [
                        T.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                T.ToTensor(),
                normalize,
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize((256, 256)),
                T.CenterCrop(image_size),
                T.ToTensor(),
                normalize,
            ]
        )


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class MURADataset(torch.utils.data.Dataset):
    """MURA dataset built from image-path CSV files.

    Attributes:
        classes: ``["negative", "positive"]``.
        class_to_idx: Mapping from class name to integer label.
    """

    def __init__(self, data_list: list[dict], transform=None):
        """
        Args:
            data_list: List of ``{"img_path": str, "label": int}`` dicts.
            transform: torchvision transform applied to each PIL image.
        """
        self._data = data_list
        self.transform = transform
        self.classes = list(MURA_CLASSES)
        self.class_to_idx = dict(MURA_CLASS_TO_IDX)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        item = self._data[idx]
        img = Image.open(item["img_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, item["label"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_paths_csv(csv_path: Path) -> pd.DataFrame:
    """Read a MURA image-paths CSV (no header, single column of paths)."""
    return pd.read_csv(csv_path, header=None, names=["path"])


def _extract_label(path: str) -> str:
    """Extract the label string from a MURA image path.

    E.g. ``"MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/image1.png"``
    → ``"positive"``.
    """
    # The study folder is the 5th component (index 4): "study1_positive"
    study_folder = path.split("/")[4]
    return study_folder.split("_")[1]


def _extract_patient(path: str) -> str:
    """Extract the patient ID from a MURA image path."""
    return path.split("/")[3]


def _build_data_list(
    df: pd.DataFrame,
    dataset_dir: Path,
) -> list[dict]:
    """Convert a DataFrame of CSV paths into a list of {img_path, label} dicts."""
    data_list = []
    for path_str in df["path"]:
        label_str = _extract_label(path_str)
        label = MURA_CLASS_TO_IDX[label_str]
        # Strip "MURA-v1.1/" prefix, prepend actual dataset directory
        rel_path = path_str.replace("MURA-v1.1/", "", 1)
        img_path = str(dataset_dir / rel_path)
        data_list.append({"img_path": img_path, "label": label})
    return data_list


def _build_mura_split(
    dataset_dir: Path,
    split: str,
) -> list[dict]:
    """Build the data list for the requested split.

    Replicates the reference implementation's split logic:

    * train/val: read ``train_image_paths.csv``, extract unique patients,
      split using ``np.random.RandomState(44)`` with
      ``n_val = math.ceil(n_patients * 0.1)`` (replicates sklearn's
      ``train_test_split(random_state=44, test_size=0.1)``).
    * test: read ``valid_image_paths.csv`` as-is.

    Each resulting list is shuffled with ``random.Random(42)``.
    """
    if split in ("train", "val"):
        csv_path = dataset_dir / "train_image_paths.csv"
        df = _read_paths_csv(csv_path)

        # Patient-based split (no patient leakage)
        df["patient"] = df["path"].map(_extract_patient)
        unique_patients = df["patient"].unique()
        n = len(unique_patients)

        # Replicate sklearn: ceil(n * test_size) for val, rest for train
        n_val = math.ceil(n * 0.1)

        rng = np.random.RandomState(44)
        perm = rng.permutation(n)

        # sklearn: ind_test = perm[:n_test], ind_train = perm[n_test:]
        val_patients = set(unique_patients[perm[:n_val]])
        train_patients = set(unique_patients[perm[n_val:]])

        if split == "train":
            df = df[df["patient"].isin(train_patients)]
        else:
            df = df[df["patient"].isin(val_patients)]

    else:  # test
        csv_path = dataset_dir / "valid_image_paths.csv"
        df = _read_paths_csv(csv_path)

    data_list = _build_data_list(df, dataset_dir)

    # Match reference: random.Random(42).shuffle(data_list)
    random.Random(42).shuffle(data_list)

    return data_list


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_mura_dataset(
    root: str | os.PathLike,
    split: str = "train",
    image_size: int = 224,
    transform=None,
    download: bool = False,
) -> MURADataset:
    """Return a dataset for the requested split of MURA.

    Args:
        root: Root directory that *contains* the ``MURA-v1.1`` sub-directory.
        split: ``"train"``, ``"val"``, or ``"test"``.
        image_size: Passed to :func:`get_mura_transforms` when *transform*
                    is ``None``.
        transform: Override the default transforms (useful for testing).
        download: Ignored (MURA requires a Stanford AIMI license).  Raises
                  ``FileNotFoundError`` if the dataset is missing regardless.

    Raises:
        AssertionError: If *split* is not one of ``"train"``, ``"val"``,
                        ``"test"``.
        FileNotFoundError: If the dataset directory does not exist.
    """
    assert split in ("train", "val", "test"), (
        f"split must be 'train', 'val', or 'test', got '{split}'"
    )

    root = Path(root)
    dataset_dir = root / _MURA_DIRNAME

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"MURA directory not found: {dataset_dir}. "
            "MURA requires a Stanford AIMI license and cannot be downloaded "
            "automatically. Please obtain the dataset from "
            "https://stanfordaimi.azurewebsites.net/datasets/"
            "4f1b406e-3a7b-43ac-8bb7-7d3434f6070e "
            "and extract it to the root directory."
        )

    data_list = _build_mura_split(dataset_dir, split)

    if transform is None:
        transform = get_mura_transforms(
            train=(split == "train"), image_size=image_size
        )

    return MURADataset(data_list, transform)
