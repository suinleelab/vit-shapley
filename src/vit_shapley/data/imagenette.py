"""ImageNette dataset download and loader utilities.

ImageNette is a 10-class subset of ImageNet curated by fast.ai.
We use the 160px variant for faster downloads.

Fast.ai S3 URL: https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz

When ``noisy_imagenette.csv`` is present in the dataset directory,
:func:`get_imagenette_dataset` uses a CSV-based loader that preserves the
exact class-to-index ordering from the original ViT-Shapley paper and
replicates ``sklearn``'s val/test split exactly.  When the CSV is absent
(e.g. in unit tests using a minimal fake directory), it falls back to an
``ImageFolder``-based loader.
"""

import math
import os
import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch.utils.data
import torchvision.transforms as T
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

_IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
_IMAGENETTE_DIRNAME = "imagenette2-160"

# ImageNet channel mean / std (ImageNette is a subset of ImageNet)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Reference class ordering from the original ViT-Shapley paper.
# The paper reads labels from noisy_imagenette.csv's 'noisy_labels_0' column
# and maps them in this specific (non-alphabetical) order.
IMAGENETTE_CLASSES = [
    "n02979186",  # 0  cassette_player
    "n03417042",  # 1  garbage_truck
    "n01440764",  # 2  tench
    "n02102040",  # 3  English_springer
    "n03028079",  # 4  church
    "n03888257",  # 5  parachute
    "n03394916",  # 6  French_horn
    "n03000684",  # 7  chain_saw
    "n03445777",  # 8  golf_ball
    "n03425413",  # 9  gas_pump
]
IMAGENETTE_CLASS_TO_IDX = {cls: i for i, cls in enumerate(IMAGENETTE_CLASSES)}

# Human-readable display names corresponding to each class index.
# Used for visualization (heatmap headers, plot labels, etc.).
IMAGENETTE_CLASS_DISPLAY_NAMES = [
    "cassette player",   # 0
    "garbage truck",     # 1
    "tench",             # 2
    "English springer",  # 3
    "church",            # 4
    "parachute",         # 5
    "French horn",       # 6
    "chain saw",         # 7
    "golf ball",         # 8
    "gas pump",          # 9
]


def download_imagenette(root: str | os.PathLike) -> Path:
    """Download and extract ImageNette-160 into *root* if not already present.

    Args:
        root: Directory where the dataset will be stored.
              After extraction the data lives at ``<root>/imagenette2-160/``.

    Returns:
        Path to the extracted ``imagenette2-160`` directory.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dataset_dir = root / _IMAGENETTE_DIRNAME

    if dataset_dir.exists():
        return dataset_dir

    tgz_path = root / "imagenette2-160.tgz"
    if not tgz_path.exists():
        print(f"Downloading ImageNette-160 from {_IMAGENETTE_URL} ...")
        urllib.request.urlretrieve(_IMAGENETTE_URL, tgz_path)
        print("Download complete.")

    print(f"Extracting {tgz_path} ...")
    with tarfile.open(tgz_path, "r:gz") as tf:
        tf.extractall(root)
    print("Extraction complete.")

    return dataset_dir


def get_imagenette_transforms(
    train: bool = True,
    image_size: int = 224,
) -> T.Compose:
    """Return augmentation transforms matching the original ViT-Shapley paper.

    Training:
        Resize(256×256) → RandomResizedCrop(scale=(0.8, 1.2)) →
        RandomVerticalFlip(0.5) → RandomHorizontalFlip(0.5) →
        RandomApply(ColorJitter(br=0.2,co=0.2,sat=0.1,hue=0.1), p=0.8) →
        ToTensor → Normalize

    Validation:
        Resize(256×256) → CenterCrop → ToTensor → Normalize

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
# CSV-based dataset (used when noisy_imagenette.csv is present)
# ---------------------------------------------------------------------------


class ImageNetteCSVDataset(torch.utils.data.Dataset):
    """ImageNette dataset built from ``noisy_imagenette.csv``.

    Preserves the reference class-to-index ordering (``IMAGENETTE_CLASSES``),
    which differs from the alphabetical ordering ``ImageFolder`` would assign.

    Attributes:
        classes: List of human-readable class names in reference order.
        class_to_idx: Mapping from display name to integer label.
    """

    def __init__(self, df: pd.DataFrame, dataset_dir: Path, transform=None):
        """
        Args:
            df: DataFrame with columns ``path`` (relative) and ``label`` (int).
            dataset_dir: Root of the imagenette2-160 directory (for resolving paths).
            transform: torchvision transform applied to each PIL image.
        """
        self._df = df.reset_index(drop=True)
        self._dataset_dir = dataset_dir
        self.transform = transform
        self.classes = list(IMAGENETTE_CLASS_DISPLAY_NAMES)
        self.class_to_idx = {name: i for i, name in enumerate(IMAGENETTE_CLASS_DISPLAY_NAMES)}

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int):
        row = self._df.iloc[idx]
        img = Image.open(self._dataset_dir / row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"])


def _build_csv_dataset(
    dataset_dir: Path,
    split: str,
    transform,
) -> ImageNetteCSVDataset:
    """Build an :class:`ImageNetteCSVDataset` from ``noisy_imagenette.csv``.

    Replicates the original paper's data pipeline:

    * Class labels come from ``noisy_labels_0`` (0 % noise = clean labels).
    * Val/test split uses the equivalent of
      ``sklearn.model_selection.train_test_split(data.index, random_state=44,
      test_size=0.5)``: sklearn uses ``ceil(n * 0.5)`` for the test portion.
    * Each resulting DataFrame is shuffled with ``random.Random(42)``,
      matching the reference's ``random.Random(42).shuffle(data_list)`` call.
    """
    csv_path = dataset_dir / "noisy_imagenette.csv"
    df = pd.read_csv(csv_path)

    # Map synset IDs to integer labels using the reference ordering
    df["label"] = df["noisy_labels_0"].map(IMAGENETTE_CLASS_TO_IDX)

    if split == "train":
        subset = df[~df["is_valid"]].copy()
    else:
        valid = df[df["is_valid"]].copy()
        n = len(valid)

        # sklearn uses ceil(n * test_size) for a float test_size
        n_test = math.ceil(n * 0.5)

        # Replicate sklearn's ShuffleSplit: permute local positions,
        # ind_test = perm[:n_test], ind_train = perm[n_test:]
        rng = np.random.RandomState(44)
        perm = rng.permutation(n)
        local_idx = perm[n_test:] if split == "val" else perm[:n_test]
        subset = valid.iloc[local_idx].copy()

    # Match the reference's random.Random(42).shuffle(data_list)
    row_order = list(range(len(subset)))
    random.Random(42).shuffle(row_order)
    subset = subset.iloc[row_order].reset_index(drop=True)

    return ImageNetteCSVDataset(subset[["path", "label"]], dataset_dir, transform)


# ---------------------------------------------------------------------------
# ImageFolder-based fallback (used when noisy_imagenette.csv is absent)
# ---------------------------------------------------------------------------


class _LabelledSubset(torch.utils.data.Subset):
    """Subset that exposes ``.classes`` and ``.class_to_idx`` from the parent dataset."""

    def __init__(self, dataset: ImageFolder, indices):
        super().__init__(dataset, indices)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx


def _split_val(val_dataset: ImageFolder, split: str) -> _LabelledSubset:
    """Split the full val ImageFolder into val (~50%) and test (~50%).

    Uses a deterministic permutation (RandomState seed 44) to replicate
    ``sklearn.model_selection.train_test_split(random_state=44, test_size=0.5)``.
    sklearn uses ``ceil(n * 0.5)`` for the test portion; the remainder is val.
    """
    n = len(val_dataset)
    n_test = math.ceil(n * 0.5)
    n_val = n - n_test

    rng = np.random.RandomState(44)
    perm = rng.permutation(n)

    # sklearn: test = perm[:n_test], train (→ our val) = perm[n_test:]
    if split == "val":
        indices = sorted(perm[n_test : n_test + n_val].tolist())
    else:  # "test"
        indices = sorted(perm[:n_test].tolist())

    return _LabelledSubset(val_dataset, indices)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_imagenette_dataset(
    root: str | os.PathLike,
    split: str = "train",
    image_size: int = 224,
    transform=None,
    download: bool = False,
) -> "ImageNetteCSVDataset | ImageFolder | _LabelledSubset":
    """Return a dataset for the requested split.

    When ``noisy_imagenette.csv`` is present, uses the CSV-based loader
    (:class:`ImageNetteCSVDataset`) which preserves the reference class
    ordering and exactly replicates the paper's val/test split.  Otherwise
    falls back to an ``ImageFolder``-based loader (used by unit tests that
    create a minimal fake directory structure).

    Args:
        root: Root directory that *contains* (or will contain) the
              ``imagenette2-160`` sub-directory.
        split: ``"train"``, ``"val"``, or ``"test"``.
        image_size: Passed to :func:`get_imagenette_transforms` when
                    *transform* is ``None``.
        transform: Override the default transforms (useful for testing).
        download: If ``True`` download the dataset if absent.

    Raises:
        AssertionError: If *split* is not one of ``"train"``, ``"val"``,
                        ``"test"``.
        FileNotFoundError: If the dataset directory does not exist and
                           *download* is ``False``.
    """
    assert split in ("train", "val", "test"), (
        f"split must be 'train', 'val', or 'test', got '{split}'"
    )

    root = Path(root)
    dataset_dir = root / _IMAGENETTE_DIRNAME

    if not dataset_dir.exists():
        if download:
            download_imagenette(root)
        else:
            raise FileNotFoundError(
                f"ImageNette directory not found: {dataset_dir}. "
                "Pass download=True to download it automatically."
            )

    # CSV-based path: exact class ordering + exact val/test split
    csv_path = dataset_dir / "noisy_imagenette.csv"
    if csv_path.exists():
        if transform is None:
            transform = get_imagenette_transforms(
                train=(split == "train"), image_size=image_size
            )
        return _build_csv_dataset(dataset_dir, split, transform)

    # Fallback: ImageFolder (no CSV — used by unit tests)
    if split == "train":
        split_dir = dataset_dir / "train"
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        if transform is None:
            transform = get_imagenette_transforms(train=True, image_size=image_size)
        return ImageFolder(str(split_dir), transform=transform)
    else:
        val_dir = dataset_dir / "val"
        if not val_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {val_dir}")
        if transform is None:
            transform = get_imagenette_transforms(train=False, image_size=image_size)
        full_val = ImageFolder(str(val_dir), transform=transform)
        return _split_val(full_val, split)
