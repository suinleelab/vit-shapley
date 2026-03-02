"""Oxford-IIIT Pet dataset download and loader utilities.

The Oxford-IIIT Pet Dataset contains 37 breed categories with roughly 200 images
per class (7,349 total). We use ``torchvision.datasets.OxfordIIITPet`` for
downloading and loading.

We combine torchvision's ``"trainval"`` and ``"test"`` splits into a single pool,
then deterministically split into train/val/test with an 80/10/10 ratio using a
fixed random seed.

Reference: https://www.robots.ox.ac.uk/~vgg/data/pets/
"""

import os
from pathlib import Path

import numpy as np
import torch.utils.data
import torchvision.transforms as T
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import InterpolationMode

# ImageNet channel mean / std (Pet uses ImageNet-pretrained models)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_PET_DIRNAME = "oxford-iiit-pet"

# Split ratios and seed (80/10/10 over the full 7,349-image pool)
_TRAIN_FRACTION = 0.8
_VAL_FRACTION = 0.1
_SPLIT_SEED = 42


def download_pet(root: str | os.PathLike) -> Path:
    """Download the Oxford-IIIT Pet dataset into *root* if not already present.

    Args:
        root: Directory where the dataset will be stored.
              After download the data lives at ``<root>/oxford-iiit-pet/``.

    Returns:
        Path to the ``oxford-iiit-pet`` directory.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # Trigger torchvision download (idempotent if already present)
    OxfordIIITPet(root=str(root), split="trainval", download=True)

    return root / _PET_DIRNAME


def get_pet_transforms(
    train: bool = True,
    image_size: int = 224,
) -> T.Compose:
    """Return augmentation transforms matching the ImageNette pipeline.

    Both datasets use ImageNet-pretrained models, so we apply the same
    augmentation strategy.

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
# Subset wrapper that preserves .classes / .class_to_idx
# ---------------------------------------------------------------------------


class _LabelledSubset(torch.utils.data.Dataset):
    """A subset of a dataset that exposes ``.classes`` and ``.class_to_idx``.

    The torchvision ``OxfordIIITPet`` dataset doesn't have a ``.classes``
    attribute in the same format as ``ImageFolder``, so this wrapper builds
    the interface expected by the rest of the pipeline.
    """

    def __init__(self, dataset, indices, classes, class_to_idx, transform=None):
        self._dataset = dataset
        self._indices = list(indices)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self._transform = transform

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        img, label = self._dataset[self._indices[idx]]
        if self._transform is not None:
            img = self._transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _ConcatPetDataset:
    """Concatenates torchvision's trainval + test splits into one pool."""

    def __init__(self, root: str):
        self._trainval = OxfordIIITPet(root=root, split="trainval", download=False)
        self._test = OxfordIIITPet(root=root, split="test", download=False)
        self._n_trainval = len(self._trainval)
        # Build sorted class list from trainval (has .classes attribute)
        self.classes = sorted(set(self._trainval.classes))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return self._n_trainval + len(self._test)

    def __getitem__(self, idx):
        if idx < self._n_trainval:
            return self._trainval[idx]
        return self._test[idx - self._n_trainval]


def _build_pet_splits(
    root: Path,
) -> tuple[list[int], list[int], list[int], list[str], dict[str, int]]:
    """Pool trainval + test, then split 80/10/10 deterministically.

    Returns (train_indices, val_indices, test_indices, classes, class_to_idx).
    """
    full = _ConcatPetDataset(str(root))

    n = len(full)
    rng = np.random.RandomState(_SPLIT_SEED)
    perm = rng.permutation(n)

    n_train = int(n * _TRAIN_FRACTION)
    n_val = int(n * _VAL_FRACTION)
    # test gets the remainder

    train_indices = sorted(perm[:n_train].tolist())
    val_indices = sorted(perm[n_train : n_train + n_val].tolist())
    test_indices = sorted(perm[n_train + n_val :].tolist())

    return train_indices, val_indices, test_indices, full.classes, full.class_to_idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_pet_dataset(
    root: str | os.PathLike,
    split: str = "train",
    image_size: int = 224,
    transform=None,
    download: bool = False,
) -> _LabelledSubset:
    """Return a dataset for the requested split of Oxford-IIIT Pet.

    Torchvision's ``"trainval"`` and ``"test"`` splits are combined into a
    single pool of all images, then deterministically split 80/10/10 into
    ``"train"`` / ``"val"`` / ``"test"``.

    Args:
        root: Root directory that *contains* (or will contain) the
              ``oxford-iiit-pet`` sub-directory.
        split: ``"train"``, ``"val"``, or ``"test"``.
        image_size: Passed to :func:`get_pet_transforms` when *transform*
                    is ``None``.
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
    dataset_dir = root / _PET_DIRNAME

    if not dataset_dir.exists():
        if download:
            download_pet(root)
        else:
            raise FileNotFoundError(
                f"Oxford-IIIT Pet directory not found: {dataset_dir}. "
                "Pass download=True to download it automatically."
            )

    train_idx, val_idx, test_idx, classes, class_to_idx = _build_pet_splits(root)
    full = _ConcatPetDataset(str(root))

    if transform is None:
        transform = get_pet_transforms(
            train=(split == "train"), image_size=image_size
        )

    indices = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
    return _LabelledSubset(full, indices, classes, class_to_idx, transform)
