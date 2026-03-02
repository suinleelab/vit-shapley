"""Tests for vit_shapley.data.pet.

These tests do NOT download the real dataset; they mock
``torchvision.datasets.OxfordIIITPet`` to exercise all code paths with a
lightweight fake dataset.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from vit_shapley.data.pet import (
    _LabelledSubset,
    _PET_DIRNAME,
    _SPLIT_SEED,
    _TRAIN_FRACTION,
    _VAL_FRACTION,
    get_pet_dataset,
    get_pet_transforms,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Fake breed names (sorted alphabetically like the real dataset)
_FAKE_CLASSES = ["Abyssinian", "Bengal", "Birman", "Bombay", "Chihuahua"]


class _FakeOxfordIIITPet:
    """Minimal mock that mimics torchvision.datasets.OxfordIIITPet."""

    def __init__(self, root, split, download=False, **kwargs):
        self.root = root
        self.split = split
        # Produce different sizes for trainval vs test
        if split == "trainval":
            self._n = 50
        else:
            self._n = 20
        # Assign labels cycling through classes
        self._labels = [i % len(_FAKE_CLASSES) for i in range(self._n)]
        self.classes = [_FAKE_CLASSES[l] for l in self._labels]

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        # Return a PIL image and an integer label
        img = Image.new("RGB", (256, 256), color=(idx % 256, 128, 64))
        return img, self._labels[idx]


def _make_fake_pet_dir(root: Path) -> Path:
    """Create the expected pet directory so existence checks pass."""
    pet_dir = root / _PET_DIRNAME
    pet_dir.mkdir(parents=True, exist_ok=True)
    return pet_dir


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------


class TestGetPetTransforms:
    def test_train_transform_output_shape(self):
        transform = get_pet_transforms(train=True, image_size=224)
        img = Image.new("RGB", (160, 160))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_val_transform_output_shape(self):
        transform = get_pet_transforms(train=False, image_size=224)
        img = Image.new("RGB", (256, 256))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_custom_image_size(self):
        for size in (32, 64, 128):
            transform = get_pet_transforms(train=False, image_size=size)
            img = Image.new("RGB", (size * 2, size * 2))
            tensor = transform(img)
            assert tensor.shape == (3, size, size), f"Failed for image_size={size}"

    def test_train_has_vertical_flip(self):
        import torchvision.transforms as T

        transform = get_pet_transforms(train=True, image_size=64)
        has_vflip = any(
            isinstance(t, T.RandomVerticalFlip) for t in transform.transforms
        )
        assert has_vflip, "Training transform is missing RandomVerticalFlip"

    def test_train_rrc_scale(self):
        import torchvision.transforms as T

        transform = get_pet_transforms(train=True, image_size=64)
        rrc = next(
            t for t in transform.transforms if isinstance(t, T.RandomResizedCrop)
        )
        assert rrc.scale == (0.8, 1.2), f"Expected scale (0.8, 1.2), got {rrc.scale}"

    def test_train_has_initial_resize(self):
        import torchvision.transforms as T

        transform = get_pet_transforms(train=True, image_size=64)
        first = transform.transforms[0]
        assert isinstance(first, T.Resize), "First transform must be Resize"

    def test_val_uses_square_resize(self):
        import torchvision.transforms as T

        transform = get_pet_transforms(train=False, image_size=64)
        first = transform.transforms[0]
        assert isinstance(first, T.Resize)
        assert tuple(first.size) == (256, 256), f"Expected (256, 256), got {first.size}"

    def test_train_color_jitter_params(self):
        import torchvision.transforms as T

        transform = get_pet_transforms(train=True, image_size=64)
        random_applies = [
            t for t in transform.transforms if isinstance(t, T.RandomApply)
        ]
        assert len(random_applies) == 1, "Expected exactly one RandomApply"
        ra = random_applies[0]
        assert abs(ra.p - 0.8) < 1e-6, f"RandomApply p must be 0.8, got {ra.p}"
        cj = ra.transforms[0]
        assert isinstance(cj, T.ColorJitter)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


@patch("vit_shapley.data.pet.OxfordIIITPet", _FakeOxfordIIITPet)
class TestGetPetDataset:
    def test_invalid_split_raises(self, tmp_path):
        with pytest.raises(AssertionError, match="train.*val.*test"):
            get_pet_dataset(root=tmp_path, split="unknown")

    def test_missing_data_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_pet_dataset(root=tmp_path, split="train", download=False)

    def test_train_dataset_loads(self, tmp_path):
        _make_fake_pet_dir(tmp_path)
        dataset = get_pet_dataset(root=tmp_path, split="train", image_size=32)
        # Pool = 50 (trainval) + 20 (test) = 70; train = 80% = 56
        total_pool = 70
        expected_train = int(total_pool * _TRAIN_FRACTION)
        assert len(dataset) == expected_train

    def test_val_dataset_loads(self, tmp_path):
        _make_fake_pet_dir(tmp_path)
        dataset = get_pet_dataset(root=tmp_path, split="val", image_size=32)
        total_pool = 70
        expected_val = int(total_pool * _VAL_FRACTION)
        assert len(dataset) == expected_val

    def test_test_dataset_loads(self, tmp_path):
        _make_fake_pet_dir(tmp_path)
        dataset = get_pet_dataset(root=tmp_path, split="test", image_size=32)
        total_pool = 70
        n_train = int(total_pool * _TRAIN_FRACTION)
        n_val = int(total_pool * _VAL_FRACTION)
        expected_test = total_pool - n_train - n_val
        assert len(dataset) == expected_test

    def test_all_splits_partition_full_pool(self, tmp_path):
        """train + val + test must equal the full pool (trainval + test)."""
        _make_fake_pet_dir(tmp_path)
        train_ds = get_pet_dataset(root=tmp_path, split="train", image_size=32)
        val_ds = get_pet_dataset(root=tmp_path, split="val", image_size=32)
        test_ds = get_pet_dataset(root=tmp_path, split="test", image_size=32)
        assert len(train_ds) + len(val_ds) + len(test_ds) == 70

    def test_all_splits_no_overlap(self, tmp_path):
        """train, val, and test subsets must be pairwise disjoint."""
        _make_fake_pet_dir(tmp_path)
        train_ds = get_pet_dataset(root=tmp_path, split="train", image_size=32)
        val_ds = get_pet_dataset(root=tmp_path, split="val", image_size=32)
        test_ds = get_pet_dataset(root=tmp_path, split="test", image_size=32)
        train_idx = set(train_ds._indices)
        val_idx = set(val_ds._indices)
        test_idx = set(test_ds._indices)
        assert train_idx.isdisjoint(val_idx), "train and val overlap"
        assert train_idx.isdisjoint(test_idx), "train and test overlap"
        assert val_idx.isdisjoint(test_idx), "val and test overlap"

    def test_dataset_item_shape(self, tmp_path):
        _make_fake_pet_dir(tmp_path)
        dataset = get_pet_dataset(root=tmp_path, split="train", image_size=32)
        img, label = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 32, 32)
        assert isinstance(label, int)

    def test_custom_transform_used(self, tmp_path):
        import torchvision.transforms as T

        _make_fake_pet_dir(tmp_path)
        custom_transform = T.Compose([T.Resize((16, 16)), T.ToTensor()])
        dataset = get_pet_dataset(
            root=tmp_path, split="train", transform=custom_transform
        )
        img, _ = dataset[0]
        assert img.shape == (3, 16, 16)

    def test_dataset_labels_in_range(self, tmp_path):
        _make_fake_pet_dir(tmp_path)
        dataset = get_pet_dataset(root=tmp_path, split="train", image_size=32)
        num_classes = len(dataset.classes)
        for i in range(len(dataset)):
            _, label = dataset[i]
            assert 0 <= label < num_classes

    def test_classes_attribute(self, tmp_path):
        _make_fake_pet_dir(tmp_path)
        dataset = get_pet_dataset(root=tmp_path, split="train", image_size=32)
        assert hasattr(dataset, "classes")
        assert isinstance(dataset.classes, list)
        assert len(dataset.classes) == len(_FAKE_CLASSES)

    def test_class_to_idx_attribute(self, tmp_path):
        _make_fake_pet_dir(tmp_path)
        dataset = get_pet_dataset(root=tmp_path, split="train", image_size=32)
        assert hasattr(dataset, "class_to_idx")
        assert isinstance(dataset.class_to_idx, dict)
        assert len(dataset.class_to_idx) == len(_FAKE_CLASSES)
        # Check consistency: class_to_idx maps classes to 0..N-1
        for cls_name, idx in dataset.class_to_idx.items():
            assert cls_name in dataset.classes
            assert 0 <= idx < len(dataset.classes)

    def test_val_classes_attribute(self, tmp_path):
        """_LabelledSubset must expose .classes from underlying dataset."""
        _make_fake_pet_dir(tmp_path)
        val_ds = get_pet_dataset(root=tmp_path, split="val", image_size=32)
        assert hasattr(val_ds, "classes")
        assert len(val_ds.classes) == len(_FAKE_CLASSES)

    def test_test_classes_attribute(self, tmp_path):
        """Test split must also expose .classes."""
        _make_fake_pet_dir(tmp_path)
        test_ds = get_pet_dataset(root=tmp_path, split="test", image_size=32)
        assert hasattr(test_ds, "classes")
        assert len(test_ds.classes) == len(_FAKE_CLASSES)

    def test_deterministic_splits(self, tmp_path):
        """Calling get_pet_dataset twice gives the same indices."""
        _make_fake_pet_dir(tmp_path)
        ds1 = get_pet_dataset(root=tmp_path, split="train", image_size=32)
        ds2 = get_pet_dataset(root=tmp_path, split="train", image_size=32)
        assert ds1._indices == ds2._indices

    def test_classes_sorted(self, tmp_path):
        """Classes should be sorted alphabetically."""
        _make_fake_pet_dir(tmp_path)
        dataset = get_pet_dataset(root=tmp_path, split="train", image_size=32)
        assert dataset.classes == sorted(dataset.classes)


# ---------------------------------------------------------------------------
# _LabelledSubset tests
# ---------------------------------------------------------------------------


class TestLabelledSubset:
    def test_len(self):
        parent = [(Image.new("RGB", (8, 8)), i) for i in range(10)]
        subset = _LabelledSubset(parent, [0, 2, 4], ["a", "b"], {"a": 0, "b": 1})
        assert len(subset) == 3

    def test_getitem(self):
        parent = [(Image.new("RGB", (8, 8)), i) for i in range(10)]
        subset = _LabelledSubset(parent, [3, 7], ["a", "b"], {"a": 0, "b": 1})
        img, label = subset[0]
        assert label == 3  # parent[3]
        img, label = subset[1]
        assert label == 7  # parent[7]

    def test_transform_applied(self):
        import torchvision.transforms as T

        parent = [(Image.new("RGB", (32, 32)), i) for i in range(5)]
        transform = T.Compose([T.Resize((16, 16)), T.ToTensor()])
        subset = _LabelledSubset(
            parent, [0, 1], ["a"], {"a": 0}, transform=transform
        )
        img, _ = subset[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 16, 16)

    def test_classes_exposed(self):
        parent = [(Image.new("RGB", (8, 8)), 0)]
        classes = ["cat", "dog"]
        class_to_idx = {"cat": 0, "dog": 1}
        subset = _LabelledSubset(parent, [0], classes, class_to_idx)
        assert subset.classes == ["cat", "dog"]
        assert subset.class_to_idx == {"cat": 0, "dog": 1}
