"""Tests for vit_shapley.data.imagenette.

These tests do NOT download the real dataset; they use a minimal ImageFolder
structure created in a tmp_path fixture to exercise all code paths.
"""

from pathlib import Path

import pytest
import torch
from PIL import Image

from vit_shapley.data.imagenette import (
    IMAGENETTE_CLASS_DISPLAY_NAMES,
    IMAGENETTE_CLASSES,
    _IMAGENETTE_DIRNAME,
    get_imagenette_dataset,
    get_imagenette_transforms,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_imagenette(
    root: Path, num_classes: int = 3, imgs_per_class: int = 2
) -> Path:
    """Create a minimal fake ImageNette directory structure."""
    dataset_dir = root / _IMAGENETTE_DIRNAME
    for split in ("train", "val"):
        for cls_idx in range(num_classes):
            cls_dir = dataset_dir / split / f"n{cls_idx:04d}"
            cls_dir.mkdir(parents=True, exist_ok=True)
            for img_idx in range(imgs_per_class):
                img_path = cls_dir / f"img_{img_idx}.JPEG"
                Image.new(
                    "RGB", (32, 32), color=(cls_idx * 40, img_idx * 60, 128)
                ).save(img_path)
    return dataset_dir


# ---------------------------------------------------------------------------
# Class name tests
# ---------------------------------------------------------------------------


class TestClassNames:
    def test_display_names_length(self):
        """IMAGENETTE_CLASS_DISPLAY_NAMES must have exactly 10 entries."""
        assert len(IMAGENETTE_CLASS_DISPLAY_NAMES) == 10

    def test_display_names_match_classes_length(self):
        """Display names and synset IDs must have the same length."""
        assert len(IMAGENETTE_CLASS_DISPLAY_NAMES) == len(IMAGENETTE_CLASSES)

    def test_display_names_are_strings(self):
        """All display names must be non-empty strings."""
        for name in IMAGENETTE_CLASS_DISPLAY_NAMES:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_display_names_are_human_readable(self):
        """Display names must not be WordNet synset IDs (nXXXXXXXX)."""
        import re

        synset_pattern = re.compile(r"^n\d{8}$")
        for name in IMAGENETTE_CLASS_DISPLAY_NAMES:
            assert not synset_pattern.match(name), (
                f"Display name '{name}' looks like a synset ID"
            )


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------


class TestGetImagenetteTransforms:
    def test_train_transform_output_shape(self):
        transform = get_imagenette_transforms(train=True, image_size=224)
        img = Image.new("RGB", (160, 160))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_val_transform_output_shape(self):
        transform = get_imagenette_transforms(train=False, image_size=224)
        img = Image.new("RGB", (256, 256))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_custom_image_size(self):
        for size in (32, 64, 128):
            transform = get_imagenette_transforms(train=False, image_size=size)
            img = Image.new("RGB", (size * 2, size * 2))
            tensor = transform(img)
            assert tensor.shape == (3, size, size), f"Failed for image_size={size}"

    def test_train_has_vertical_flip(self):
        """Training transform must include RandomVerticalFlip (fix 5)."""
        import torchvision.transforms as T

        transform = get_imagenette_transforms(train=True, image_size=64)
        has_vflip = any(
            isinstance(t, T.RandomVerticalFlip) for t in transform.transforms
        )
        assert has_vflip, "Training transform is missing RandomVerticalFlip"

    def test_train_rrc_scale(self):
        """RandomResizedCrop scale must be (0.8, 1.2) (fix 6)."""
        import torchvision.transforms as T

        transform = get_imagenette_transforms(train=True, image_size=64)
        rrc = next(
            t for t in transform.transforms if isinstance(t, T.RandomResizedCrop)
        )
        assert rrc.scale == (0.8, 1.2), f"Expected scale (0.8, 1.2), got {rrc.scale}"

    def test_train_has_initial_resize(self):
        """Training transform must start with Resize((256,256)) (fix 6)."""
        import torchvision.transforms as T

        transform = get_imagenette_transforms(train=True, image_size=64)
        first = transform.transforms[0]
        assert isinstance(first, T.Resize), "First transform must be Resize"

    def test_val_uses_square_resize(self):
        """Validation Resize must force 256×256 (fix 6)."""
        import torchvision.transforms as T

        transform = get_imagenette_transforms(train=False, image_size=64)
        first = transform.transforms[0]
        assert isinstance(first, T.Resize)
        assert tuple(first.size) == (256, 256), f"Expected (256, 256), got {first.size}"

    def test_train_color_jitter_params(self):
        """ColorJitter must have (br=0.2, co=0.2, sat=0.1, hue=0.1) wrapped in RandomApply(p=0.8) (fix 7)."""
        import torchvision.transforms as T

        transform = get_imagenette_transforms(train=True, image_size=64)
        random_applies = [
            t for t in transform.transforms if isinstance(t, T.RandomApply)
        ]
        assert len(random_applies) == 1, "Expected exactly one RandomApply"
        ra = random_applies[0]
        assert abs(ra.p - 0.8) < 1e-6, f"RandomApply p must be 0.8, got {ra.p}"
        cj = ra.transforms[0]
        assert isinstance(cj, T.ColorJitter)
        assert cj.brightness == (0.8, 1.2), f"brightness: {cj.brightness}"
        assert cj.contrast == (0.8, 1.2), f"contrast: {cj.contrast}"
        assert cj.saturation == (0.9, 1.1), f"saturation: {cj.saturation}"
        assert abs(cj.hue[0] - (-0.1)) < 1e-6 and abs(cj.hue[1] - 0.1) < 1e-6, (
            f"hue: {cj.hue}"
        )


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


class TestGetImagenetteDataset:
    def test_invalid_split_raises(self, tmp_path):
        with pytest.raises(AssertionError, match="train.*val.*test"):
            get_imagenette_dataset(root=tmp_path, split="unknown")

    def test_missing_data_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_imagenette_dataset(root=tmp_path, split="train", download=False)

    def test_train_dataset_loads(self, tmp_path):
        _make_fake_imagenette(tmp_path, num_classes=3, imgs_per_class=2)
        dataset = get_imagenette_dataset(root=tmp_path, split="train", image_size=32)
        assert len(dataset) == 3 * 2  # num_classes * imgs_per_class
        assert len(dataset.classes) == 3

    def test_val_dataset_loads(self, tmp_path):
        # val = larger half of the val/ folder (~50%)
        _make_fake_imagenette(tmp_path, num_classes=3, imgs_per_class=2)
        dataset = get_imagenette_dataset(root=tmp_path, split="val", image_size=32)
        total_val = 3 * 2  # 6 images in val/
        expected = total_val - int(total_val * 0.5)  # n_val = n - n_test
        assert len(dataset) == expected

    def test_test_dataset_loads(self, tmp_path):
        # test = smaller half of the val/ folder (~50%)
        _make_fake_imagenette(tmp_path, num_classes=3, imgs_per_class=2)
        dataset = get_imagenette_dataset(root=tmp_path, split="test", image_size=32)
        total_val = 3 * 2
        expected = int(total_val * 0.5)  # n_test
        assert len(dataset) == expected

    def test_val_plus_test_equals_total_val(self, tmp_path):
        """val and test subsets must partition the full val/ folder."""
        _make_fake_imagenette(tmp_path, num_classes=4, imgs_per_class=3)
        val_ds = get_imagenette_dataset(root=tmp_path, split="val", image_size=32)
        test_ds = get_imagenette_dataset(root=tmp_path, split="test", image_size=32)
        total = 4 * 3
        assert len(val_ds) + len(test_ds) == total

    def test_val_test_no_overlap(self, tmp_path):
        """val and test subsets must be disjoint."""
        _make_fake_imagenette(tmp_path, num_classes=4, imgs_per_class=3)
        val_ds = get_imagenette_dataset(root=tmp_path, split="val", image_size=32)
        test_ds = get_imagenette_dataset(root=tmp_path, split="test", image_size=32)
        val_idx = set(val_ds.indices)
        test_idx = set(test_ds.indices)
        assert val_idx.isdisjoint(test_idx), "val and test subsets overlap"

    def test_dataset_item_shape(self, tmp_path):
        _make_fake_imagenette(tmp_path, num_classes=3, imgs_per_class=2)
        dataset = get_imagenette_dataset(root=tmp_path, split="train", image_size=32)
        img, label = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 32, 32)
        assert isinstance(label, int)

    def test_custom_transform_used(self, tmp_path):
        import torchvision.transforms as T

        _make_fake_imagenette(tmp_path, num_classes=2, imgs_per_class=1)
        custom_transform = T.Compose([T.Resize((16, 16)), T.ToTensor()])
        dataset = get_imagenette_dataset(
            root=tmp_path, split="train", transform=custom_transform
        )
        img, _ = dataset[0]
        assert img.shape == (3, 16, 16)

    def test_dataset_labels_in_range(self, tmp_path):
        _make_fake_imagenette(tmp_path, num_classes=5, imgs_per_class=3)
        dataset = get_imagenette_dataset(root=tmp_path, split="train", image_size=32)
        for _, label in dataset:
            assert 0 <= label < 5

    def test_val_classes_attribute(self, tmp_path):
        """_LabelledSubset must expose .classes from the underlying ImageFolder."""
        _make_fake_imagenette(tmp_path, num_classes=3, imgs_per_class=2)
        val_ds = get_imagenette_dataset(root=tmp_path, split="val", image_size=32)
        assert hasattr(val_ds, "classes")
        assert len(val_ds.classes) == 3
