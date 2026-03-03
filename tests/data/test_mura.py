"""Tests for vit_shapley.data.mura.

These tests do NOT require the real MURA dataset; they create fake directory
structures with synthetic CSV files and tiny PNG images to exercise all code
paths.
"""

import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from vit_shapley.data.mura import (
    MURA_CLASS_TO_IDX,
    MURA_CLASSES,
    MURADataset,
    _MURA_DIRNAME,
    _build_mura_split,
    _extract_label,
    _extract_patient,
    get_mura_dataset,
    get_mura_transforms,
)

# ---------------------------------------------------------------------------
# Helpers — build a fake MURA directory tree
# ---------------------------------------------------------------------------

_BODY_PARTS = ["XR_ELBOW", "XR_WRIST"]
_LABELS = ["positive", "negative"]


def _make_fake_mura(
    root: Path,
    n_train_patients: int = 20,
    n_valid_patients: int = 6,
    studies_per_patient: int = 1,
    images_per_study: int = 2,
) -> Path:
    """Create a minimal fake MURA-v1.1 directory with CSV files and images.

    Returns the ``MURA-v1.1`` directory path.
    """
    mura_dir = root / _MURA_DIRNAME
    train_paths = []
    valid_paths = []

    patient_id = 0
    for split_name, n_patients, path_list in [
        ("train", n_train_patients, train_paths),
        ("valid", n_valid_patients, valid_paths),
    ]:
        for _ in range(n_patients):
            patient_id += 1
            patient_str = f"patient{patient_id:05d}"
            body_part = _BODY_PARTS[patient_id % len(_BODY_PARTS)]
            label = _LABELS[patient_id % len(_LABELS)]

            for s in range(1, studies_per_patient + 1):
                study_folder = f"study{s}_{label}"
                study_dir = (
                    mura_dir / split_name / body_part / patient_str / study_folder
                )
                study_dir.mkdir(parents=True, exist_ok=True)

                for img_idx in range(1, images_per_study + 1):
                    img_name = f"image{img_idx}.png"
                    img_path = study_dir / img_name
                    # Create a tiny grayscale PNG (like real MURA X-rays)
                    Image.new("L", (16, 16), color=128).save(img_path)

                    # CSV paths use "MURA-v1.1/" prefix
                    csv_path = (
                        f"MURA-v1.1/{split_name}/{body_part}/"
                        f"{patient_str}/{study_folder}/{img_name}"
                    )
                    path_list.append(csv_path)

    # Write CSV files (no header, one path per line)
    (mura_dir / "train_image_paths.csv").write_text(
        "\n".join(train_paths) + "\n"
    )
    (mura_dir / "valid_image_paths.csv").write_text(
        "\n".join(valid_paths) + "\n"
    )

    return mura_dir


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------


class TestGetMuraTransforms:
    def test_train_transform_output_shape(self):
        transform = get_mura_transforms(train=True, image_size=224)
        img = Image.new("RGB", (160, 160))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_val_transform_output_shape(self):
        transform = get_mura_transforms(train=False, image_size=224)
        img = Image.new("RGB", (256, 256))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_custom_image_size(self):
        for size in (32, 64, 128):
            transform = get_mura_transforms(train=False, image_size=size)
            img = Image.new("RGB", (size * 2, size * 2))
            tensor = transform(img)
            assert tensor.shape == (3, size, size), f"Failed for image_size={size}"

    def test_train_has_vertical_flip(self):
        import torchvision.transforms as T

        transform = get_mura_transforms(train=True, image_size=64)
        has_vflip = any(
            isinstance(t, T.RandomVerticalFlip) for t in transform.transforms
        )
        assert has_vflip, "Training transform is missing RandomVerticalFlip"

    def test_train_rrc_scale(self):
        import torchvision.transforms as T

        transform = get_mura_transforms(train=True, image_size=64)
        rrc = next(
            t for t in transform.transforms if isinstance(t, T.RandomResizedCrop)
        )
        assert rrc.scale == (0.8, 1.2), f"Expected scale (0.8, 1.2), got {rrc.scale}"

    def test_train_has_initial_resize(self):
        import torchvision.transforms as T

        transform = get_mura_transforms(train=True, image_size=64)
        first = transform.transforms[0]
        assert isinstance(first, T.Resize), "First transform must be Resize"

    def test_val_uses_square_resize(self):
        import torchvision.transforms as T

        transform = get_mura_transforms(train=False, image_size=64)
        first = transform.transforms[0]
        assert isinstance(first, T.Resize)
        assert tuple(first.size) == (256, 256), (
            f"Expected (256, 256), got {first.size}"
        )

    def test_train_color_jitter_params(self):
        import torchvision.transforms as T

        transform = get_mura_transforms(train=True, image_size=64)
        random_applies = [
            t for t in transform.transforms if isinstance(t, T.RandomApply)
        ]
        assert len(random_applies) == 1, "Expected exactly one RandomApply"
        ra = random_applies[0]
        assert abs(ra.p - 0.8) < 1e-6, f"RandomApply p must be 0.8, got {ra.p}"
        cj = ra.transforms[0]
        assert isinstance(cj, T.ColorJitter)

    def test_grayscale_image_converted_to_rgb(self):
        """MURA X-rays are grayscale; the transform should handle 3-ch input
        from .convert('RGB')."""
        transform = get_mura_transforms(train=False, image_size=32)
        # Simulate what MURADataset does: open grayscale, convert to RGB
        gray = Image.new("L", (64, 64), color=128)
        rgb = gray.convert("RGB")
        tensor = transform(rgb)
        assert tensor.shape == (3, 32, 32)


# ---------------------------------------------------------------------------
# Label / patient extraction helpers
# ---------------------------------------------------------------------------


class TestExtractLabel:
    def test_positive(self):
        path = "MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/image1.png"
        assert _extract_label(path) == "positive"

    def test_negative(self):
        path = "MURA-v1.1/valid/XR_WRIST/patient11185/study1_negative/image2.png"
        assert _extract_label(path) == "negative"


class TestExtractPatient:
    def test_extract(self):
        path = "MURA-v1.1/train/XR_ELBOW/patient00042/study1_positive/image1.png"
        assert _extract_patient(path) == "patient00042"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_classes(self):
        assert MURA_CLASSES == ["negative", "positive"]

    def test_class_to_idx(self):
        assert MURA_CLASS_TO_IDX == {"negative": 0, "positive": 1}

    def test_binary(self):
        assert len(MURA_CLASSES) == 2


# ---------------------------------------------------------------------------
# Dataset class unit tests
# ---------------------------------------------------------------------------


class TestMURADataset:
    def test_len(self, tmp_path):
        data_list = [
            {"img_path": str(tmp_path / "a.png"), "label": 0},
            {"img_path": str(tmp_path / "b.png"), "label": 1},
        ]
        # Create dummy images
        for d in data_list:
            Image.new("RGB", (8, 8)).save(d["img_path"])
        ds = MURADataset(data_list)
        assert len(ds) == 2

    def test_classes_attribute(self, tmp_path):
        ds = MURADataset([])
        assert ds.classes == ["negative", "positive"]
        assert ds.class_to_idx == {"negative": 0, "positive": 1}

    def test_getitem_returns_rgb(self, tmp_path):
        import torchvision.transforms as T

        img_path = tmp_path / "test.png"
        # Save a grayscale image
        Image.new("L", (32, 32), color=100).save(img_path)
        data_list = [{"img_path": str(img_path), "label": 1}]
        transform = T.Compose([T.Resize((16, 16)), T.ToTensor()])
        ds = MURADataset(data_list, transform=transform)
        img, label = ds[0]
        assert img.shape == (3, 16, 16), "Image must be converted to 3-channel RGB"
        assert label == 1


# ---------------------------------------------------------------------------
# Split logic tests
# ---------------------------------------------------------------------------


class TestBuildMuraSplit:
    def test_train_val_no_patient_overlap(self, tmp_path):
        """Train and val must not share any patients."""
        _make_fake_mura(tmp_path, n_train_patients=20)
        train_list = _build_mura_split(tmp_path / _MURA_DIRNAME, "train")
        val_list = _build_mura_split(tmp_path / _MURA_DIRNAME, "val")

        train_patients = {
            Path(d["img_path"]).parent.parent.name for d in train_list
        }
        val_patients = {
            Path(d["img_path"]).parent.parent.name for d in val_list
        }
        assert train_patients.isdisjoint(val_patients), (
            "Train and val sets must not share patients"
        )

    def test_train_val_partition_all_train_images(self, tmp_path):
        """Train + val must include all images from train_image_paths.csv."""
        n_train = 20
        imgs_per = 2
        _make_fake_mura(tmp_path, n_train_patients=n_train, images_per_study=imgs_per)
        train_list = _build_mura_split(tmp_path / _MURA_DIRNAME, "train")
        val_list = _build_mura_split(tmp_path / _MURA_DIRNAME, "val")
        total = len(train_list) + len(val_list)
        assert total == n_train * imgs_per

    def test_val_size_matches_ceil(self, tmp_path):
        """Number of val patients = ceil(n_patients * 0.1)."""
        n_train = 20
        _make_fake_mura(tmp_path, n_train_patients=n_train)
        val_list = _build_mura_split(tmp_path / _MURA_DIRNAME, "val")
        val_patients = {
            Path(d["img_path"]).parent.parent.name for d in val_list
        }
        expected_n_val_patients = math.ceil(n_train * 0.1)
        assert len(val_patients) == expected_n_val_patients

    def test_test_split_uses_valid_csv(self, tmp_path):
        """Test split must match valid_image_paths.csv."""
        n_valid = 6
        imgs_per = 2
        _make_fake_mura(
            tmp_path, n_train_patients=10, n_valid_patients=n_valid,
            images_per_study=imgs_per,
        )
        test_list = _build_mura_split(tmp_path / _MURA_DIRNAME, "test")
        assert len(test_list) == n_valid * imgs_per

    def test_deterministic_splits(self, tmp_path):
        """Calling twice must produce the same result."""
        _make_fake_mura(tmp_path, n_train_patients=20)
        list1 = _build_mura_split(tmp_path / _MURA_DIRNAME, "train")
        list2 = _build_mura_split(tmp_path / _MURA_DIRNAME, "train")
        paths1 = [d["img_path"] for d in list1]
        paths2 = [d["img_path"] for d in list2]
        assert paths1 == paths2

    def test_shuffle_determinism(self, tmp_path):
        """The random.Random(42) shuffle must be reproducible."""
        _make_fake_mura(tmp_path, n_train_patients=20)
        list1 = _build_mura_split(tmp_path / _MURA_DIRNAME, "test")
        list2 = _build_mura_split(tmp_path / _MURA_DIRNAME, "test")
        assert [d["img_path"] for d in list1] == [d["img_path"] for d in list2]

    def test_labels_are_binary(self, tmp_path):
        """All labels must be 0 or 1."""
        _make_fake_mura(tmp_path)
        for split in ("train", "val", "test"):
            data = _build_mura_split(tmp_path / _MURA_DIRNAME, split)
            labels = {d["label"] for d in data}
            assert labels.issubset({0, 1}), f"Non-binary labels in {split}: {labels}"

    def test_img_paths_exist(self, tmp_path):
        """All image paths in the data list must point to existing files."""
        _make_fake_mura(tmp_path)
        for split in ("train", "val", "test"):
            data = _build_mura_split(tmp_path / _MURA_DIRNAME, split)
            for d in data:
                assert Path(d["img_path"]).exists(), (
                    f"Image path does not exist: {d['img_path']}"
                )


# ---------------------------------------------------------------------------
# get_mura_dataset tests
# ---------------------------------------------------------------------------


class TestGetMuraDataset:
    def test_invalid_split_raises(self, tmp_path):
        with pytest.raises(AssertionError, match="train.*val.*test"):
            get_mura_dataset(root=tmp_path, split="unknown")

    def test_missing_data_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="MURA"):
            get_mura_dataset(root=tmp_path, split="train", download=False)

    def test_download_true_still_raises_if_missing(self, tmp_path):
        """MURA cannot be auto-downloaded; download=True still raises."""
        with pytest.raises(FileNotFoundError, match="license"):
            get_mura_dataset(root=tmp_path, split="train", download=True)

    def test_train_loads(self, tmp_path):
        _make_fake_mura(tmp_path, n_train_patients=20, images_per_study=2)
        ds = get_mura_dataset(root=tmp_path, split="train", image_size=32)
        n_val_patients = math.ceil(20 * 0.1)
        # Each patient has 2 images; val gets n_val_patients
        # But assignment is per-patient, so we just check train + val = 40
        assert len(ds) > 0

    def test_val_loads(self, tmp_path):
        _make_fake_mura(tmp_path, n_train_patients=20, images_per_study=2)
        ds = get_mura_dataset(root=tmp_path, split="val", image_size=32)
        assert len(ds) > 0

    def test_test_loads(self, tmp_path):
        _make_fake_mura(tmp_path, n_valid_patients=6, images_per_study=2)
        ds = get_mura_dataset(root=tmp_path, split="test", image_size=32)
        assert len(ds) == 12

    def test_train_val_test_sizes_add_up(self, tmp_path):
        n_train_p = 20
        n_valid_p = 6
        imgs = 2
        _make_fake_mura(
            tmp_path, n_train_patients=n_train_p, n_valid_patients=n_valid_p,
            images_per_study=imgs,
        )
        train_ds = get_mura_dataset(root=tmp_path, split="train", image_size=32)
        val_ds = get_mura_dataset(root=tmp_path, split="val", image_size=32)
        test_ds = get_mura_dataset(root=tmp_path, split="test", image_size=32)
        assert len(train_ds) + len(val_ds) == n_train_p * imgs
        assert len(test_ds) == n_valid_p * imgs

    def test_item_shape(self, tmp_path):
        _make_fake_mura(tmp_path)
        ds = get_mura_dataset(root=tmp_path, split="train", image_size=32)
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 32, 32)
        assert isinstance(label, int)
        assert label in (0, 1)

    def test_custom_transform(self, tmp_path):
        import torchvision.transforms as T

        _make_fake_mura(tmp_path)
        custom = T.Compose([T.Resize((16, 16)), T.ToTensor()])
        ds = get_mura_dataset(root=tmp_path, split="train", transform=custom)
        img, _ = ds[0]
        assert img.shape == (3, 16, 16)

    def test_classes_attribute(self, tmp_path):
        _make_fake_mura(tmp_path)
        ds = get_mura_dataset(root=tmp_path, split="train", image_size=32)
        assert ds.classes == ["negative", "positive"]

    def test_class_to_idx_attribute(self, tmp_path):
        _make_fake_mura(tmp_path)
        ds = get_mura_dataset(root=tmp_path, split="train", image_size=32)
        assert ds.class_to_idx == {"negative": 0, "positive": 1}

    def test_labels_in_range(self, tmp_path):
        _make_fake_mura(tmp_path)
        ds = get_mura_dataset(root=tmp_path, split="train", image_size=32)
        for i in range(len(ds)):
            _, label = ds[i]
            assert label in (0, 1)

    def test_val_classes_attribute(self, tmp_path):
        _make_fake_mura(tmp_path)
        ds = get_mura_dataset(root=tmp_path, split="val", image_size=32)
        assert ds.classes == ["negative", "positive"]

    def test_test_classes_attribute(self, tmp_path):
        _make_fake_mura(tmp_path)
        ds = get_mura_dataset(root=tmp_path, split="test", image_size=32)
        assert ds.classes == ["negative", "positive"]
