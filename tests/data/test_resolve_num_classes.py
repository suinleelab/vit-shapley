"""Tests for vit_shapley.data.resolve_num_classes."""

import pytest

from vit_shapley.data import resolve_num_classes


class _FakeDataset:
    """Minimal dataset stub with a .classes attribute."""

    def __init__(self, classes: list[str]):
        self.classes = classes


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestBinaryHappyPath:
    def test_returns_1_for_two_classes(self):
        ds = _FakeDataset(["negative", "positive"])
        assert resolve_num_classes(ds, "binary") == 1

    def test_any_two_class_names(self):
        ds = _FakeDataset(["cat", "dog"])
        assert resolve_num_classes(ds, "binary") == 1


class TestMulticlassHappyPath:
    def test_returns_class_count_for_10(self):
        ds = _FakeDataset([f"c{i}" for i in range(10)])
        assert resolve_num_classes(ds, "multiclass") == 10

    def test_returns_class_count_for_37(self):
        ds = _FakeDataset([f"breed_{i}" for i in range(37)])
        assert resolve_num_classes(ds, "multiclass") == 37

    def test_returns_class_count_for_3(self):
        ds = _FakeDataset(["a", "b", "c"])
        assert resolve_num_classes(ds, "multiclass") == 3


# ---------------------------------------------------------------------------
# Mismatch errors
# ---------------------------------------------------------------------------


class TestBinaryMismatch:
    def test_binary_with_10_classes_raises(self):
        ds = _FakeDataset([f"c{i}" for i in range(10)])
        with pytest.raises(ValueError, match="binary.*exactly 2"):
            resolve_num_classes(ds, "binary")

    def test_binary_with_3_classes_raises(self):
        ds = _FakeDataset(["a", "b", "c"])
        with pytest.raises(ValueError, match="binary.*exactly 2"):
            resolve_num_classes(ds, "binary")

    def test_binary_with_1_class_raises(self):
        ds = _FakeDataset(["only"])
        with pytest.raises(ValueError, match="binary.*exactly 2"):
            resolve_num_classes(ds, "binary")


class TestMulticlassMismatch:
    def test_multiclass_with_2_classes_raises(self):
        ds = _FakeDataset(["negative", "positive"])
        with pytest.raises(ValueError, match="multiclass.*3 or more"):
            resolve_num_classes(ds, "multiclass")

    def test_multiclass_with_1_class_raises(self):
        ds = _FakeDataset(["only"])
        with pytest.raises(ValueError, match="multiclass.*3 or more"):
            resolve_num_classes(ds, "multiclass")


# ---------------------------------------------------------------------------
# Invalid target_type
# ---------------------------------------------------------------------------


class TestInvalidTargetType:
    def test_unknown_target_type_raises(self):
        ds = _FakeDataset(["a", "b"])
        with pytest.raises(ValueError, match="Unknown target_type"):
            resolve_num_classes(ds, "regression")

    def test_empty_string_raises(self):
        ds = _FakeDataset(["a", "b"])
        with pytest.raises(ValueError, match="Unknown target_type"):
            resolve_num_classes(ds, "")


# ---------------------------------------------------------------------------
# Error messages include dataset classes
# ---------------------------------------------------------------------------


class TestErrorMessages:
    def test_binary_error_shows_class_list(self):
        ds = _FakeDataset(["cat", "dog", "fish"])
        with pytest.raises(ValueError, match=r"\['cat', 'dog', 'fish'\]"):
            resolve_num_classes(ds, "binary")

    def test_multiclass_error_suggests_binary(self):
        ds = _FakeDataset(["neg", "pos"])
        with pytest.raises(ValueError, match="target_type='binary'"):
            resolve_num_classes(ds, "multiclass")
