"""Tests for vit_shapley.baselines.perturbation.

Uses vit_tiny_patch16_224 without pretrained weights for speed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vit_shapley.baselines.perturbation import leave_one_out, rise
from vit_shapley.models.surrogate import build_vit_surrogate

_TINY_MODEL = "vit_tiny_patch16_224"
_NUM_CLASSES = 5
_NUM_PATCHES = 196  # 14×14 for 224px with patch16


@pytest.fixture(scope="module")
def surrogate():
    m = build_vit_surrogate(_TINY_MODEL, num_classes=_NUM_CLASSES)
    m.eval()
    return m


@pytest.fixture(scope="module")
def single_image():
    torch.manual_seed(0)
    return torch.randn(3, 224, 224)


# ---------------------------------------------------------------------------
# leave_one_out
# ---------------------------------------------------------------------------


class TestLeaveOneOut:
    def test_output_shape_multiclass(self, surrogate, single_image):
        result = leave_one_out(single_image, surrogate)
        assert result.shape == (_NUM_CLASSES, _NUM_PATCHES)

    def test_output_shape_binary(self, single_image):
        surrogate_bin = build_vit_surrogate(_TINY_MODEL, num_classes=1)
        surrogate_bin.eval()
        result = leave_one_out(single_image, surrogate_bin, binary=True)
        assert result.shape == (1, _NUM_PATCHES)

    def test_values_finite(self, surrogate, single_image):
        result = leave_one_out(single_image, surrogate)
        assert np.all(np.isfinite(result))

    def test_num_patches_explicit(self, surrogate, single_image):
        """Passing num_patches explicitly should give same result."""
        r1 = leave_one_out(single_image, surrogate)
        r2 = leave_one_out(single_image, surrogate, num_patches=_NUM_PATCHES)
        np.testing.assert_array_equal(r1, r2)

    def test_constant_model_zero_attribution(self, single_image):
        """A surrogate that outputs the same logits for all masks must give zero LOO."""
        import torch.nn as nn

        class _ConstantSurrogate(nn.Module):
            def forward(self, x, patch_mask=None):
                return torch.zeros(x.shape[0], _NUM_CLASSES)

        const_surrogate = _ConstantSurrogate()
        result = leave_one_out(single_image, const_surrogate, num_patches=_NUM_PATCHES)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# rise
# ---------------------------------------------------------------------------


class TestRise:
    def test_output_shape_multiclass(self, surrogate, single_image):
        result = rise(single_image, surrogate, N=200, batch_size=100)
        assert result.shape == (_NUM_CLASSES, _NUM_PATCHES)

    def test_output_shape_binary(self, single_image):
        surrogate_bin = build_vit_surrogate(_TINY_MODEL, num_classes=1)
        surrogate_bin.eval()
        result = rise(single_image, surrogate_bin, N=100, batch_size=50, binary=True)
        assert result.shape == (1, _NUM_PATCHES)

    def test_values_finite(self, surrogate, single_image):
        result = rise(single_image, surrogate, N=200, batch_size=100)
        assert np.all(np.isfinite(result))

    def test_n_not_divisible_raises(self, surrogate, single_image):
        with pytest.raises(ValueError, match="divisible"):
            rise(single_image, surrogate, N=101, batch_size=100)

    def test_num_patches_explicit(self, surrogate, single_image):
        result = rise(
            single_image, surrogate, N=100, batch_size=100, num_patches=_NUM_PATCHES
        )
        assert result.shape == (_NUM_CLASSES, _NUM_PATCHES)

    def test_values_nonnegative_for_softmax_output(self, surrogate, single_image):
        """RISE scores for softmax probabilities should be non-negative."""
        result = rise(single_image, surrogate, N=200, batch_size=100)
        assert np.all(result >= 0)
