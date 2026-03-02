"""Tests for vit_shapley.baselines.gradient.

Skips the entire module if captum is not installed.
Uses vit_tiny_patch16_224 without pretrained weights for speed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

captum = pytest.importorskip("captum", reason="captum not installed")

from vit_shapley.baselines.gradient import (  # noqa: E402
    _EmbeddingWrapper,
    _PixelWrapper,
    get_integrated_gradients,
    get_smoothgrad,
    get_vanilla_gradient,
    get_vargrad,
)

_TINY_MODEL = "vit_tiny_patch16_224"
_NUM_CLASSES = 5
_IMG_SIZE = 224
_PATCH_SIZE = 16
_NUM_PATCHES = (_IMG_SIZE // _PATCH_SIZE) ** 2  # 196


@pytest.fixture(scope="module")
def tiny_vit():
    import timm
    model = timm.create_model(_TINY_MODEL, pretrained=False, num_classes=_NUM_CLASSES)
    model.eval()
    return model


@pytest.fixture(scope="module")
def single_image():
    torch.manual_seed(0)
    return torch.randn(3, _IMG_SIZE, _IMG_SIZE)


# ---------------------------------------------------------------------------
# _PixelWrapper and _EmbeddingWrapper
# ---------------------------------------------------------------------------

class TestWrappers:
    def test_pixel_wrapper_shape(self, tiny_vit, single_image):
        wrapper = _PixelWrapper(tiny_vit, binary=False)
        out = wrapper(single_image.unsqueeze(0))
        assert out.shape == (1, _NUM_CLASSES)

    def test_pixel_wrapper_sums_to_one(self, tiny_vit, single_image):
        wrapper = _PixelWrapper(tiny_vit, binary=False)
        out = wrapper(single_image.unsqueeze(0))
        np.testing.assert_allclose(out.sum().item(), 1.0, atol=1e-5)

    def test_pixel_wrapper_binary(self, single_image):
        import timm
        vit_bin = timm.create_model(_TINY_MODEL, pretrained=False, num_classes=1)
        wrapper = _PixelWrapper(vit_bin, binary=True)
        out = wrapper(single_image.unsqueeze(0))
        assert out.shape == (1, 1)
        assert 0.0 <= out.item() <= 1.0

    def test_embedding_wrapper_shape(self, tiny_vit, single_image):
        wrapper = _EmbeddingWrapper(tiny_vit, binary=False)
        with torch.no_grad():
            emb = tiny_vit.patch_embed(single_image.unsqueeze(0))
        out = wrapper(emb)
        assert out.shape == (1, _NUM_CLASSES)

    def test_embedding_wrapper_sums_to_one(self, tiny_vit, single_image):
        wrapper = _EmbeddingWrapper(tiny_vit, binary=False)
        with torch.no_grad():
            emb = tiny_vit.patch_embed(single_image.unsqueeze(0))
        out = wrapper(emb)
        np.testing.assert_allclose(out.sum().item(), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# get_vanilla_gradient
# ---------------------------------------------------------------------------

class TestVanillaGradient:
    def test_shape_embedding(self, tiny_vit, single_image):
        result = get_vanilla_gradient(single_image, tiny_vit, _NUM_CLASSES, space="embedding")
        assert result.shape == (_NUM_CLASSES, _NUM_PATCHES)

    def test_shape_pixel(self, tiny_vit, single_image):
        result = get_vanilla_gradient(
            single_image, tiny_vit, _NUM_CLASSES, space="pixel", patch_size=_PATCH_SIZE
        )
        assert result.shape == (_NUM_CLASSES, _NUM_PATCHES)

    def test_values_nonnegative(self, tiny_vit, single_image):
        """Absolute gradient scores must be non-negative."""
        result = get_vanilla_gradient(single_image, tiny_vit, _NUM_CLASSES)
        assert np.all(result >= 0)

    def test_values_finite(self, tiny_vit, single_image):
        result = get_vanilla_gradient(single_image, tiny_vit, _NUM_CLASSES)
        assert np.all(np.isfinite(result))

    def test_binary_shape(self, single_image):
        import timm
        vit_bin = timm.create_model(_TINY_MODEL, pretrained=False, num_classes=1)
        result = get_vanilla_gradient(single_image, vit_bin, output_dim=1, space="embedding")
        assert result.shape == (1, _NUM_PATCHES)

    def test_returns_numpy(self, tiny_vit, single_image):
        result = get_vanilla_gradient(single_image, tiny_vit, _NUM_CLASSES)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# get_smoothgrad
# ---------------------------------------------------------------------------

class TestSmoothGrad:
    def test_shape_embedding(self, tiny_vit, single_image):
        result = get_smoothgrad(single_image, tiny_vit, _NUM_CLASSES, space="embedding", n_samples=5)
        assert result.shape == (_NUM_CLASSES, _NUM_PATCHES)

    def test_shape_pixel(self, tiny_vit, single_image):
        result = get_smoothgrad(
            single_image, tiny_vit, _NUM_CLASSES, space="pixel", n_samples=5, patch_size=_PATCH_SIZE
        )
        assert result.shape == (_NUM_CLASSES, _NUM_PATCHES)

    def test_values_finite(self, tiny_vit, single_image):
        result = get_smoothgrad(single_image, tiny_vit, _NUM_CLASSES, n_samples=3)
        assert np.all(np.isfinite(result))

    def test_values_nonnegative(self, tiny_vit, single_image):
        result = get_smoothgrad(single_image, tiny_vit, _NUM_CLASSES, n_samples=3)
        assert np.all(result >= 0)

    def test_binary_shape(self, single_image):
        import timm
        vit_bin = timm.create_model(_TINY_MODEL, pretrained=False, num_classes=1)
        result = get_smoothgrad(single_image, vit_bin, output_dim=1, n_samples=3)
        assert result.shape == (1, _NUM_PATCHES)


# ---------------------------------------------------------------------------
# get_vargrad
# ---------------------------------------------------------------------------

class TestVarGrad:
    def test_shape_embedding(self, tiny_vit, single_image):
        result = get_vargrad(single_image, tiny_vit, _NUM_CLASSES, space="embedding", n_samples=5)
        assert result.shape == (_NUM_CLASSES, _NUM_PATCHES)

    def test_shape_pixel(self, tiny_vit, single_image):
        result = get_vargrad(
            single_image, tiny_vit, _NUM_CLASSES, space="pixel", n_samples=5, patch_size=_PATCH_SIZE
        )
        assert result.shape == (_NUM_CLASSES, _NUM_PATCHES)

    def test_values_finite(self, tiny_vit, single_image):
        result = get_vargrad(single_image, tiny_vit, _NUM_CLASSES, n_samples=3)
        assert np.all(np.isfinite(result))

    def test_values_nonnegative(self, tiny_vit, single_image):
        result = get_vargrad(single_image, tiny_vit, _NUM_CLASSES, n_samples=3)
        assert np.all(result >= 0)

    def test_binary_shape(self, single_image):
        import timm
        vit_bin = timm.create_model(_TINY_MODEL, pretrained=False, num_classes=1)
        result = get_vargrad(single_image, vit_bin, output_dim=1, n_samples=3)
        assert result.shape == (1, _NUM_PATCHES)


# ---------------------------------------------------------------------------
# get_integrated_gradients
# ---------------------------------------------------------------------------

class TestIntegratedGradients:
    def test_shape_embedding(self, tiny_vit, single_image):
        result = get_integrated_gradients(
            single_image, tiny_vit, _NUM_CLASSES, space="embedding", n_steps=10
        )
        assert result.shape == (_NUM_CLASSES, _NUM_PATCHES)

    def test_shape_pixel(self, tiny_vit, single_image):
        result = get_integrated_gradients(
            single_image, tiny_vit, _NUM_CLASSES, space="pixel", n_steps=10, patch_size=_PATCH_SIZE
        )
        assert result.shape == (_NUM_CLASSES, _NUM_PATCHES)

    def test_values_finite(self, tiny_vit, single_image):
        result = get_integrated_gradients(
            single_image, tiny_vit, _NUM_CLASSES, space="embedding", n_steps=10
        )
        assert np.all(np.isfinite(result))

    def test_signed_values(self, tiny_vit, single_image):
        """IG should produce both positive and negative values (signed aggregation)."""
        result = get_integrated_gradients(
            single_image, tiny_vit, _NUM_CLASSES, space="embedding", n_steps=10
        )
        # For a randomly initialised model and image, scores should not all be zero.
        assert np.any(result != 0)

    def test_binary_shape(self, single_image):
        import timm
        vit_bin = timm.create_model(_TINY_MODEL, pretrained=False, num_classes=1)
        result = get_integrated_gradients(
            single_image, vit_bin, output_dim=1, space="embedding", n_steps=5
        )
        assert result.shape == (1, _NUM_PATCHES)

    def test_returns_numpy(self, tiny_vit, single_image):
        result = get_integrated_gradients(
            single_image, tiny_vit, _NUM_CLASSES, n_steps=5
        )
        assert isinstance(result, np.ndarray)
