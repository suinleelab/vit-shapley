"""Tests for vit_shapley.models.classifier.

Uses vit_tiny_patch16_224 (smallest ViT, fast) without pretrained weights
so tests run offline and quickly.
"""

import pytest
import torch

from vit_shapley.models.classifier import build_vit_classifier

# Use the smallest ViT for all tests
_TINY_MODEL = "vit_tiny_patch16_224"


class TestBuildVitClassifier:
    def test_default_output_shape(self):
        model = build_vit_classifier(_TINY_MODEL, num_classes=10, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)

    def test_custom_num_classes(self):
        for num_classes in (2, 5, 100, 1000):
            model = build_vit_classifier(
                _TINY_MODEL, num_classes=num_classes, pretrained=False
            )
            model.eval()
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (1, num_classes), f"Failed for num_classes={num_classes}"

    def test_returns_nn_module(self):
        import torch.nn as nn
        model = build_vit_classifier(_TINY_MODEL, num_classes=10, pretrained=False)
        assert isinstance(model, nn.Module)

    def test_dropout_zero(self):
        model = build_vit_classifier(
            _TINY_MODEL, num_classes=10, pretrained=False, dropout=0.0
        )
        assert model is not None

    def test_dropout_nonzero(self):
        model = build_vit_classifier(
            _TINY_MODEL, num_classes=10, pretrained=False, dropout=0.5
        )
        assert model is not None

    def test_model_is_trainable(self):
        model = build_vit_classifier(_TINY_MODEL, num_classes=10, pretrained=False)
        # All parameters should have requires_grad=True by default
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable) > 0

    def test_batch_independence(self):
        """Different batch sizes should give same per-sample outputs."""
        model = build_vit_classifier(_TINY_MODEL, num_classes=5, pretrained=False)
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out_batch = model(x)
            out_single = torch.stack([model(x[i : i + 1]) for i in range(4)]).squeeze(1)
        # Outputs should be close (within float32 tolerance)
        assert torch.allclose(out_batch, out_single, atol=1e-5)

    def test_invalid_model_name_raises(self):
        with pytest.raises(Exception):
            build_vit_classifier("not_a_real_model_xyz", num_classes=10, pretrained=False)
