"""Tests for src/vit_shapley/evaluation/kl_vs_cardinality.py."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from vit_shapley.evaluation.kl_vs_cardinality import (
    compute_kl_vs_cardinality,
    sample_fixed_cardinality_masks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _IdentitySurrogate(nn.Module):
    """Wraps a classifier and ignores the patch_mask argument."""

    def __init__(self, classifier: nn.Module) -> None:
        super().__init__()
        self.vit = classifier  # expose .vit.patch_embed.num_patches for script compat
        self._clf = classifier

    def forward(
        self, x: torch.Tensor, patch_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self._clf(x)


def _tiny_classifier(num_patches: int = 9, num_classes: int = 4) -> nn.Module:
    """Minimal classifier: linear map from flattened image to logits."""

    class _FlatClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(3, num_classes)
            # Mimic timm's .patch_embed.num_patches for script compatibility.
            self.patch_embed = type("_PE", (), {"num_patches": num_patches})()

        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # (B,C,H,W)
            return self.fc(x.mean(dim=[2, 3]))

    return _FlatClassifier()


# ---------------------------------------------------------------------------
# TestSampleFixedCardinalityMasks  (8 tests)
# ---------------------------------------------------------------------------


class TestSampleFixedCardinalityMasks:
    B, P = 8, 16

    def test_output_shape(self):
        masks = sample_fixed_cardinality_masks(self.B, self.P, num_masked=4)
        assert masks.shape == (self.B, self.P)

    def test_values_binary(self):
        masks = sample_fixed_cardinality_masks(self.B, self.P, num_masked=4)
        unique = masks.unique().tolist()
        assert set(unique).issubset({0.0, 1.0})

    def test_exact_zero_count(self):
        num_masked = 6
        masks = sample_fixed_cardinality_masks(self.B, self.P, num_masked=num_masked)
        zeros_per_row = (masks == 0.0).sum(dim=1)
        assert (zeros_per_row == num_masked).all()

    def test_num_masked_zero_gives_all_ones(self):
        masks = sample_fixed_cardinality_masks(self.B, self.P, num_masked=0)
        assert masks.eq(1.0).all()

    def test_num_masked_full_gives_all_zeros(self):
        masks = sample_fixed_cardinality_masks(self.B, self.P, num_masked=self.P)
        assert masks.eq(0.0).all()

    def test_rows_differ_across_samples(self):
        """With high probability, 8 random rows won't all be identical."""
        masks = sample_fixed_cardinality_masks(self.B, self.P, num_masked=4)
        # At least two rows should differ.
        assert not (masks[0] == masks).all(dim=1).all()

    @pytest.mark.parametrize("num_masked", [-1, 17])
    def test_invalid_num_masked_raises(self, num_masked):
        with pytest.raises(ValueError):
            sample_fixed_cardinality_masks(self.B, self.P, num_masked=num_masked)


# ---------------------------------------------------------------------------
# TestComputeKlVsCardinality  (7 tests)
# ---------------------------------------------------------------------------


class TestComputeKlVsCardinality:
    """Tests for compute_kl_vs_cardinality."""

    NUM_PATCHES = 9
    NUM_CLASSES = 4
    N = 3  # images in the batch
    K = 2  # masks per cardinality

    @pytest.fixture()
    def clf(self):
        m = _tiny_classifier(self.NUM_PATCHES, self.NUM_CLASSES)
        m.eval()
        return m

    @pytest.fixture()
    def surrogate(self, clf):
        s = _IdentitySurrogate(clf)
        s.eval()
        return s

    @pytest.fixture()
    def images(self):
        return torch.randn(self.N, 3, 12, 12)

    def _run(self, surrogate, clf, images, step=3):
        return compute_kl_vs_cardinality(
            surrogate=surrogate,
            classifier=clf,
            images=images,
            num_patches=self.NUM_PATCHES,
            num_masks_per_cardinality=self.K,
            cardinality_step=step,
            device=torch.device("cpu"),
        )

    def test_contains_zero_and_num_patches(self, surrogate, clf, images):
        results = self._run(surrogate, clf, images)
        assert 0 in results
        assert self.NUM_PATCHES in results

    def test_cardinality_step_respected(self, surrogate, clf, images):
        step = 3
        results = self._run(surrogate, clf, images, step=step)
        keys = sorted(results.keys())
        # All keys should be multiples of step, plus num_patches.
        for k in keys:
            assert k % step == 0 or k == self.NUM_PATCHES

    def test_values_nonnegative(self, surrogate, clf, images):
        # KL divergence is theoretically ≥ 0; allow a tiny negative tolerance
        # for floating-point rounding (softmax→log vs log_softmax differ by ~1e-7).
        results = self._run(surrogate, clf, images)
        for vals in results.values():
            assert all(v >= -1e-5 for v in vals)

    def test_list_length_per_cardinality(self, surrogate, clf, images):
        results = self._run(surrogate, clf, images)
        expected = self.N * self.K
        for vals in results.values():
            assert len(vals) == expected

    def test_no_grad_updates(self, surrogate, clf, images):
        """Models should not be modified during evaluation."""
        params_before = {n: p.clone() for n, p in surrogate.named_parameters()}
        self._run(surrogate, clf, images)
        for n, p in surrogate.named_parameters():
            assert torch.equal(p, params_before[n]), f"Parameter {n} changed!"

    def test_kl_near_zero_when_identical_weights(self, clf, images):
        """If surrogate == classifier (identity surrogate), KL should be ~0."""
        surrogate = _IdentitySurrogate(clf)
        surrogate.eval()
        results = compute_kl_vs_cardinality(
            surrogate=surrogate,
            classifier=clf,
            images=images,
            num_patches=self.NUM_PATCHES,
            num_masks_per_cardinality=self.K,
            cardinality_step=self.NUM_PATCHES,  # only 0 and num_patches
            device=torch.device("cpu"),
        )
        # At cardinality 0, surrogate sees full image → KL should be 0.
        kl_at_zero = results[0]
        assert all(math.isclose(v, 0.0, abs_tol=1e-5) for v in kl_at_zero), (
            f"Expected KL ≈ 0 at cardinality 0, got {kl_at_zero}"
        )
