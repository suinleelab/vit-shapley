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


def _tiny_binary_classifier(num_patches: int = 9) -> nn.Module:
    """Minimal binary classifier: outputs (B, 1) logit."""
    return _tiny_classifier(num_patches=num_patches, num_classes=1)


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

    def test_generator_reproducibility(self):
        """Same generator seed produces identical masks."""
        g1 = torch.Generator().manual_seed(99)
        g2 = torch.Generator().manual_seed(99)
        m1 = sample_fixed_cardinality_masks(self.B, self.P, 4, generator=g1)
        m2 = sample_fixed_cardinality_masks(self.B, self.P, 4, generator=g2)
        assert torch.equal(m1, m2)

    def test_generator_differs_from_no_generator(self):
        """Generator-based masks differ from two ungoverned calls (with high
        probability), confirming the generator actually controls randomness."""
        g = torch.Generator().manual_seed(123)
        m_seeded = sample_fixed_cardinality_masks(self.B, self.P, 4, generator=g)
        # Two unseeded calls should (almost certainly) not match.
        m_a = sample_fixed_cardinality_masks(self.B, self.P, 4)
        m_b = sample_fixed_cardinality_masks(self.B, self.P, 4)
        # At least one pair should differ.
        assert not torch.equal(m_a, m_b) or not torch.equal(m_seeded, m_a)

    def test_generator_sequential_calls_differ(self):
        """Two calls with the *same* generator object yield different masks
        (the generator state advances)."""
        g = torch.Generator().manual_seed(0)
        m1 = sample_fixed_cardinality_masks(self.B, self.P, 4, generator=g)
        m2 = sample_fixed_cardinality_masks(self.B, self.P, 4, generator=g)
        assert not torch.equal(m1, m2)


# ---------------------------------------------------------------------------
# TestComputeKlVsCardinality  (7 + seed tests)
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

    # ---- seed reproducibility tests ----------------------------------------

    def test_seed_produces_identical_results(self, clf, images):
        """Two calls with the same seed must give exactly the same KL values."""
        surrogate = _IdentitySurrogate(clf)
        surrogate.eval()
        kwargs = dict(
            surrogate=surrogate,
            classifier=clf,
            images=images,
            num_patches=self.NUM_PATCHES,
            num_masks_per_cardinality=self.K,
            cardinality_step=3,
            device=torch.device("cpu"),
        )
        r1 = compute_kl_vs_cardinality(**kwargs, seed=42)
        r2 = compute_kl_vs_cardinality(**kwargs, seed=42)
        assert r1.keys() == r2.keys()
        for m in r1:
            for v1, v2 in zip(r1[m], r2[m]):
                assert math.isclose(v1, v2, abs_tol=1e-7), (
                    f"Mismatch at cardinality {m}: {v1} vs {v2}"
                )

    def test_different_seeds_give_different_results(self, clf, images):
        """Different seeds should (with overwhelming probability) yield
        different KL values at non-trivial cardinalities."""
        num_p = self.NUM_PATCHES

        class _NoisySurrogate(nn.Module):
            """Returns different logits depending on the mask content."""

            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(num_p, 4)

            def forward(self, x, patch_mask=None):
                # Mix mask into output so mask differences affect KL.
                return self.fc(patch_mask)

        noisy = _NoisySurrogate()
        noisy.eval()
        kwargs = dict(
            surrogate=noisy,
            classifier=clf,
            images=images,
            num_patches=num_p,
            num_masks_per_cardinality=self.K,
            cardinality_step=3,
            device=torch.device("cpu"),
        )
        r1 = compute_kl_vs_cardinality(**kwargs, seed=1)
        r2 = compute_kl_vs_cardinality(**kwargs, seed=2)
        # At cardinalities 0 and num_patches masks are deterministic (all-ones
        # / all-zeros), so only intermediate cardinalities can differ.
        any_diff = any(
            not math.isclose(v1, v2, abs_tol=1e-7)
            for m in r1
            for v1, v2 in zip(r1[m], r2[m])
        )
        assert any_diff, "Different seeds produced identical results"

    def test_seed_independent_of_global_rng(self, clf, images):
        """Seeded calls should not be affected by global torch RNG state."""
        surrogate = _IdentitySurrogate(clf)
        surrogate.eval()
        kwargs = dict(
            surrogate=surrogate,
            classifier=clf,
            images=images,
            num_patches=self.NUM_PATCHES,
            num_masks_per_cardinality=self.K,
            cardinality_step=3,
            device=torch.device("cpu"),
            seed=7,
        )
        torch.manual_seed(0)
        r1 = compute_kl_vs_cardinality(**kwargs)
        torch.manual_seed(999)
        r2 = compute_kl_vs_cardinality(**kwargs)
        for m in r1:
            for v1, v2 in zip(r1[m], r2[m]):
                assert math.isclose(v1, v2, abs_tol=1e-7)

    def test_same_seed_different_surrogates_use_same_masks(self, clf, images):
        """Two *different* surrogates called with the same seed must see the
        exact same mask sequence—verified by a mask-recording surrogate."""

        recorded_masks: list[list[torch.Tensor]] = [[], []]

        class _RecordingSurrogate(nn.Module):
            def __init__(self, idx, inner_clf):
                super().__init__()
                self._idx = idx
                self._clf = inner_clf

            def forward(self, x, patch_mask=None):
                recorded_masks[self._idx].append(patch_mask.clone())
                return self._clf(x)

        s0 = _RecordingSurrogate(0, clf)
        s1 = _RecordingSurrogate(1, clf)
        s0.eval()
        s1.eval()

        kwargs = dict(
            classifier=clf,
            images=images,
            num_patches=self.NUM_PATCHES,
            num_masks_per_cardinality=self.K,
            cardinality_step=3,
            device=torch.device("cpu"),
            seed=123,
        )
        compute_kl_vs_cardinality(surrogate=s0, **kwargs)
        compute_kl_vs_cardinality(surrogate=s1, **kwargs)

        assert len(recorded_masks[0]) == len(recorded_masks[1])
        for m0, m1 in zip(recorded_masks[0], recorded_masks[1]):
            assert torch.equal(m0, m1), "Masks differ between surrogates!"


# ---------------------------------------------------------------------------
# Binary KL evaluation tests
# ---------------------------------------------------------------------------


class TestBinaryComputeKlVsCardinality:
    """Tests for compute_kl_vs_cardinality with target_type='binary'."""

    NUM_PATCHES = 9
    N = 3
    K = 2

    @pytest.fixture()
    def clf(self):
        m = _tiny_binary_classifier(self.NUM_PATCHES)
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

    def test_contains_zero_and_num_patches(self, surrogate, clf, images):
        results = compute_kl_vs_cardinality(
            surrogate=surrogate, classifier=clf, images=images,
            num_patches=self.NUM_PATCHES, num_masks_per_cardinality=self.K,
            cardinality_step=3, device=torch.device("cpu"),
            target_type="binary",
        )
        assert 0 in results
        assert self.NUM_PATCHES in results

    def test_values_nonnegative(self, surrogate, clf, images):
        results = compute_kl_vs_cardinality(
            surrogate=surrogate, classifier=clf, images=images,
            num_patches=self.NUM_PATCHES, num_masks_per_cardinality=self.K,
            cardinality_step=3, device=torch.device("cpu"),
            target_type="binary",
        )
        for vals in results.values():
            assert all(v >= -1e-5 for v in vals)

    def test_kl_near_zero_when_identical(self, clf, images):
        """If surrogate == classifier (identity), binary KL should be ~0."""
        surrogate = _IdentitySurrogate(clf)
        surrogate.eval()
        results = compute_kl_vs_cardinality(
            surrogate=surrogate, classifier=clf, images=images,
            num_patches=self.NUM_PATCHES, num_masks_per_cardinality=self.K,
            cardinality_step=self.NUM_PATCHES,
            device=torch.device("cpu"),
            target_type="binary",
        )
        kl_at_zero = results[0]
        assert all(math.isclose(v, 0.0, abs_tol=1e-5) for v in kl_at_zero), (
            f"Expected binary KL ≈ 0 at cardinality 0, got {kl_at_zero}"
        )

    def test_seed_reproducibility(self, surrogate, clf, images):
        kwargs = dict(
            surrogate=surrogate, classifier=clf, images=images,
            num_patches=self.NUM_PATCHES, num_masks_per_cardinality=self.K,
            cardinality_step=3, device=torch.device("cpu"),
            target_type="binary",
        )
        r1 = compute_kl_vs_cardinality(**kwargs, seed=42)
        r2 = compute_kl_vs_cardinality(**kwargs, seed=42)
        assert r1.keys() == r2.keys()
        for m in r1:
            for v1, v2 in zip(r1[m], r2[m]):
                assert math.isclose(v1, v2, abs_tol=1e-7)
