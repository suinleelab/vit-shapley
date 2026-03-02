"""Tests for vit_shapley.baselines.utils.

All tests are pure-numpy; no torch or model dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

from vit_shapley.baselines.utils import (
    explanation_to_mask,
    generate_mask,
    get_random_explanation,
    get_relative_value,
)

_P = 16  # small num_players for speed


# ---------------------------------------------------------------------------
# generate_mask
# ---------------------------------------------------------------------------

class TestGenerateMask:
    def test_shape_single(self):
        mask = generate_mask(_P, num_mask_samples=None)
        assert mask.shape == (_P,)

    def test_shape_batch(self):
        mask = generate_mask(_P, num_mask_samples=10, paired_mask_samples=False)
        assert mask.shape == (10, _P)

    def test_values_binary(self):
        mask = generate_mask(_P, num_mask_samples=20, paired_mask_samples=False)
        assert set(np.unique(mask)).issubset({0, 1})

    def test_paired_shape(self):
        mask = generate_mask(_P, num_mask_samples=8, paired_mask_samples=True)
        assert mask.shape == (8, _P)

    def test_paired_complements(self):
        rng = np.random.RandomState(0)
        mask = generate_mask(_P, num_mask_samples=8, paired_mask_samples=True, random_state=rng)
        # Each pair (2i, 2i+1) should be complementary.
        for i in range(4):
            np.testing.assert_array_equal(mask[2 * i] + mask[2 * i + 1], np.ones(_P, dtype=int))

    def test_paired_odd_raises(self):
        with pytest.raises(ValueError, match="even"):
            generate_mask(_P, num_mask_samples=3, paired_mask_samples=True)

    def test_mode_uniform_default(self):
        rng = np.random.RandomState(1)
        mask = generate_mask(_P, num_mask_samples=100, paired_mask_samples=False, mode="uniform", random_state=rng)
        assert mask.shape == (100, _P)

    def test_mode_shapley(self):
        rng = np.random.RandomState(2)
        mask = generate_mask(_P, num_mask_samples=100, paired_mask_samples=False, mode="shapley", random_state=rng)
        assert set(np.unique(mask)).issubset({0, 1})

    def test_mode_unknown_raises(self):
        with pytest.raises(ValueError):
            generate_mask(_P, num_mask_samples=4, mode="bad_mode")

    def test_reproducibility(self):
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        m1 = generate_mask(_P, num_mask_samples=10, random_state=rng1)
        m2 = generate_mask(_P, num_mask_samples=10, random_state=rng2)
        np.testing.assert_array_equal(m1, m2)


# ---------------------------------------------------------------------------
# get_random_explanation
# ---------------------------------------------------------------------------

class TestGetRandomExplanation:
    def test_shape_1d(self):
        x = get_random_explanation(_P)
        assert x.shape == (_P,)

    def test_shape_2d(self):
        x = get_random_explanation(_P, num_samples=5)
        assert x.shape == (5, _P)

    def test_near_zero(self):
        x = get_random_explanation(_P)
        assert np.all(np.abs(x) < 1e-35)

    def test_seeded_reproducibility(self):
        x1 = get_random_explanation(_P, random_seed=0)
        x2 = get_random_explanation(_P, random_seed=0)
        np.testing.assert_array_equal(x1, x2)

    def test_different_seeds_differ(self):
        x1 = get_random_explanation(_P, random_seed=0)
        x2 = get_random_explanation(_P, random_seed=1)
        assert not np.array_equal(x1, x2)


# ---------------------------------------------------------------------------
# get_relative_value
# ---------------------------------------------------------------------------

class TestGetRelativeValue:
    def test_shape(self):
        x = np.array([3.0, 1.0, 2.0])
        r = get_relative_value(x)
        assert r.shape == (3,)

    def test_known_order(self):
        # [3, 1, 2] → ranks should be [2, 0, 1] (ascending)
        x = np.array([3.0, 1.0, 2.0])
        r = get_relative_value(x, random_seed=0)
        assert r[1] < r[2] < r[0]

    def test_permutation_of_0_to_P(self):
        x = np.random.default_rng(7).random(_P)
        r = get_relative_value(x, random_seed=7)
        assert set(r.tolist()) == set(range(_P))

    def test_seeded_tie_breaking_reproducible(self):
        x = np.ones(_P)  # all ties
        r1 = get_relative_value(x, random_seed=3)
        r2 = get_relative_value(x, random_seed=3)
        np.testing.assert_array_equal(r1, r2)

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError):
            get_relative_value(np.ones((3, 4)))


# ---------------------------------------------------------------------------
# explanation_to_mask
# ---------------------------------------------------------------------------

class TestExplanationToMask:
    def _make_explanation(self, B=2, P=6):
        rng = np.random.default_rng(0)
        return rng.random((B, P))

    def test_output_shape(self):
        B, P = 3, _P
        exp = self._make_explanation(B, P)
        mask = explanation_to_mask(exp, mode="insertion")
        assert mask.shape == (B, P + 1, P)

    def test_output_shape_deletion(self):
        B, P = 3, _P
        exp = self._make_explanation(B, P)
        mask = explanation_to_mask(exp, mode="deletion")
        assert mask.shape == (B, P + 1, P)

    def test_insertion_starts_all_masked(self):
        exp = self._make_explanation()
        mask = explanation_to_mask(exp, mode="insertion")
        assert not mask[:, 0, :].any(), "Step 0 of insertion should be all-masked"

    def test_insertion_ends_all_visible(self):
        exp = self._make_explanation()
        mask = explanation_to_mask(exp, mode="insertion")
        assert mask[:, -1, :].all(), "Final step of insertion should be all-visible"

    def test_deletion_starts_all_visible(self):
        exp = self._make_explanation()
        mask = explanation_to_mask(exp, mode="deletion")
        assert mask[:, 0, :].all(), "Step 0 of deletion should be all-visible"

    def test_deletion_ends_all_masked(self):
        exp = self._make_explanation()
        mask = explanation_to_mask(exp, mode="deletion")
        assert not mask[:, -1, :].any(), "Final step of deletion should be all-masked"

    def test_insertion_monotone_visible_count(self):
        exp = self._make_explanation(B=1, P=8)
        mask = explanation_to_mask(exp, mode="insertion")
        counts = mask[0].sum(axis=-1)  # (P+1,)
        assert np.all(np.diff(counts) >= 0), "Insertion visible count must be non-decreasing"

    def test_deletion_monotone_visible_count(self):
        exp = self._make_explanation(B=1, P=8)
        mask = explanation_to_mask(exp, mode="deletion")
        counts = mask[0].sum(axis=-1)  # (P+1,)
        assert np.all(np.diff(counts) <= 0), "Deletion visible count must be non-increasing"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            explanation_to_mask(self._make_explanation(), mode="bad")
