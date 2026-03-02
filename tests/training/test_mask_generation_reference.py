"""Tests verifying mask generation consistency with the reference implementation.

The reference implementation (ref/vit-shapley/main.py ``generate_mask``) uses
a Bernoulli-threshold approach for both uniform and Shapley mode mask sampling.
These tests verify that our ``sample_subset_masks`` and ``sample_shapley_masks``
produce statistically equivalent outputs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import stats

from vit_shapley.training.train_explainer import sample_shapley_masks
from vit_shapley.training.train_surrogate import sample_subset_masks


# ── Reference implementation (copied from ref/vit-shapley/main.py) ───────────

def _ref_generate_mask(
    num_players: int,
    num_mask_samples: int | None = None,
    paired_mask_samples: bool = True,
    mode: str = "uniform",
    random_state: np.random.RandomState | None = None,
) -> np.ndarray:
    """Reference generate_mask verbatim from ref/vit-shapley/main.py."""
    random_state = random_state or np.random

    num_samples_ = num_mask_samples or 1

    if paired_mask_samples:
        assert num_samples_ % 2 == 0
        num_samples_ = num_samples_ // 2
    else:
        num_samples_ = num_samples_

    if mode == "uniform":
        masks = (
            random_state.rand(num_samples_, num_players)
            > random_state.rand(num_samples_, 1)
        ).astype("int")
    elif mode == "shapley":
        probs = 1 / (
            np.arange(1, num_players)
            * (num_players - np.arange(1, num_players))
        )
        probs = probs / probs.sum()
        masks = (
            random_state.rand(num_samples_, num_players)
            > 1
            / num_players
            * random_state.choice(
                np.arange(num_players - 1),
                p=probs,
                size=[num_samples_, 1],
            )
        ).astype("int")
    else:
        raise ValueError("'mode' must be 'uniform' or 'shapley'")

    if paired_mask_samples:
        masks = np.stack([masks, 1 - masks], axis=1).reshape(
            num_samples_ * 2, num_players
        )

    if num_mask_samples is None:
        masks = masks.squeeze(0)
        return masks
    else:
        return masks


# ── Constants ─────────────────────────────────────────────────────────────────

NUM_PLAYERS = 14  # small for fast tests (like a 14-patch grid, not 196)
NUM_SAMPLES_LARGE = 50_000  # enough for statistical tests
CHI2_ALPHA = 0.001  # significance level for chi-squared tests
KS_ALPHA = 0.001  # significance level for KS tests


# ===========================================================================
# Part 1: sample_subset_masks (uniform mode)
# ===========================================================================


class TestSampleSubsetMasksVsReference:
    """Verify sample_subset_masks is statistically equivalent to
    generate_mask(mode='uniform')."""

    def test_output_shape(self):
        masks = sample_subset_masks(10, NUM_PLAYERS, torch.device("cpu"))
        assert masks.shape == (10, NUM_PLAYERS)

    def test_binary_values(self):
        masks = sample_subset_masks(100, NUM_PLAYERS, torch.device("cpu"))
        assert torch.all((masks == 0.0) | (masks == 1.0))

    def test_cardinality_distribution_uniform(self):
        """Both implementations should produce uniform cardinality over
        {0, 1, ..., n}."""
        n = NUM_PLAYERS

        # Current implementation
        masks_cur = sample_subset_masks(
            NUM_SAMPLES_LARGE, n, torch.device("cpu"),
        )
        cards_cur = masks_cur.sum(dim=1).long().numpy()

        # Reference implementation
        rs = np.random.RandomState(42)
        masks_ref = _ref_generate_mask(
            n, num_mask_samples=NUM_SAMPLES_LARGE,
            paired_mask_samples=False, mode="uniform", random_state=rs,
        )
        cards_ref = masks_ref.sum(axis=1)

        # Both should be approximately uniform over {0, ..., n}
        expected_freq = NUM_SAMPLES_LARGE / (n + 1)

        # Chi-squared test for current
        observed_cur = np.bincount(cards_cur, minlength=n + 1)
        chi2_cur, p_cur = stats.chisquare(
            observed_cur, f_exp=[expected_freq] * (n + 1),
        )
        assert p_cur > CHI2_ALPHA, (
            f"Current cardinality not uniform: chi2={chi2_cur:.1f}, p={p_cur:.4f}"
        )

        # Chi-squared test for reference
        observed_ref = np.bincount(cards_ref.astype(int), minlength=n + 1)
        chi2_ref, p_ref = stats.chisquare(
            observed_ref, f_exp=[expected_freq] * (n + 1),
        )
        assert p_ref > CHI2_ALPHA, (
            f"Reference cardinality not uniform: chi2={chi2_ref:.1f}, p={p_ref:.4f}"
        )

    def test_cardinality_distributions_match(self):
        """KS test between cardinality distributions of both implementations."""
        n = NUM_PLAYERS

        masks_cur = sample_subset_masks(
            NUM_SAMPLES_LARGE, n, torch.device("cpu"),
        )
        cards_cur = masks_cur.sum(dim=1).numpy()

        rs = np.random.RandomState(123)
        masks_ref = _ref_generate_mask(
            n, num_mask_samples=NUM_SAMPLES_LARGE,
            paired_mask_samples=False, mode="uniform", random_state=rs,
        )
        cards_ref = masks_ref.sum(axis=1).astype(float)

        ks_stat, p_val = stats.ks_2samp(cards_cur, cards_ref)
        assert p_val > KS_ALPHA, (
            f"Cardinality distributions differ: KS={ks_stat:.4f}, p={p_val:.4f}"
        )

    def test_marginal_inclusion_rate(self):
        """Each patch should be included ~50% of the time (marginal rate)."""
        n = NUM_PLAYERS

        masks_cur = sample_subset_masks(
            NUM_SAMPLES_LARGE, n, torch.device("cpu"),
        )
        inclusion_rates = masks_cur.mean(dim=0).numpy()
        # Under uniform cardinality, E[inclusion] = 0.5
        np.testing.assert_allclose(inclusion_rates, 0.5, atol=0.02)

        rs = np.random.RandomState(0)
        masks_ref = _ref_generate_mask(
            n, num_mask_samples=NUM_SAMPLES_LARGE,
            paired_mask_samples=False, mode="uniform", random_state=rs,
        )
        inclusion_rates_ref = masks_ref.mean(axis=0)
        np.testing.assert_allclose(inclusion_rates_ref, 0.5, atol=0.02)

    def test_within_cardinality_correlation(self):
        """In the Bernoulli-threshold method, patches within a single mask
        share the same threshold, making them positively correlated.  Verify
        this positive correlation exists (it would NOT exist with the
        randperm approach)."""
        n = NUM_PLAYERS

        masks = sample_subset_masks(
            NUM_SAMPLES_LARGE, n, torch.device("cpu"),
        )
        # Compute pairwise correlation between patch 0 and patch 1
        p0 = masks[:, 0].numpy()
        p1 = masks[:, 1].numpy()
        corr, _ = stats.pearsonr(p0, p1)
        # Bernoulli-threshold gives positive correlation ~1/3
        assert corr > 0.2, (
            f"Patches should be positively correlated (Bernoulli threshold); "
            f"got corr={corr:.4f}"
        )

    def test_reproducible_with_generator(self):
        n = NUM_PLAYERS
        g1 = torch.Generator().manual_seed(99)
        g2 = torch.Generator().manual_seed(99)
        m1 = sample_subset_masks(10, n, torch.device("cpu"), generator=g1)
        m2 = sample_subset_masks(10, n, torch.device("cpu"), generator=g2)
        assert torch.equal(m1, m2)


# ===========================================================================
# Part 2: sample_shapley_masks (shapley mode)
# ===========================================================================


class TestSampleShapleyMasksVsReference:
    """Verify sample_shapley_masks is statistically equivalent to
    generate_mask(mode='shapley')."""

    def test_output_shape_unpaired(self):
        masks = sample_shapley_masks(
            4, NUM_PLAYERS, 8, paired=False,
        )
        assert masks.shape == (4, 8, NUM_PLAYERS)

    def test_output_shape_paired(self):
        masks = sample_shapley_masks(
            4, NUM_PLAYERS, 8, paired=True,
        )
        assert masks.shape == (4, 8, NUM_PLAYERS)

    def test_binary_values(self):
        masks = sample_shapley_masks(
            10, NUM_PLAYERS, 8, paired=True,
        )
        assert torch.all((masks == 0.0) | (masks == 1.0))

    def test_paired_complement(self):
        """Second half of masks should be complement of first half."""
        masks = sample_shapley_masks(
            4, NUM_PLAYERS, 8, paired=True,
        )
        first_half = masks[:, :4, :]
        second_half = masks[:, 4:, :]
        assert torch.allclose(first_half + second_half, torch.ones_like(first_half))

    def test_shapley_cardinality_distribution(self):
        """Both implementations should produce cardinalities drawn from
        the Shapley kernel distribution ∝ 1/(k(n-k))."""
        n = NUM_PLAYERS
        num_masks = NUM_SAMPLES_LARGE

        # Current implementation (unpaired to avoid pairing effects)
        masks_cur = sample_shapley_masks(
            1, n, num_masks, paired=False,
        ).squeeze(0)  # (num_masks, n)
        cards_cur = masks_cur.sum(dim=1).numpy()

        # Reference implementation
        rs = np.random.RandomState(42)
        masks_ref = _ref_generate_mask(
            n, num_mask_samples=num_masks,
            paired_mask_samples=False, mode="shapley", random_state=rs,
        )
        cards_ref = masks_ref.sum(axis=1).astype(float)

        # Expected Shapley distribution over cardinalities
        ks_arr = np.arange(1, n)
        expected_probs = 1.0 / (ks_arr * (n - ks_arr))
        expected_probs = expected_probs / expected_probs.sum()

        # NOTE: Both implementations use a Bernoulli threshold approach,
        # so the *observed* cardinality is stochastic and won't exactly match
        # the target distribution. Instead, we check that the two
        # implementations produce the *same* cardinality distribution.
        ks_stat, p_val = stats.ks_2samp(cards_cur, cards_ref)
        assert p_val > KS_ALPHA, (
            f"Cardinality distributions differ: KS={ks_stat:.4f}, p={p_val:.4f}"
        )

    def test_extreme_cardinality_rate_matches_reference(self):
        """The Bernoulli-threshold method produces some all-0 and all-1 masks
        because the Shapley distribution has high weight near k=1 and k=n-1,
        and the threshold c/n can be very small (→ all included) or large
        (→ all excluded).  Verify the rates are consistent between the
        current and reference implementations."""
        n = NUM_PLAYERS

        # Current
        masks_cur = sample_shapley_masks(
            1, n, NUM_SAMPLES_LARGE, paired=False,
        ).squeeze(0)
        cards_cur = masks_cur.sum(dim=1)
        frac_empty_cur = (cards_cur == 0).float().mean().item()
        frac_full_cur = (cards_cur == n).float().mean().item()

        # Reference
        rs = np.random.RandomState(42)
        masks_ref = _ref_generate_mask(
            n, num_mask_samples=NUM_SAMPLES_LARGE,
            paired_mask_samples=False, mode="shapley", random_state=rs,
        )
        cards_ref = masks_ref.sum(axis=1)
        frac_empty_ref = (cards_ref == 0).mean()
        frac_full_ref = (cards_ref == n).mean()

        # Both should have similar rates of extreme cardinalities
        assert abs(frac_empty_cur - frac_empty_ref) < 0.02, (
            f"Empty mask rates differ: cur={frac_empty_cur:.3f} "
            f"ref={frac_empty_ref:.3f}"
        )
        assert abs(frac_full_cur - frac_full_ref) < 0.02, (
            f"Full mask rates differ: cur={frac_full_cur:.3f} "
            f"ref={frac_full_ref:.3f}"
        )

    def test_mean_cardinality_matches(self):
        """Mean cardinality should be close between the two implementations."""
        n = NUM_PLAYERS

        masks_cur = sample_shapley_masks(
            1, n, NUM_SAMPLES_LARGE, paired=False,
        ).squeeze(0)
        mean_cur = masks_cur.sum(dim=1).mean().item()

        rs = np.random.RandomState(0)
        masks_ref = _ref_generate_mask(
            n, num_mask_samples=NUM_SAMPLES_LARGE,
            paired_mask_samples=False, mode="shapley", random_state=rs,
        )
        mean_ref = masks_ref.sum(axis=1).mean()

        # Should be close (both centered around n/2 for Shapley weights)
        assert abs(mean_cur - mean_ref) < 0.2, (
            f"Mean cardinality differs: current={mean_cur:.2f}, "
            f"ref={mean_ref:.2f}"
        )

    def test_within_cardinality_correlation_shapley(self):
        """Bernoulli-threshold method produces positively correlated patches
        within each mask (shared threshold)."""
        n = NUM_PLAYERS

        masks = sample_shapley_masks(
            1, n, NUM_SAMPLES_LARGE, paired=False,
        ).squeeze(0)
        p0 = masks[:, 0].numpy()
        p1 = masks[:, 1].numpy()
        corr, _ = stats.pearsonr(p0, p1)
        assert corr > 0.1, (
            f"Patches should be positively correlated; got corr={corr:.4f}"
        )

    def test_marginal_inclusion_rates_match(self):
        """Per-patch marginal inclusion rates should be similar between
        the two implementations."""
        n = NUM_PLAYERS

        masks_cur = sample_shapley_masks(
            1, n, NUM_SAMPLES_LARGE, paired=False,
        ).squeeze(0)
        rates_cur = masks_cur.mean(dim=0).numpy()

        rs = np.random.RandomState(7)
        masks_ref = _ref_generate_mask(
            n, num_mask_samples=NUM_SAMPLES_LARGE,
            paired_mask_samples=False, mode="shapley", random_state=rs,
        )
        rates_ref = masks_ref.mean(axis=0)

        # All patches should have similar marginal rates (~0.5 for symmetric
        # Shapley weights), and the two implementations should agree.
        np.testing.assert_allclose(rates_cur, rates_ref, atol=0.03)

    def test_paired_cardinality_sum(self):
        """For paired masks, S and 1-S should sum to n for each pair."""
        n = NUM_PLAYERS
        M = 8
        masks = sample_shapley_masks(4, n, M, paired=True)
        first = masks[:, : M // 2, :]
        second = masks[:, M // 2 :, :]
        cards_sum = first.sum(dim=2) + second.sum(dim=2)
        assert torch.all(cards_sum == n)

    def test_reproducible_with_generator(self):
        n = NUM_PLAYERS
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        m1 = sample_shapley_masks(4, n, 8, paired=True, generator=g1)
        m2 = sample_shapley_masks(4, n, 8, paired=True, generator=g2)
        assert torch.equal(m1, m2)

    def test_odd_num_mask_samples_raises_when_paired(self):
        with pytest.raises(ValueError, match="even"):
            sample_shapley_masks(2, NUM_PLAYERS, 7, paired=True)


# ===========================================================================
# Part 3: Edge cases
# ===========================================================================


class TestMaskGenerationEdgeCases:
    """Edge cases for both mask generation functions."""

    def test_subset_masks_single_patch(self):
        """With 1 patch, masks should be 0 or 1."""
        masks = sample_subset_masks(1000, 1, torch.device("cpu"))
        assert masks.shape == (1000, 1)
        assert torch.all((masks == 0.0) | (masks == 1.0))
        # Should get roughly 50% each
        frac_ones = masks.mean().item()
        assert 0.4 < frac_ones < 0.6

    def test_subset_masks_two_patches(self):
        """With 2 patches, cardinalities should be uniform over {0,1,2}."""
        masks = sample_subset_masks(
            NUM_SAMPLES_LARGE, 2, torch.device("cpu"),
        )
        cards = masks.sum(dim=1).long().numpy()
        counts = np.bincount(cards, minlength=3)
        expected = NUM_SAMPLES_LARGE / 3
        chi2, p = stats.chisquare(counts, f_exp=[expected] * 3)
        assert p > CHI2_ALPHA

    def test_shapley_masks_small_n(self):
        """Shapley masks with n=3 (smallest meaningful case)."""
        masks = sample_shapley_masks(
            10, 3, 4, paired=True,
        )
        assert masks.shape == (10, 4, 3)
        assert torch.all((masks == 0.0) | (masks == 1.0))
