"""Tests for vit_shapley.baselines.attention.

Uses vit_tiny_patch16_224 (no pretrained weights) for model-dependent tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vit_shapley.baselines.attention import (
    attentions_to_explanation,
    compute_joint_attention,
    extract_attention_maps,
)

_TINY_MODEL = "vit_tiny_patch16_224"
_IMG_SIZE = 224
_PATCH_SIZE = 16
_NUM_PATCHES = (_IMG_SIZE // _PATCH_SIZE) ** 2  # 196
_N_TOKENS = _NUM_PATCHES + 1                    # 197 (CLS + patches)


@pytest.fixture(scope="module")
def tiny_vit():
    import timm
    model = timm.create_model(_TINY_MODEL, pretrained=False, num_classes=10)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# compute_joint_attention
# ---------------------------------------------------------------------------

class TestComputeJointAttention:
    def test_shape_preserved(self):
        B, L, N = 2, 4, 8
        attn = np.random.rand(B, L, N, N)
        attn = attn / attn.sum(axis=-1, keepdims=True)
        result = compute_joint_attention(attn, add_residual=True)
        assert result.shape == (B, L, N, N)

    def test_identity_input_returns_identity_like(self):
        """Identity attention at every layer → rollout at each layer is identity."""
        B, L, N = 1, 3, 5
        eye = np.eye(N)[np.newaxis, np.newaxis, :, :]
        attn = np.broadcast_to(eye, (B, L, N, N)).copy()
        result = compute_joint_attention(attn, add_residual=False)
        for layer in range(L):
            np.testing.assert_allclose(result[0, layer], np.eye(N), atol=1e-6)

    def test_add_residual_changes_result(self):
        B, L, N = 1, 2, 5
        rng = np.random.default_rng(0)
        attn = rng.dirichlet(np.ones(N), size=(B, L, N))  # rows sum to 1
        r_with = compute_joint_attention(attn, add_residual=True)
        r_without = compute_joint_attention(attn, add_residual=False)
        assert not np.allclose(r_with, r_without)

    def test_rollout_rows_sum_to_one(self):
        """After residual + re-normalisation, all rows of the rollout should sum to 1."""
        B, L, N = 2, 3, 6
        rng = np.random.default_rng(1)
        attn = rng.dirichlet(np.ones(N), size=(B, L, N))  # row-stochastic
        result = compute_joint_attention(attn, add_residual=True)
        row_sums = result.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones((B, L, N)), atol=1e-6)

    def test_wrong_ndim_raises(self):
        with pytest.raises(AssertionError):
            compute_joint_attention(np.ones((2, 3, 4)))


# ---------------------------------------------------------------------------
# attentions_to_explanation
# ---------------------------------------------------------------------------

class TestAttentionsToExplanation:
    def _make_attentions(self, B=2, L=4, H=3, N=_N_TOKENS):
        rng = np.random.default_rng(42)
        raw = rng.random((B, L, H, N, N))
        # Normalise rows within each head (simulate softmax).
        return raw / raw.sum(axis=-1, keepdims=True)

    def test_shape_rollout(self):
        attn = self._make_attentions()
        B = attn.shape[0]
        result = attentions_to_explanation(attn, mode="rollout")
        assert result.shape == (B, _NUM_PATCHES)

    def test_shape_raw(self):
        attn = self._make_attentions()
        result = attentions_to_explanation(attn, mode="raw")
        assert result.shape == (attn.shape[0], _NUM_PATCHES)

    def test_shape_int_mode(self):
        attn = self._make_attentions(L=4)
        result = attentions_to_explanation(attn, mode=2)
        assert result.shape == (attn.shape[0], _NUM_PATCHES)

    def test_values_nonnegative(self):
        attn = self._make_attentions()
        for mode in ("rollout", "raw", 0):
            result = attentions_to_explanation(attn, mode=mode)
            assert np.all(result >= 0), f"Negative values found for mode={mode}"

    def test_unknown_mode_raises(self):
        attn = self._make_attentions()
        with pytest.raises(ValueError):
            attentions_to_explanation(attn, mode="bad_mode")

    def test_wrong_ndim_raises(self):
        with pytest.raises(AssertionError):
            attentions_to_explanation(np.ones((2, 3, 4, 5)))


# ---------------------------------------------------------------------------
# extract_attention_maps
# ---------------------------------------------------------------------------

class TestExtractAttentionMaps:
    def test_output_shape(self, tiny_vit):
        B = 2
        x = torch.randn(B, 3, _IMG_SIZE, _IMG_SIZE)
        result = extract_attention_maps(tiny_vit, x)
        L = len(tiny_vit.blocks)
        H = tiny_vit.blocks[0].attn.num_heads
        assert result.shape == (B, L, H, _N_TOKENS, _N_TOKENS)

    def test_rows_sum_to_one(self, tiny_vit):
        """Attention weights must sum to 1 along the key axis (softmax property)."""
        x = torch.randn(1, 3, _IMG_SIZE, _IMG_SIZE)
        result = extract_attention_maps(tiny_vit, x)
        row_sums = result.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-5)

    def test_values_nonnegative(self, tiny_vit):
        x = torch.randn(1, 3, _IMG_SIZE, _IMG_SIZE)
        result = extract_attention_maps(tiny_vit, x)
        assert np.all(result >= 0)

    def test_model_restored_to_eval(self, tiny_vit):
        """extract_attention_maps should leave the model in eval mode."""
        tiny_vit.eval()
        x = torch.randn(1, 3, _IMG_SIZE, _IMG_SIZE)
        extract_attention_maps(tiny_vit, x)
        assert not tiny_vit.training

    def test_model_restored_to_train(self, tiny_vit):
        """extract_attention_maps should restore training mode if model was training."""
        tiny_vit.train()
        x = torch.randn(1, 3, _IMG_SIZE, _IMG_SIZE)
        extract_attention_maps(tiny_vit, x)
        assert tiny_vit.training
        tiny_vit.eval()  # restore for other tests

    def test_fused_attn_restored(self, tiny_vit):
        """fused_attn attribute must be restored after extraction."""
        # Set all blocks to fused_attn=True if attribute exists.
        orig_vals = {}
        for i, blk in enumerate(tiny_vit.blocks):
            if hasattr(blk.attn, "fused_attn"):
                orig_vals[i] = blk.attn.fused_attn
                blk.attn.fused_attn = True

        x = torch.randn(1, 3, _IMG_SIZE, _IMG_SIZE)
        extract_attention_maps(tiny_vit, x)

        for i, blk in enumerate(tiny_vit.blocks):
            if i in orig_vals:
                assert blk.attn.fused_attn == orig_vals[i], \
                    f"Block {i} fused_attn not restored"

    def test_pipeline_to_explanation(self, tiny_vit):
        """Full pipeline: extract maps → attentions_to_explanation gives (B, P)."""
        B = 2
        x = torch.randn(B, 3, _IMG_SIZE, _IMG_SIZE)
        attn_maps = extract_attention_maps(tiny_vit, x)
        explanation = attentions_to_explanation(attn_maps, mode="rollout")
        assert explanation.shape == (B, _NUM_PATCHES)
        assert np.all(explanation >= 0)
