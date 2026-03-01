"""Tests for vit_shapley.models.surrogate.

All tests use vit_tiny_patch16_224 without pretrained weights for speed.
No real data download is required.
"""

from __future__ import annotations

import copy

import pytest
import torch

from vit_shapley.models.classifier import build_vit_classifier
from vit_shapley.models.surrogate import MASKING_STRATEGIES, SurrogateViT, build_vit_surrogate

_TINY_MODEL = "vit_tiny_patch16_224"
_NUM_CLASSES = 5
_NUM_PATCHES = 196  # 14×14 for 224px images with 16px patches


@pytest.fixture
def surrogate():
    return build_vit_surrogate(_TINY_MODEL, num_classes=_NUM_CLASSES)


# ---------------------------------------------------------------------------
# build_vit_surrogate
# ---------------------------------------------------------------------------

class TestBuildVitSurrogate:
    def test_returns_surrogate_vit(self):
        m = build_vit_surrogate(_TINY_MODEL, num_classes=_NUM_CLASSES)
        assert isinstance(m, SurrogateViT)

    def test_is_nn_module(self, surrogate):
        assert isinstance(surrogate, torch.nn.Module)

    def test_has_trainable_params(self, surrogate):
        trainable = [p for p in surrogate.parameters() if p.requires_grad]
        assert len(trainable) > 0

    def test_loads_classifier_checkpoint(self, tmp_path):
        """Surrogate built from a saved classifier should have the same weights."""
        classifier = build_vit_classifier(_TINY_MODEL, num_classes=_NUM_CLASSES, pretrained=False)
        ckpt_path = tmp_path / "classifier.pth"
        torch.save({"model_state_dict": classifier.state_dict()}, ckpt_path)

        surrogate = build_vit_surrogate(
            _TINY_MODEL, num_classes=_NUM_CLASSES, classifier_ckpt_path=str(ckpt_path)
        )
        # Surrogate's vit weights should match the saved classifier weights.
        for (n1, p1), (n2, p2) in zip(
            classifier.named_parameters(), surrogate.vit.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Mismatch at {n1} vs {n2}"


# ---------------------------------------------------------------------------
# SurrogateViT forward
# ---------------------------------------------------------------------------

class TestSurrogateVitForward:
    def test_output_shape_no_mask(self, surrogate):
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = surrogate(x)
        assert out.shape == (2, _NUM_CLASSES)

    def test_output_shape_with_partial_mask(self, surrogate):
        B = 2
        x = torch.randn(B, 3, 224, 224)
        mask = torch.zeros(B, _NUM_PATCHES)
        mask[:, : _NUM_PATCHES // 2] = 1.0
        with torch.no_grad():
            out = surrogate(x, patch_mask=mask)
        assert out.shape == (B, _NUM_CLASSES)

    def test_output_shape_all_masked(self, surrogate):
        """All patches masked (only CLS visible) — must not error or produce NaN."""
        B = 2
        x = torch.randn(B, 3, 224, 224)
        mask = torch.zeros(B, _NUM_PATCHES)
        with torch.no_grad():
            out = surrogate(x, patch_mask=mask)
        assert out.shape == (B, _NUM_CLASSES)
        assert torch.isfinite(out).all(), "Output contains NaN/Inf with all patches masked"

    def test_output_shape_all_visible(self, surrogate):
        B = 2
        x = torch.randn(B, 3, 224, 224)
        mask = torch.ones(B, _NUM_PATCHES)
        with torch.no_grad():
            out = surrogate(x, patch_mask=mask)
        assert out.shape == (B, _NUM_CLASSES)

    def test_no_mask_equals_original_vit(self):
        """SurrogateViT(x, mask=None) must exactly match the original timm ViT."""
        vit = build_vit_classifier(_TINY_MODEL, num_classes=_NUM_CLASSES, pretrained=False)
        vit_ref = copy.deepcopy(vit)   # reference before SurrogateViT modifies nothing
        surrogate = SurrogateViT(vit)  # wraps vit (no attention replacement needed)

        vit_ref.eval()
        surrogate.eval()

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            ref_out = vit_ref(x)
            surr_out = surrogate(x, patch_mask=None)

        assert torch.allclose(ref_out, surr_out, atol=1e-5)

    def test_all_visible_mask_matches_no_mask(self, surrogate):
        """Full mask (all 1s) should give the same result as no mask."""
        surrogate.eval()
        x = torch.randn(1, 3, 224, 224)
        mask = torch.ones(1, _NUM_PATCHES)
        with torch.no_grad():
            no_mask_out = surrogate(x)
            full_mask_out = surrogate(x, patch_mask=mask)
        assert torch.allclose(no_mask_out, full_mask_out, atol=1e-4)

    def test_masking_changes_output(self, surrogate):
        """Masking half the patches should produce a different output."""
        surrogate.eval()
        x = torch.randn(1, 3, 224, 224)
        mask = torch.zeros(1, _NUM_PATCHES)
        mask[0, : _NUM_PATCHES // 2] = 1.0
        with torch.no_grad():
            no_mask_out = surrogate(x)
            masked_out = surrogate(x, patch_mask=mask)
        assert not torch.allclose(no_mask_out, masked_out, atol=1e-3)

    def test_mask_cleared_between_calls(self, surrogate):
        """After a masked forward, the next call without mask must use no masking."""
        surrogate.eval()
        x = torch.randn(1, 3, 224, 224)
        mask = torch.zeros(1, _NUM_PATCHES)  # all masked

        with torch.no_grad():
            # First call with mask
            surrogate(x, patch_mask=mask)
            # Second call — no mask; should match a fresh no-mask call
            out_after = surrogate(x)
            out_clean = surrogate(x)

        assert torch.allclose(out_after, out_clean, atol=1e-6)

    def test_per_sample_masks_are_independent(self, surrogate):
        """Sample 0 with full mask + sample 1 with no-patches should match solo calls."""
        surrogate.eval()
        x = torch.randn(2, 3, 224, 224)

        full_mask = torch.zeros(2, _NUM_PATCHES)
        full_mask[0] = 1.0  # sample 0: all visible; sample 1: all masked

        with torch.no_grad():
            batch_out = surrogate(x, patch_mask=full_mask)

            solo_full = surrogate(x[0:1], patch_mask=torch.ones(1, _NUM_PATCHES))
            solo_none = surrogate(x[1:2], patch_mask=torch.zeros(1, _NUM_PATCHES))

        assert torch.allclose(batch_out[0], solo_full[0], atol=1e-4)
        assert torch.allclose(batch_out[1], solo_none[0], atol=1e-4)

    def test_output_finite_for_random_masks(self, surrogate):
        """Outputs should be finite for random masks across a batch."""
        surrogate.eval()
        B = 4
        x = torch.randn(B, 3, 224, 224)
        mask = (torch.rand(B, _NUM_PATCHES) > 0.5).float()
        with torch.no_grad():
            out = surrogate(x, patch_mask=mask)
        assert torch.isfinite(out).all()

    def test_state_dict_keys(self, surrogate):
        """State dict should contain vit.* keys (no double-wrapped attn)."""
        keys = list(surrogate.state_dict().keys())
        assert any(k.startswith("vit.") for k in keys)
        # Because we do NOT wrap attention, keys should be standard timm layout.
        attn_keys = [k for k in keys if "attn" in k]
        # Wrapped layout would have "attn.attn."; standard timm has just "attn."
        double_wrapped = [k for k in attn_keys if "attn.attn." in k]
        assert len(double_wrapped) == 0, "Unexpected double-wrapped attn keys found"


# ---------------------------------------------------------------------------
# Zero-input masking strategy
# ---------------------------------------------------------------------------

@pytest.fixture
def zero_input_surrogate():
    return build_vit_surrogate(_TINY_MODEL, num_classes=_NUM_CLASSES, masking_strategy="zero_input")


class TestZeroInputMasking:
    def test_invalid_strategy_raises(self):
        vit = build_vit_classifier(_TINY_MODEL, num_classes=_NUM_CLASSES, pretrained=False)
        with pytest.raises(ValueError, match="masking_strategy"):
            SurrogateViT(vit, masking_strategy="bad_strategy")

    def test_known_strategies_constant(self):
        assert "attn_mask" in MASKING_STRATEGIES
        assert "zero_input" in MASKING_STRATEGIES

    def test_build_with_zero_input(self):
        m = build_vit_surrogate(_TINY_MODEL, num_classes=_NUM_CLASSES, masking_strategy="zero_input")
        assert isinstance(m, SurrogateViT)
        assert m.masking_strategy == "zero_input"

    def test_output_shape_no_mask(self, zero_input_surrogate):
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = zero_input_surrogate(x)
        assert out.shape == (2, _NUM_CLASSES)

    def test_output_shape_with_partial_mask(self, zero_input_surrogate):
        B = 2
        x = torch.randn(B, 3, 224, 224)
        mask = torch.zeros(B, _NUM_PATCHES)
        mask[:, : _NUM_PATCHES // 2] = 1.0
        with torch.no_grad():
            out = zero_input_surrogate(x, patch_mask=mask)
        assert out.shape == (B, _NUM_CLASSES)

    def test_output_finite_all_masked(self, zero_input_surrogate):
        """All patches zeroed (blank image) — must not produce NaN/Inf."""
        B = 2
        x = torch.randn(B, 3, 224, 224)
        mask = torch.zeros(B, _NUM_PATCHES)
        with torch.no_grad():
            out = zero_input_surrogate(x, patch_mask=mask)
        assert out.shape == (B, _NUM_CLASSES)
        assert torch.isfinite(out).all()

    def test_full_mask_matches_no_mask(self, zero_input_surrogate):
        """All-ones mask should give same result as no mask."""
        zero_input_surrogate.eval()
        x = torch.randn(1, 3, 224, 224)
        mask = torch.ones(1, _NUM_PATCHES)
        with torch.no_grad():
            no_mask_out = zero_input_surrogate(x)
            full_mask_out = zero_input_surrogate(x, patch_mask=mask)
        assert torch.allclose(no_mask_out, full_mask_out, atol=1e-5)

    def test_masking_changes_output(self, zero_input_surrogate):
        """Zeroing half the patches should produce different output."""
        zero_input_surrogate.eval()
        x = torch.randn(1, 3, 224, 224)
        mask = torch.zeros(1, _NUM_PATCHES)
        mask[0, : _NUM_PATCHES // 2] = 1.0
        with torch.no_grad():
            no_mask_out = zero_input_surrogate(x)
            masked_out = zero_input_surrogate(x, patch_mask=mask)
        assert not torch.allclose(no_mask_out, masked_out, atol=1e-3)

    def test_pixel_regions_zeroed(self, zero_input_surrogate):
        """Masked patch pixel regions should be exactly zero after _zero_masked_patches."""
        B, ps = 1, zero_input_surrogate.patch_size
        x = torch.ones(B, 3, 224, 224)
        mask = torch.zeros(B, _NUM_PATCHES)
        mask[0, 0] = 1.0   # only patch 0 visible
        x_masked = zero_input_surrogate._zero_masked_patches(x, mask)
        # Patch 0 is at pixels [0:ps, 0:ps] — should remain 1
        assert x_masked[0, :, :ps, :ps].allclose(torch.ones(3, ps, ps))
        # All other pixels should be zero
        x_rest = x_masked.clone()
        x_rest[0, :, :ps, :ps] = 0.0
        assert x_rest.sum() == 0.0

    def test_output_finite_random_masks(self, zero_input_surrogate):
        zero_input_surrogate.eval()
        B = 4
        x = torch.randn(B, 3, 224, 224)
        mask = (torch.rand(B, _NUM_PATCHES) > 0.5).float()
        with torch.no_grad():
            out = zero_input_surrogate(x, patch_mask=mask)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Three-way mask equivalence
# ---------------------------------------------------------------------------


class TestMaskEquivalence:
    """Three ways to apply the same mask must produce identical logits.

    (A) Official attn_mask forward — surrogate(images, batch_mask).
    (B) Token-deletion forward — physically remove masked patch tokens before
        the transformer, keeping only visible ones.
    (C) Pixel-perturbation forward — corrupt masked-patch pixels with a large
        sentinel value, but still pass the same attention mask.

    Cases A and B are mathematically equivalent because the attention mask only
    blocks masked columns (keys): visible and CLS tokens attend to the exact
    same set of tokens in both approaches.  Case C is exactly equivalent to A
    because the attention mask prevents any token from attending to the corrupted
    positions, so their pixel values cannot affect the output.
    """

    @pytest.fixture
    def surrogate_eval(self):
        m = build_vit_surrogate(_TINY_MODEL, num_classes=_NUM_CLASSES)
        m.eval()
        return m

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _slice_pos_embed(pos_embed: torch.Tensor, mask_1d: torch.Tensor) -> torch.Tensor:
        """Return pos_embed for [CLS, *visible_patches].

        Args:
            pos_embed: ``(1, N+1, D)`` — full position embedding tensor.
            mask_1d:   ``(N,)`` binary tensor; 1 = visible patch.

        Returns:
            ``(1, K+1, D)`` where K = number of visible patches.
        """
        cls_pe = pos_embed[:, 0:1, :]                      # (1, 1, D)
        kept_pe = pos_embed[:, 1:, :][:, mask_1d.bool(), :]  # (1, K, D)
        return torch.cat([cls_pe, kept_pe], dim=1)          # (1, K+1, D)

    @staticmethod
    def _token_deletion_forward(
        vit: torch.nn.Module,
        images: torch.Tensor,
        mask_1d: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with masked patch tokens physically removed.

        Replicates timm's forward_features manually so we can substitute a
        sliced position embedding without modifying any model parameters.

        Args:
            vit:      timm VisionTransformer (must already be in eval mode).
            images:   ``(B, C, H, W)`` image batch.
            mask_1d:  ``(N,)`` binary mask shared across the batch.

        Returns:
            Logits ``(B, num_classes)``.
        """
        B = images.shape[0]

        # 1. Patch embed — all N patches
        patch_tokens = vit.patch_embed(images)  # (B, N, D)

        # 2. Keep only visible patches
        kept = patch_tokens[:, mask_1d.bool(), :]  # (B, K, D)

        # 3. Build sequence: CLS token prepended to kept patches
        cls_tokens = vit.cls_token.expand(B, -1, -1)   # (B, 1, D)
        x = torch.cat([cls_tokens, kept], dim=1)         # (B, K+1, D)

        # 4. Add sliced positional embeddings (CLS pos + visible patch pos)
        sliced_pe = TestMaskEquivalence._slice_pos_embed(vit.pos_embed, mask_1d)
        x = x + sliced_pe                                # broadcast over B
        x = vit.pos_drop(x)

        # 5. Optional identity layers present in some timm variants
        x = vit.patch_drop(x)
        x = vit.norm_pre(x)

        # 6. Transformer blocks — no attn_mask needed (tokens physically absent)
        for blk in vit.blocks:
            x = blk(x)

        # 7. Post-transformer norm
        x = vit.norm(x)

        # 8. Classification head (pools CLS token for global_pool='token')
        return vit.forward_head(x)

    @staticmethod
    def _perturb_masked_patches(
        images: torch.Tensor,
        mask_1d: torch.Tensor,
        patch_size: int,
        grid_w: int,
    ) -> torch.Tensor:
        """Replace pixel regions of masked patches with a large sentinel value.

        Args:
            images:     ``(B, C, H, W)`` image batch.
            mask_1d:    ``(N,)`` binary mask; 0 = patch to corrupt.
            patch_size: Side length of each patch in pixels.
            grid_w:     Number of patches along the width axis.

        Returns:
            Cloned image tensor with masked patch pixels set to 4242.
        """
        out = images.clone()
        for idx, visible in enumerate(mask_1d):
            if not visible:
                row, col = idx // grid_w, idx % grid_w
                r0, r1 = row * patch_size, (row + 1) * patch_size
                c0, c1 = col * patch_size, (col + 1) * patch_size
                out[:, :, r0:r1, c0:c1] = 4242.0
        return out

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_attn_mask_equals_token_deletion(self, surrogate_eval):
        """Attention masking (A) and token deletion (B) must agree."""
        torch.manual_seed(0)
        B = 2
        images = torch.randn(B, 3, 224, 224)

        mask_1d = torch.zeros(_NUM_PATCHES)
        mask_1d[: _NUM_PATCHES // 2] = 1.0          # first half visible
        batch_mask = mask_1d.unsqueeze(0).expand(B, -1)

        with torch.no_grad():
            logits_A = surrogate_eval(images, patch_mask=batch_mask)
            logits_B = self._token_deletion_forward(surrogate_eval.vit, images, mask_1d)

        assert torch.allclose(logits_A, logits_B, atol=1e-5), (
            f"Token deletion (B) differs from attention masking (A).\n"
            f"Max |diff|: {(logits_A - logits_B).abs().max():.2e}"
        )

    def test_attn_mask_equals_pixel_perturbation(self, surrogate_eval):
        """Attention masking (A) and pixel perturbation (C) must agree exactly."""
        torch.manual_seed(1)
        B = 2
        images = torch.randn(B, 3, 224, 224)

        mask_1d = torch.zeros(_NUM_PATCHES)
        mask_1d[: _NUM_PATCHES // 2] = 1.0
        batch_mask = mask_1d.unsqueeze(0).expand(B, -1)

        ps = surrogate_eval.patch_size
        grid_w = 224 // ps

        with torch.no_grad():
            logits_A = surrogate_eval(images, patch_mask=batch_mask)
            images_perturbed = self._perturb_masked_patches(images, mask_1d, ps, grid_w)
            logits_C = surrogate_eval(images_perturbed, patch_mask=batch_mask)

        assert torch.allclose(logits_A, logits_C, atol=1e-5), (
            f"Pixel perturbation (C) differs from attention masking (A).\n"
            f"Max |diff|: {(logits_A - logits_C).abs().max():.2e}"
        )

    def test_all_three_strategies_equivalent(self, surrogate_eval):
        """A, B, and C all agree for a random sparse mask."""
        torch.manual_seed(2)
        B = 3
        images = torch.randn(B, 3, 224, 224)

        # Sparse mask: ~25 % of patches visible
        mask_1d = (torch.rand(_NUM_PATCHES) < 0.25).float()
        batch_mask = mask_1d.unsqueeze(0).expand(B, -1)

        ps = surrogate_eval.patch_size
        grid_w = 224 // ps

        with torch.no_grad():
            logits_A = surrogate_eval(images, patch_mask=batch_mask)
            logits_B = self._token_deletion_forward(surrogate_eval.vit, images, mask_1d)
            images_perturbed = self._perturb_masked_patches(images, mask_1d, ps, grid_w)
            logits_C = surrogate_eval(images_perturbed, patch_mask=batch_mask)

        assert torch.allclose(logits_A, logits_B, atol=1e-5), (
            f"Token deletion (B) vs attn_mask (A): max |diff| = "
            f"{(logits_A - logits_B).abs().max():.2e}"
        )
        assert torch.allclose(logits_A, logits_C, atol=1e-5), (
            f"Pixel perturbation (C) vs attn_mask (A): max |diff| = "
            f"{(logits_A - logits_C).abs().max():.2e}"
        )
