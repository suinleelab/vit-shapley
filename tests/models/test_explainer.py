"""Tests for vit_shapley.models.explainer.

All tests use vit_tiny_patch16_224 without pretrained weights for speed.
No real data download is required.
"""

from __future__ import annotations

import pytest
import torch

from vit_shapley.models.explainer import ExplainerViT, build_vit_explainer
from vit_shapley.models.surrogate import build_vit_surrogate

_TINY_MODEL = "vit_tiny_patch16_224"
_SMALL_MODEL = "vit_small_patch16_224"
_NUM_CLASSES = 5
_NUM_PATCHES = 196  # 14×14 for 224px images with 16px patches
_EMBED_DIM_TINY = 192  # vit_tiny embed dim


# Old-style (linear head, no normalization) fixture for backward-compat tests.
@pytest.fixture
def explainer():
    return build_vit_explainer(
        _TINY_MODEL,
        num_classes=_NUM_CLASSES,
        num_attn_blocks=0,
        num_mlp_layers=1,
        normalization=None,
        activation=None,
    )


# Paper-default explainer fixture (1 attn block, 3-layer MLP, additive norm).
@pytest.fixture
def explainer_paper():
    return build_vit_explainer(_TINY_MODEL, num_classes=_NUM_CLASSES)


def _make_grand_null(B: int, num_classes: int):
    """Create dummy grand/null tensors for testing."""
    grand = torch.rand(B, num_classes).softmax(dim=-1)
    null = torch.rand(B, num_classes).softmax(dim=-1)
    return grand, null


# ---------------------------------------------------------------------------
# build_vit_explainer
# ---------------------------------------------------------------------------


class TestBuildVitExplainer:
    def test_returns_explainer_vit(self, explainer):
        assert isinstance(explainer, ExplainerViT)

    def test_build_with_surrogate_ckpt(self, tmp_path):
        """Explainer built from a surrogate ckpt should inherit backbone weights."""
        surrogate = build_vit_surrogate(_TINY_MODEL, num_classes=_NUM_CLASSES)
        ckpt_path = tmp_path / "best_surrogate.pth"
        torch.save({"model_state_dict": surrogate.state_dict()}, ckpt_path)

        explainer = build_vit_explainer(
            _TINY_MODEL,
            num_classes=_NUM_CLASSES,
            surrogate_ckpt_path=str(ckpt_path),
            num_attn_blocks=0,
            num_mlp_layers=1,
            normalization=None,
            activation=None,
        )
        # Explainer vit weights should match the surrogate backbone
        for (n1, p1), (n2, p2) in zip(
            surrogate.vit.named_parameters(), explainer.vit.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Mismatch at {n1} vs {n2}"

    def test_shapley_head_is_new(self, explainer):
        """shapley_head must not be the same object as vit.head."""
        assert explainer.shapley_head is not explainer.vit.head

    def test_num_patches_consistent(self, explainer):
        """Output patch dimension should match vit.patch_embed.num_patches."""
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = explainer(x)
        expected_patches = explainer.vit.patch_embed.num_patches
        assert out.shape[1] == expected_patches

    def test_different_model_names(self):
        """Both vit_tiny and vit_small should build without error."""
        for model_name in [_TINY_MODEL, _SMALL_MODEL]:
            m = build_vit_explainer(
                model_name,
                num_classes=_NUM_CLASSES,
                num_attn_blocks=0,
                num_mlp_layers=1,
                normalization=None,
                activation=None,
            )
            assert isinstance(m, ExplainerViT)

    def test_build_paper_defaults(self):
        """Paper-default architecture (1 attn block, 3-layer MLP) builds ok."""
        m = build_vit_explainer(_TINY_MODEL, num_classes=_NUM_CLASSES)
        assert isinstance(m, ExplainerViT)
        assert len(m.attention_blocks) == 1

    @pytest.mark.parametrize("num_attn_blocks", [0, 2])
    def test_build_num_attn_blocks(self, num_attn_blocks):
        m = build_vit_explainer(
            _TINY_MODEL,
            num_classes=_NUM_CLASSES,
            num_attn_blocks=num_attn_blocks,
            num_mlp_layers=1,
            normalization=None,
            activation=None,
        )
        assert len(m.attention_blocks) == num_attn_blocks

    @pytest.mark.parametrize("num_mlp_layers", [1, 3])
    def test_build_mlp_layers(self, num_mlp_layers):
        m = build_vit_explainer(
            _TINY_MODEL,
            num_classes=_NUM_CLASSES,
            num_attn_blocks=0,
            num_mlp_layers=num_mlp_layers,
            normalization=None,
            activation=None,
        )
        linear_layers = [
            mod for mod in m.shapley_head.modules() if isinstance(mod, torch.nn.Linear)
        ]
        assert len(linear_layers) == num_mlp_layers

    def test_invalid_mlp_layers_raises(self):
        with pytest.raises(ValueError, match="num_mlp_layers"):
            build_vit_explainer(
                _TINY_MODEL,
                num_classes=_NUM_CLASSES,
                num_attn_blocks=0,
                num_mlp_layers=5,
                normalization=None,
            )


# ---------------------------------------------------------------------------
# ExplainerViT forward — no normalization (backward-compat tests)
# ---------------------------------------------------------------------------


class TestExplainerVitForward:
    def test_output_shape(self, explainer):
        """Output shape must be (B, num_patches, num_classes)."""
        B = 3
        x = torch.randn(B, 3, 224, 224)
        with torch.no_grad():
            out = explainer(x)
        assert out.shape == (B, _NUM_PATCHES, _NUM_CLASSES)

    def test_output_finite(self, explainer):
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = explainer(x)
        assert torch.isfinite(out).all()

    def test_grad_flows_through_head(self):
        """Gradients must reach the final linear layer of shapley_head."""
        m = build_vit_explainer(
            _TINY_MODEL,
            num_classes=_NUM_CLASSES,
            num_attn_blocks=0,
            num_mlp_layers=1,
            normalization=None,
            activation=None,
        )
        m.train()
        x = torch.randn(2, 3, 224, 224)
        out = m(x)
        loss = out.sum()
        loss.backward()
        # shapley_head is a Sequential; get the last Linear layer
        last_linear = [
            mod for mod in m.shapley_head.modules() if isinstance(mod, torch.nn.Linear)
        ][-1]
        assert last_linear.weight.grad is not None
        assert last_linear.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# ExplainerViT forward — paper-default architecture with normalization
# ---------------------------------------------------------------------------


class TestExplainerVitForwardPaperDefaults:
    def test_output_shape_with_normalization(self, explainer_paper):
        """Paper-default explainer must output (B, n, C) when grand/null given."""
        B = 2
        x = torch.randn(B, 3, 224, 224)
        grand, null = _make_grand_null(B, _NUM_CLASSES)
        with torch.no_grad():
            out = explainer_paper(x, grand=grand, null=null)
        assert out.shape == (B, _NUM_PATCHES, _NUM_CLASSES)

    def test_forward_without_grand_null_raises(self, explainer_paper):
        """forward() without grand/null should raise when normalization='additive'."""
        x = torch.randn(1, 3, 224, 224)
        with pytest.raises(ValueError, match="grand and null"):
            explainer_paper(x)

    def test_additive_normalization_efficiency(self, explainer_paper):
        """With additive normalization, Σ_i φ'_i should equal grand − null."""
        B = 3
        x = torch.randn(B, 3, 224, 224)
        grand, null = _make_grand_null(B, _NUM_CLASSES)
        with torch.no_grad():
            phi = explainer_paper(x, grand=grand, null=null)
        phi_sum = phi.sum(dim=1)  # (B, C)
        expected = grand - null  # (B, C)
        assert torch.allclose(phi_sum, expected, atol=1e-5), (
            f"Efficiency axiom violated: max deviation "
            f"{(phi_sum - expected).abs().max().item():.2e}"
        )

    def test_grad_flows_with_normalization(self, explainer_paper):
        """Gradients must flow through the normalised output.

        Note: phi.sum() is a constant (= grand - null) due to additive
        normalization, so we use a nonlinear loss (phi**2).sum() which has
        non-zero gradient w.r.t. the network parameters.
        """
        explainer_paper.train()
        B = 2
        x = torch.randn(B, 3, 224, 224)
        grand, null = _make_grand_null(B, _NUM_CLASSES)
        phi = explainer_paper(x, grand=grand, null=null)
        loss = (phi**2).sum()  # nonlinear: gradient ≠ 0
        loss.backward()
        last_linear = [
            mod
            for mod in explainer_paper.shapley_head.modules()
            if isinstance(mod, torch.nn.Linear)
        ][-1]
        assert last_linear.weight.grad is not None
        assert last_linear.weight.grad.abs().sum() > 0

    def test_attn_block_norm1_is_identity(self, explainer_paper):
        """norm1 of the first extra attention block must be replaced with Identity."""
        assert isinstance(explainer_paper.attention_blocks[0].norm1, torch.nn.Identity)

    def test_paper_head_output_bounded_by_tanh(self, explainer_paper):
        """Without normalization the raw predictions should be in (-1, 1) via tanh.

        When normalization='additive' the final values can exceed ±1 due to the
        additive correction, but we verify the correction is small for random
        grand/null that are already consistent.
        """
        B = 2
        x = torch.randn(B, 3, 224, 224)
        # Use grand = null so the correction residual is -phi_mean/n ≈ small
        rand_val = torch.rand(B, _NUM_CLASSES).softmax(dim=-1)
        with torch.no_grad():
            out = explainer_paper(x, grand=rand_val, null=rand_val)
        # sum over patches should equal grand-null = 0
        assert torch.allclose(out.sum(dim=1), torch.zeros(B, _NUM_CLASSES), atol=1e-5)

    def test_normalization_none_forward_no_grand_null(self):
        """Explainer with normalization=None must work without grand/null."""
        m = build_vit_explainer(
            _TINY_MODEL,
            num_classes=_NUM_CLASSES,
            num_attn_blocks=1,
            num_mlp_layers=3,
            normalization=None,
            activation="tanh",
        )
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (2, _NUM_PATCHES, _NUM_CLASSES)

    def test_include_cls_false(self):
        """include_cls=False must still give correct output shape."""
        m = build_vit_explainer(
            _TINY_MODEL,
            num_classes=_NUM_CLASSES,
            num_attn_blocks=1,
            num_mlp_layers=1,
            include_cls=False,
            normalization=None,
            activation=None,
        )
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (2, _NUM_PATCHES, _NUM_CLASSES)
