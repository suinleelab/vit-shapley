"""Tests for vit_shapley.training.train_explainer.

Uses tiny synthetic datasets (no ImageNette download) and vit_tiny_patch16_224
without pretrained weights for speed.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from vit_shapley.models.explainer import build_vit_explainer
from vit_shapley.models.surrogate import build_vit_surrogate
from vit_shapley.training.train_explainer import (
    evaluate_explainer,
    sample_shapley_masks,
    shapley_kernel_weights,
    train_explainer,
    train_one_epoch_explainer,
)

_TINY_MODEL = "vit_tiny_patch16_224"
_NUM_CLASSES = 3
_NUM_PATCHES = 196  # 14×14
_IMAGE_SIZE = 224
_BATCH_SIZE = 2
_NUM_SAMPLES = 6


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_dataset(
    num_samples: int = _NUM_SAMPLES,
    num_classes: int = _NUM_CLASSES,
) -> TensorDataset:
    images = torch.randn(num_samples, 3, _IMAGE_SIZE, _IMAGE_SIZE)
    labels = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(images, labels)


@pytest.fixture
def tiny_explainer():
    """Old-style linear-head explainer (no normalization) for fast tests."""
    return build_vit_explainer(
        _TINY_MODEL,
        num_classes=_NUM_CLASSES,
        num_attn_blocks=0,
        num_mlp_layers=1,
        normalization="additive",
        activation=None,
    )


@pytest.fixture
def tiny_surrogate():
    return build_vit_surrogate(_TINY_MODEL, num_classes=_NUM_CLASSES)


@pytest.fixture
def train_loader():
    return DataLoader(_make_synthetic_dataset(), batch_size=_BATCH_SIZE, shuffle=True)


@pytest.fixture
def val_loader():
    return DataLoader(_make_synthetic_dataset(num_samples=4), batch_size=_BATCH_SIZE)


@pytest.fixture
def device():
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# shapley_kernel_weights
# ---------------------------------------------------------------------------


class TestShapleyKernelWeights:
    def test_kernel_weights_length(self):
        n = 10
        weights = shapley_kernel_weights(n)
        assert len(weights) == n + 1

    def test_kernel_weights_boundary_zero(self):
        """w(0) and w(n) must be exactly 0."""
        for n in [5, 10, 50]:
            weights = shapley_kernel_weights(n)
            assert weights[0] == 0.0, f"w(0) != 0 for n={n}"
            assert weights[n] == 0.0, f"w(n) != 0 for n={n}"

    def test_kernel_weights_positive(self):
        """All non-boundary weights must be non-negative."""
        weights = shapley_kernel_weights(10)
        assert all(w >= 0.0 for w in weights)

    def test_kernel_weights_symmetric(self):
        """w(k) must equal w(n-k) for all k."""
        n = 12
        weights = shapley_kernel_weights(n)
        for k in range(n + 1):
            assert abs(weights[k] - weights[n - k]) < 1e-10, (
                f"Asymmetry at k={k}: w({k})={weights[k]}, w({n - k})={weights[n - k]}"
            )

    def test_kernel_weights_large_n(self):
        """Must not crash or produce NaN for n=196 (typical ViT patch count)."""
        weights = shapley_kernel_weights(196)
        assert len(weights) == 197
        assert all(w >= 0.0 for w in weights)
        import math

        assert all(not math.isnan(w) for w in weights)

    def test_kernel_weights_middle_positive(self):
        """At least some middle weights should be strictly positive."""
        n = 10
        weights = shapley_kernel_weights(n)
        middle_weights = weights[1:n]
        assert any(w > 0 for w in middle_weights)


# ---------------------------------------------------------------------------
# sample_shapley_masks
# ---------------------------------------------------------------------------


class TestSampleShapleyMasks:
    def test_output_shape_unpaired(self):
        """Unpaired masks: (B, M, n)."""
        masks = sample_shapley_masks(4, 10, 4, paired=False)
        assert masks.shape == (4, 4, 10)

    def test_output_shape_paired(self):
        """Paired masks: (B, M, n) with M even."""
        masks = sample_shapley_masks(3, 8, 4, paired=True)
        assert masks.shape == (3, 4, 8)

    def test_binary_values(self):
        """All entries must be 0.0 or 1.0."""
        masks = sample_shapley_masks(5, 10, 2, paired=False)
        assert ((masks == 0.0) | (masks == 1.0)).all()

    def test_paired_complement(self):
        """Each mask and its complement must sum to all-ones."""
        B, n, M = 4, 10, 4
        masks = sample_shapley_masks(B, n, M, paired=True)
        half = M // 2
        first_half = masks[:, :half, :]
        second_half = masks[:, half:, :]
        sums = first_half + second_half  # should be all 1s
        assert torch.all(sums == 1.0), "Paired masks do not complement each other"

    def test_cardinalities_mostly_in_range(self):
        """The Bernoulli-threshold method targets cardinalities in [1, n-1],
        but because patches are independently sampled given a threshold,
        k=0 and k=n can occur.  The vast majority should still be interior."""
        B, n, M = 1, 12, 10000
        masks = sample_shapley_masks(B, n, M, paired=False)
        cardinalities = masks.sum(dim=-1)  # (B, M)
        interior = (cardinalities >= 1) & (cardinalities <= n - 1)
        frac_interior = interior.float().mean().item()
        assert frac_interior > 0.5, (
            f"Expected majority interior cardinalities; got {frac_interior:.3f}"
        )

    def test_paired_cardinalities_valid(self):
        """Paired masks should be binary and complement each other;
        cardinalities are in [0, n] (Bernoulli threshold allows extremes)."""
        B, n, M = 4, 10, 4
        masks = sample_shapley_masks(B, n, M, paired=True)
        cardinalities = masks.sum(dim=-1)
        assert (cardinalities >= 0).all()
        assert (cardinalities <= n).all()

    def test_odd_num_mask_samples_with_paired_raises(self):
        """paired=True with odd num_mask_samples must raise ValueError."""
        with pytest.raises(ValueError, match="even"):
            sample_shapley_masks(2, 10, 3, paired=True)

    def test_device_respected(self):
        """Masks must be on the requested device."""
        masks = sample_shapley_masks(2, 10, 2, paired=False, device=torch.device("cpu"))
        assert masks.device.type == "cpu"

    def test_reproducible_with_generator(self):
        """Same generator seed must produce identical masks."""
        gen1 = torch.Generator()
        gen1.manual_seed(42)
        gen2 = torch.Generator()
        gen2.manual_seed(42)
        m1 = sample_shapley_masks(3, 8, 2, paired=False, generator=gen1)
        m2 = sample_shapley_masks(3, 8, 2, paired=False, generator=gen2)
        assert torch.equal(m1, m2)

    def test_float_dtype(self):
        masks = sample_shapley_masks(2, 10, 2, paired=False)
        assert masks.dtype == torch.float32

    def test_large_n(self):
        """Must handle n=196 (full ViT patch grid) without error."""
        masks = sample_shapley_masks(2, 196, 2, paired=True)
        assert masks.shape == (2, 2, 196)


# ---------------------------------------------------------------------------
# train_one_epoch_explainer
# ---------------------------------------------------------------------------


class TestTrainOneEpochExplainer:
    def test_returns_valid_metrics(
        self, tiny_explainer, tiny_surrogate, train_loader, device
    ):
        optimizer = torch.optim.AdamW(tiny_explainer.parameters(), lr=1e-4)
        metrics = train_one_epoch_explainer(
            tiny_explainer,
            tiny_surrogate,
            train_loader,
            optimizer,
            device,
            num_mask_samples=2,
            paired=True,
        )
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float) and metrics["loss"] >= 0.0

    def test_surrogate_weights_frozen(
        self, tiny_explainer, tiny_surrogate, train_loader, device
    ):
        """Surrogate parameters must not change after an epoch."""
        surrogate_before = {k: v.clone() for k, v in tiny_surrogate.named_parameters()}
        tiny_surrogate.eval()
        for p in tiny_surrogate.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.AdamW(tiny_explainer.parameters(), lr=1e-4)
        train_one_epoch_explainer(
            tiny_explainer,
            tiny_surrogate,
            train_loader,
            optimizer,
            device,
            num_mask_samples=2,
            paired=True,
        )

        for k, v_before in surrogate_before.items():
            v_after = dict(tiny_surrogate.named_parameters())[k]
            assert torch.allclose(v_before, v_after), f"Surrogate param {k} changed!"

    def test_explainer_weights_update(
        self, tiny_explainer, tiny_surrogate, train_loader, device
    ):
        """Explainer parameters must change after a training epoch."""
        tiny_surrogate.eval()
        for p in tiny_surrogate.parameters():
            p.requires_grad_(False)

        params_before = {k: v.clone() for k, v in tiny_explainer.named_parameters()}
        optimizer = torch.optim.AdamW(tiny_explainer.parameters(), lr=1e-3)
        train_one_epoch_explainer(
            tiny_explainer,
            tiny_surrogate,
            train_loader,
            optimizer,
            device,
            num_mask_samples=2,
            paired=True,
        )

        changed = any(
            not torch.allclose(params_before[k], v)
            for k, v in tiny_explainer.named_parameters()
        )
        assert changed, "No explainer parameters changed after training epoch"

    def test_scheduler_is_stepped(
        self, tiny_explainer, tiny_surrogate, train_loader, device
    ):
        """The per-step scheduler must be called once per gradient update."""
        optimizer = torch.optim.AdamW(tiny_explainer.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda s: 1.0
        )
        steps_before = scheduler.last_epoch

        train_one_epoch_explainer(
            tiny_explainer,
            tiny_surrogate,
            train_loader,
            optimizer,
            device,
            num_mask_samples=2,
            paired=True,
            scheduler=scheduler,
        )
        # scheduler should have been stepped once per batch
        num_batches = len(train_loader)
        assert scheduler.last_epoch == steps_before + num_batches

    def test_unpaired_masks_accepted(
        self, tiny_explainer, tiny_surrogate, train_loader, device
    ):
        """Training with paired=False must not raise."""
        optimizer = torch.optim.AdamW(tiny_explainer.parameters(), lr=1e-4)
        metrics = train_one_epoch_explainer(
            tiny_explainer,
            tiny_surrogate,
            train_loader,
            optimizer,
            device,
            num_mask_samples=4,
            paired=False,
        )
        assert metrics["loss"] >= 0.0

    def test_gradient_accumulation_runs(
        self, tiny_explainer, tiny_surrogate, train_loader, device
    ):
        """gradient_accumulation_steps > 1 must not raise and produce valid loss."""
        optimizer = torch.optim.AdamW(tiny_explainer.parameters(), lr=1e-4)
        metrics = train_one_epoch_explainer(
            tiny_explainer,
            tiny_surrogate,
            train_loader,
            optimizer,
            device,
            num_mask_samples=2,
            paired=True,
            gradient_accumulation_steps=2,
        )
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float) and metrics["loss"] >= 0.0

    def test_gradient_accumulation_scheduler_steps(
        self, tiny_explainer, tiny_surrogate, train_loader, device
    ):
        """With accum=K, scheduler should step ceil(num_batches/K) times."""
        import math as _math

        accum = 2
        optimizer = torch.optim.AdamW(tiny_explainer.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda s: 1.0
        )
        steps_before = scheduler.last_epoch

        train_one_epoch_explainer(
            tiny_explainer,
            tiny_surrogate,
            train_loader,
            optimizer,
            device,
            num_mask_samples=2,
            paired=True,
            scheduler=scheduler,
            gradient_accumulation_steps=accum,
        )
        num_batches = len(train_loader)
        expected_steps = _math.ceil(num_batches / accum)
        assert scheduler.last_epoch == steps_before + expected_steps

    def test_gradient_accumulation_weights_update(
        self, tiny_explainer, tiny_surrogate, train_loader, device
    ):
        """Explainer weights should still update with gradient accumulation."""
        tiny_surrogate.eval()
        for p in tiny_surrogate.parameters():
            p.requires_grad_(False)

        params_before = {k: v.clone() for k, v in tiny_explainer.named_parameters()}
        optimizer = torch.optim.AdamW(tiny_explainer.parameters(), lr=1e-3)
        train_one_epoch_explainer(
            tiny_explainer,
            tiny_surrogate,
            train_loader,
            optimizer,
            device,
            num_mask_samples=2,
            paired=True,
            gradient_accumulation_steps=2,
        )

        changed = any(
            not torch.allclose(params_before[k], v)
            for k, v in tiny_explainer.named_parameters()
        )
        assert changed, "No explainer parameters changed with gradient accumulation"


# ---------------------------------------------------------------------------
# evaluate_explainer
# ---------------------------------------------------------------------------


class TestEvaluateExplainer:
    def test_evaluate_returns_valid_metrics(
        self, tiny_explainer, tiny_surrogate, val_loader, device
    ):
        metrics = evaluate_explainer(
            tiny_explainer,
            tiny_surrogate,
            val_loader,
            device,
            num_mask_samples=2,
            paired=True,
        )
        assert "loss" in metrics and "efficiency_gap" in metrics
        assert metrics["loss"] >= 0.0

    def test_evaluate_no_gradient(
        self, tiny_explainer, tiny_surrogate, val_loader, device
    ):
        """Surrogate and explainer grad buffers must be None after evaluate."""
        tiny_surrogate.zero_grad()
        tiny_explainer.zero_grad()

        evaluate_explainer(
            tiny_explainer,
            tiny_surrogate,
            val_loader,
            device,
            num_mask_samples=2,
            paired=True,
        )

        for p in tiny_surrogate.parameters():
            assert p.grad is None, "Surrogate gradient was populated during evaluate"
        for p in tiny_explainer.parameters():
            assert p.grad is None, "Explainer gradient was populated during evaluate"

    def test_evaluate_efficiency_gap_near_zero_after_normalization(
        self, tiny_surrogate, val_loader, device
    ):
        """Explainer with additive normalization must have efficiency_gap ≈ 0."""
        explainer = build_vit_explainer(
            _TINY_MODEL,
            num_classes=_NUM_CLASSES,
            num_attn_blocks=0,
            num_mlp_layers=1,
            normalization="additive",
            activation=None,
        )
        metrics = evaluate_explainer(
            explainer,
            tiny_surrogate,
            val_loader,
            device,
            num_mask_samples=2,
            paired=True,
        )
        # Additive norm hard-enforces Σφ = grand - null → gap should be ~0
        assert metrics["efficiency_gap"] < 1e-4, (
            f"Efficiency gap too large: {metrics['efficiency_gap']}"
        )

    def test_evaluate_efficiency_gap_positive(self, tiny_surrogate, val_loader, device):
        """Without normalization the efficiency gap is generally > 0."""
        explainer_no_norm = build_vit_explainer(
            _TINY_MODEL,
            num_classes=_NUM_CLASSES,
            num_attn_blocks=0,
            num_mlp_layers=1,
            normalization=None,
            activation=None,
        )
        metrics = evaluate_explainer(
            explainer_no_norm,
            tiny_surrogate,
            val_loader,
            device,
            num_mask_samples=2,
            paired=True,
        )
        assert metrics["efficiency_gap"] >= 0.0


# ---------------------------------------------------------------------------
# train_explainer
# ---------------------------------------------------------------------------


class TestTrainExplainer:
    def test_train_explainer_history_keys(
        self, tiny_explainer, tiny_surrogate, train_loader, val_loader, device
    ):
        history = train_explainer(
            tiny_explainer,
            tiny_surrogate,
            train_loader,
            val_loader,
            epochs=1,
            device=device,
            use_amp=False,
        )
        assert "train_loss" in history
        assert "val_loss" in history
        assert "val_efficiency_gap" in history
        assert "best_val_loss" in history
        assert "best_epoch" in history

    def test_train_explainer_history_length(
        self, tiny_explainer, tiny_surrogate, train_loader, val_loader, device
    ):
        epochs = 2
        history = train_explainer(
            tiny_explainer,
            tiny_surrogate,
            train_loader,
            val_loader,
            epochs=epochs,
            device=device,
            use_amp=False,
        )
        assert len(history["train_loss"]) == epochs
        assert len(history["val_loss"]) == epochs
        assert len(history["val_efficiency_gap"]) == epochs

    def test_train_explainer_checkpoint_loadable(
        self, tiny_explainer, tiny_surrogate, train_loader, val_loader, device, tmp_path
    ):
        train_explainer(
            tiny_explainer,
            tiny_surrogate,
            train_loader,
            val_loader,
            epochs=1,
            device=device,
            use_amp=False,
            save_dir=tmp_path,
        )
        ckpt = torch.load(
            tmp_path / "best_explainer.pth", map_location="cpu", weights_only=True
        )
        assert "model_state_dict" in ckpt
        assert "epoch" in ckpt
        # Reload into a fresh explainer
        fresh = build_vit_explainer(
            _TINY_MODEL,
            num_classes=_NUM_CLASSES,
            num_attn_blocks=0,
            num_mlp_layers=1,
            normalization="additive",
            activation=None,
        )
        fresh.load_state_dict(ckpt["model_state_dict"])

    def test_train_explainer_no_save_dir(
        self, tiny_explainer, tiny_surrogate, train_loader, val_loader, device
    ):
        """Training without save_dir must not raise."""
        history = train_explainer(
            tiny_explainer,
            tiny_surrogate,
            train_loader,
            val_loader,
            epochs=1,
            device=device,
            use_amp=False,
            save_dir=None,
        )
        assert history is not None

    def test_best_val_loss_is_min(
        self, tiny_explainer, tiny_surrogate, train_loader, val_loader, device
    ):
        epochs = 3
        history = train_explainer(
            tiny_explainer,
            tiny_surrogate,
            train_loader,
            val_loader,
            epochs=epochs,
            device=device,
            use_amp=False,
        )
        assert history["best_val_loss"] <= min(history["val_loss"]) + 1e-9

    def test_best_epoch_in_range(
        self, tiny_explainer, tiny_surrogate, train_loader, val_loader, device
    ):
        epochs = 2
        history = train_explainer(
            tiny_explainer,
            tiny_surrogate,
            train_loader,
            val_loader,
            epochs=epochs,
            device=device,
            use_amp=False,
        )
        assert 1 <= history["best_epoch"] <= epochs

    def test_train_explainer_with_gradient_accumulation(
        self, tiny_surrogate, train_loader, val_loader, device, tmp_path
    ):
        """Full training with gradient_accumulation_steps > 1."""
        explainer = build_vit_explainer(
            _TINY_MODEL,
            num_classes=_NUM_CLASSES,
            num_attn_blocks=0,
            num_mlp_layers=1,
            normalization="additive",
            activation=None,
        )
        history = train_explainer(
            explainer,
            tiny_surrogate,
            train_loader,
            val_loader,
            epochs=1,
            device=device,
            use_amp=False,
            save_dir=tmp_path,
            gradient_accumulation_steps=2,
        )
        assert len(history["train_loss"]) == 1
        assert (tmp_path / "best_explainer.pth").exists()

    def test_surrogate_device_none_defaults_to_device(
        self, tiny_surrogate, train_loader, val_loader, device
    ):
        """surrogate_device=None should behave identically to omitting it."""
        explainer1 = build_vit_explainer(
            _TINY_MODEL,
            num_classes=_NUM_CLASSES,
            num_attn_blocks=0,
            num_mlp_layers=1,
            normalization="additive",
            activation=None,
        )
        explainer2 = build_vit_explainer(
            _TINY_MODEL,
            num_classes=_NUM_CLASSES,
            num_attn_blocks=0,
            num_mlp_layers=1,
            normalization="additive",
            activation=None,
        )
        explainer2.load_state_dict(explainer1.state_dict())
        surrogate2 = build_vit_surrogate(_TINY_MODEL, num_classes=_NUM_CLASSES)
        surrogate2.load_state_dict(tiny_surrogate.state_dict())

        torch.manual_seed(0)
        h1 = train_explainer(
            explainer1,
            tiny_surrogate,
            train_loader,
            val_loader,
            epochs=1,
            device=device,
            use_amp=False,
        )
        torch.manual_seed(0)
        h2 = train_explainer(
            explainer2,
            surrogate2,
            train_loader,
            val_loader,
            epochs=1,
            device=device,
            surrogate_device=None,
            use_amp=False,
        )
        assert abs(h1["val_loss"][0] - h2["val_loss"][0]) < 1e-6
