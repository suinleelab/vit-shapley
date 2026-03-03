"""Tests for vit_shapley.training.train_surrogate.

Uses tiny synthetic datasets (no ImageNette download) and vit_tiny_patch16_224
without pretrained weights for speed.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from vit_shapley.models.classifier import build_vit_classifier
from vit_shapley.models.surrogate import build_vit_surrogate
from vit_shapley.training.train_surrogate import (
    _cosine_schedule_with_warmup,
    evaluate_surrogate,
    sample_subset_masks,
    train_one_epoch_surrogate,
    train_surrogate,
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


def _make_binary_dataset(
    num_samples: int = _NUM_SAMPLES,
) -> TensorDataset:
    images = torch.randn(num_samples, 3, _IMAGE_SIZE, _IMAGE_SIZE)
    labels = torch.randint(0, 2, (num_samples,))
    return TensorDataset(images, labels)


@pytest.fixture
def tiny_surrogate():
    return build_vit_surrogate(_TINY_MODEL, num_classes=_NUM_CLASSES)


@pytest.fixture
def tiny_classifier():
    return build_vit_classifier(_TINY_MODEL, num_classes=_NUM_CLASSES, pretrained=False)


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
# sample_subset_masks
# ---------------------------------------------------------------------------


class TestSampleSubsetMasks:
    def test_output_shape(self, device):
        masks = sample_subset_masks(4, _NUM_PATCHES, device)
        assert masks.shape == (4, _NUM_PATCHES)

    def test_values_binary(self, device):
        masks = sample_subset_masks(8, _NUM_PATCHES, device)
        unique = masks.unique()
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_cardinality_in_range(self, device):
        masks = sample_subset_masks(32, _NUM_PATCHES, device)
        row_sums = masks.sum(dim=1)
        assert (row_sums >= 0).all()
        assert (row_sums <= _NUM_PATCHES).all()

    def test_varying_cardinalities(self, device):
        """Over many samples, cardinalities should vary (not all the same)."""
        masks = sample_subset_masks(64, _NUM_PATCHES, device)
        row_sums = masks.sum(dim=1)
        assert row_sums.unique().numel() > 1

    def test_device_matches(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")
        masks = sample_subset_masks(4, _NUM_PATCHES, device)
        assert masks.device.type == "cuda"

    def test_all_zero_cardinality_possible(self, device):
        """Masks with m=0 (all zeros) should be possible.

        Use num_patches=5 so P(m=0) = 1/6 ≈ 16.7%; the probability of never
        sampling it in 100 tries is (5/6)^100 < 1e-7.
        """
        found_zero = False
        for _ in range(100):
            masks = sample_subset_masks(1, 5, device)
            if masks.sum() == 0:
                found_zero = True
                break
        assert found_zero, "Never sampled an all-zero mask in 100 tries"

    def test_generator_reproducibility(self, device):
        """Same seeded generator → identical masks."""
        gen1 = torch.Generator(device=device)
        gen1.manual_seed(42)
        masks1 = sample_subset_masks(4, _NUM_PATCHES, device, generator=gen1)

        gen2 = torch.Generator(device=device)
        gen2.manual_seed(42)
        masks2 = sample_subset_masks(4, _NUM_PATCHES, device, generator=gen2)

        assert torch.equal(masks1, masks2)

    def test_different_seeds_differ(self, device):
        """Different seeds should (almost certainly) produce different masks."""
        gen1 = torch.Generator(device=device)
        gen1.manual_seed(0)
        gen2 = torch.Generator(device=device)
        gen2.manual_seed(1)
        masks1 = sample_subset_masks(4, _NUM_PATCHES, device, generator=gen1)
        masks2 = sample_subset_masks(4, _NUM_PATCHES, device, generator=gen2)
        assert not torch.equal(masks1, masks2)


# ---------------------------------------------------------------------------
# _cosine_schedule_with_warmup
# ---------------------------------------------------------------------------


class TestCosineScheduleWithWarmup:
    def test_lr_increases_during_warmup(self):
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        scheduler = _cosine_schedule_with_warmup(
            optimizer, warmup_steps=10, total_steps=100
        )
        lrs = []
        for _ in range(10):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
        # LR should be monotonically increasing during warmup
        assert all(lrs[i] <= lrs[i + 1] for i in range(len(lrs) - 1))

    def test_lr_peaks_at_warmup_end(self):
        model = torch.nn.Linear(4, 2)
        base_lr = 1.0
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
        warmup_steps = 5
        scheduler = _cosine_schedule_with_warmup(
            optimizer, warmup_steps=warmup_steps, total_steps=50
        )
        for _ in range(warmup_steps):
            scheduler.step()
        assert abs(optimizer.param_groups[0]["lr"] - base_lr) < 1e-6

    def test_lr_decreases_after_warmup(self):
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        warmup_steps = 5
        total_steps = 50
        scheduler = _cosine_schedule_with_warmup(
            optimizer, warmup_steps=warmup_steps, total_steps=total_steps
        )
        for _ in range(warmup_steps):
            scheduler.step()
        peak_lr = optimizer.param_groups[0]["lr"]
        lrs_after = []
        for _ in range(total_steps - warmup_steps):
            scheduler.step()
            lrs_after.append(optimizer.param_groups[0]["lr"])
        # LR should be strictly below peak after warmup
        assert all(lr <= peak_lr for lr in lrs_after)
        # And should decrease monotonically
        assert all(lrs_after[i] >= lrs_after[i + 1] for i in range(len(lrs_after) - 1))

    def test_lr_reaches_near_zero_at_end(self):
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        total_steps = 20
        scheduler = _cosine_schedule_with_warmup(
            optimizer, warmup_steps=2, total_steps=total_steps
        )
        for _ in range(total_steps):
            scheduler.step()
        assert optimizer.param_groups[0]["lr"] < 0.01


# ---------------------------------------------------------------------------
# train_one_epoch_surrogate
# ---------------------------------------------------------------------------


class TestTrainOneEpochSurrogate:
    def test_returns_valid_metrics(
        self, tiny_surrogate, tiny_classifier, train_loader, device
    ):
        optimizer = torch.optim.AdamW(tiny_surrogate.parameters(), lr=1e-4)
        metrics = train_one_epoch_surrogate(
            tiny_surrogate, tiny_classifier, train_loader, optimizer, device
        )
        assert "loss" in metrics and "acc" in metrics
        assert isinstance(metrics["loss"], float) and metrics["loss"] > 0.0
        assert 0.0 <= metrics["acc"] <= 1.0

    def test_surrogate_weights_update(
        self, tiny_surrogate, tiny_classifier, train_loader, device
    ):
        optimizer = torch.optim.AdamW(tiny_surrogate.parameters(), lr=1e-3)
        before = {n: p.clone() for n, p in tiny_surrogate.named_parameters()}
        train_one_epoch_surrogate(
            tiny_surrogate, tiny_classifier, train_loader, optimizer, device
        )
        after = dict(tiny_surrogate.named_parameters())
        changed = any(not torch.allclose(before[n], after[n]) for n in before)
        assert changed, "No surrogate parameter changed after training"

    def test_classifier_weights_frozen(
        self, tiny_surrogate, tiny_classifier, train_loader, device
    ):
        optimizer = torch.optim.AdamW(tiny_surrogate.parameters(), lr=1e-3)
        before = {n: p.clone() for n, p in tiny_classifier.named_parameters()}
        train_one_epoch_surrogate(
            tiny_surrogate, tiny_classifier, train_loader, optimizer, device
        )
        after = dict(tiny_classifier.named_parameters())
        for n in before:
            assert torch.allclose(before[n], after[n]), (
                f"Classifier param '{n}' changed"
            )

    def test_scheduler_steps_each_batch(
        self, tiny_surrogate, tiny_classifier, train_loader, device
    ):
        """Scheduler should step once per gradient update, not once per epoch."""
        optimizer = torch.optim.AdamW(tiny_surrogate.parameters(), lr=1e-3)
        # A short warmup so LR is still rising; we can verify it moved.
        num_batches = len(train_loader)
        total_steps = num_batches * 10
        scheduler = _cosine_schedule_with_warmup(
            optimizer, warmup_steps=total_steps, total_steps=total_steps
        )
        lr_before = optimizer.param_groups[0]["lr"]
        train_one_epoch_surrogate(
            tiny_surrogate,
            tiny_classifier,
            train_loader,
            optimizer,
            device,
            scheduler=scheduler,
        )
        lr_after = optimizer.param_groups[0]["lr"]
        # LR should have increased (still in warmup phase)
        assert lr_after > lr_before

    def test_scheduler_none_still_trains(
        self, tiny_surrogate, tiny_classifier, train_loader, device
    ):
        """Passing scheduler=None should not raise and training should proceed."""
        optimizer = torch.optim.AdamW(tiny_surrogate.parameters(), lr=1e-4)
        metrics = train_one_epoch_surrogate(
            tiny_surrogate,
            tiny_classifier,
            train_loader,
            optimizer,
            device,
            scheduler=None,
        )
        assert "loss" in metrics


# ---------------------------------------------------------------------------
# evaluate_surrogate
# ---------------------------------------------------------------------------


class TestEvaluateSurrogate:
    def test_returns_valid_metrics(
        self, tiny_surrogate, tiny_classifier, val_loader, device
    ):
        metrics = evaluate_surrogate(
            tiny_surrogate, tiny_classifier, val_loader, device
        )
        assert "loss" in metrics and "acc" in metrics
        assert isinstance(metrics["loss"], float) and metrics["loss"] > 0.0
        assert 0.0 <= metrics["acc"] <= 1.0

    def test_no_gradient_updates(
        self, tiny_surrogate, tiny_classifier, val_loader, device
    ):
        before_s = {n: p.clone() for n, p in tiny_surrogate.named_parameters()}
        before_c = {n: p.clone() for n, p in tiny_classifier.named_parameters()}
        evaluate_surrogate(tiny_surrogate, tiny_classifier, val_loader, device)
        for n in before_s:
            assert torch.allclose(
                before_s[n], dict(tiny_surrogate.named_parameters())[n]
            )
        for n in before_c:
            assert torch.allclose(
                before_c[n], dict(tiny_classifier.named_parameters())[n]
            )

    def test_val_loss_reproducible_same_seed(
        self, tiny_surrogate, tiny_classifier, val_loader, device
    ):
        """Two calls with the same val_seed must return identical loss."""
        m1 = evaluate_surrogate(
            tiny_surrogate, tiny_classifier, val_loader, device, val_seed=7
        )
        m2 = evaluate_surrogate(
            tiny_surrogate, tiny_classifier, val_loader, device, val_seed=7
        )
        assert m1["loss"] == m2["loss"]

    def test_val_loss_differs_across_seeds(
        self, tiny_surrogate, tiny_classifier, val_loader, device
    ):
        """Different seeds should produce different masks and (almost certainly) different KL."""
        m0 = evaluate_surrogate(
            tiny_surrogate, tiny_classifier, val_loader, device, val_seed=0
        )
        m1 = evaluate_surrogate(
            tiny_surrogate, tiny_classifier, val_loader, device, val_seed=999
        )
        # With 4 val samples and 196 patches the masks will almost surely differ
        assert m0["loss"] != m1["loss"]


# ---------------------------------------------------------------------------
# train_surrogate
# ---------------------------------------------------------------------------


class TestTrainSurrogate:
    def test_returns_history_keys(
        self, tiny_surrogate, tiny_classifier, train_loader, val_loader, device
    ):
        history = train_surrogate(
            tiny_surrogate,
            tiny_classifier,
            train_loader,
            val_loader,
            epochs=1,
            lr=1e-4,
            device=device,
            save_dir=None,
            use_amp=False,
        )
        for key in ("train_loss", "val_loss", "val_acc", "best_val_loss", "best_epoch"):
            assert key in history, f"Missing key '{key}'"

    def test_history_length_matches_epochs(
        self, tiny_surrogate, tiny_classifier, train_loader, val_loader, device
    ):
        epochs = 2
        history = train_surrogate(
            tiny_surrogate,
            tiny_classifier,
            train_loader,
            val_loader,
            epochs=epochs,
            lr=1e-4,
            device=device,
            save_dir=None,
            use_amp=False,
        )
        for key in ("train_loss", "val_loss", "val_acc"):
            assert len(history[key]) == epochs

    def test_best_val_loss_is_min(
        self, tiny_surrogate, tiny_classifier, train_loader, val_loader, device
    ):
        history = train_surrogate(
            tiny_surrogate,
            tiny_classifier,
            train_loader,
            val_loader,
            epochs=2,
            lr=1e-4,
            device=device,
            save_dir=None,
            use_amp=False,
        )
        assert history["best_val_loss"] == min(history["val_loss"])

    def test_checkpoint_loadable(
        self, tmp_path, tiny_classifier, train_loader, val_loader, device
    ):
        surrogate = build_vit_surrogate(_TINY_MODEL, num_classes=_NUM_CLASSES)
        train_surrogate(
            surrogate,
            tiny_classifier,
            train_loader,
            val_loader,
            epochs=1,
            lr=1e-4,
            device=device,
            save_dir=tmp_path,
            use_amp=False,
        )
        ckpt = torch.load(
            tmp_path / "best_surrogate.pth", map_location="cpu", weights_only=True
        )
        assert "model_state_dict" in ckpt
        assert "val_loss" in ckpt

        fresh = build_vit_surrogate(_TINY_MODEL, num_classes=_NUM_CLASSES)
        fresh.load_state_dict(ckpt["model_state_dict"])

    def test_no_save_dir_skips_checkpoint(
        self,
        tmp_path,
        tiny_surrogate,
        tiny_classifier,
        train_loader,
        val_loader,
        device,
    ):
        train_surrogate(
            tiny_surrogate,
            tiny_classifier,
            train_loader,
            val_loader,
            epochs=1,
            lr=1e-4,
            device=device,
            save_dir=None,
            use_amp=False,
        )
        assert not (tmp_path / "best_surrogate.pth").exists()

    def test_val_loss_positive(
        self, tiny_surrogate, tiny_classifier, train_loader, val_loader, device
    ):
        history = train_surrogate(
            tiny_surrogate,
            tiny_classifier,
            train_loader,
            val_loader,
            epochs=1,
            lr=1e-4,
            device=device,
            save_dir=None,
            use_amp=False,
        )
        assert all(loss > 0.0 for loss in history["val_loss"])

    def test_warmup_steps_param_accepted(
        self, tiny_surrogate, tiny_classifier, train_loader, val_loader, device
    ):
        """warmup_steps kwarg should be accepted and not raise."""
        history = train_surrogate(
            tiny_surrogate,
            tiny_classifier,
            train_loader,
            val_loader,
            epochs=1,
            lr=1e-4,
            warmup_steps=2,
            device=device,
            save_dir=None,
            use_amp=False,
        )
        assert "train_loss" in history

    def test_val_loss_reproducible_across_epochs(
        self, tiny_classifier, train_loader, val_loader, device
    ):
        """Val loss reported by train_surrogate must exactly match a standalone
        evaluate_surrogate call using the same (post-training) model and val_seed.

        After train_surrogate completes, the surrogate holds the post-training
        weights.  A direct evaluate_surrogate call with val_seed=0 (the same
        seed train_surrogate uses internally) must reproduce the recorded loss.
        """
        surrogate = build_vit_surrogate(_TINY_MODEL, num_classes=_NUM_CLASSES)
        h1 = train_surrogate(
            surrogate,
            tiny_classifier,
            train_loader,
            val_loader,
            epochs=1,
            lr=1e-4,
            device=device,
            save_dir=None,
            use_amp=False,
        )
        # surrogate now has post-training weights; same seed → same masks → same loss
        val_loss_direct = evaluate_surrogate(
            surrogate, tiny_classifier, val_loader, device, val_seed=0
        )["loss"]
        assert abs(h1["val_loss"][0] - val_loss_direct) < 1e-6

    def test_classifier_device_none_defaults_to_device(
        self, tiny_surrogate, tiny_classifier, train_loader, val_loader, device
    ):
        """classifier_device=None should produce identical results to omitting it."""
        surrogate1 = build_vit_surrogate(_TINY_MODEL, num_classes=_NUM_CLASSES)
        surrogate2 = build_vit_surrogate(_TINY_MODEL, num_classes=_NUM_CLASSES)
        # Sync weights
        surrogate2.load_state_dict(surrogate1.state_dict())
        classifier2 = build_vit_classifier(
            _TINY_MODEL, num_classes=_NUM_CLASSES, pretrained=False
        )
        classifier2.load_state_dict(tiny_classifier.state_dict())

        torch.manual_seed(0)
        h1 = train_surrogate(
            surrogate1,
            tiny_classifier,
            train_loader,
            val_loader,
            epochs=1,
            lr=1e-4,
            device=device,
            save_dir=None,
            use_amp=False,
        )
        torch.manual_seed(0)
        h2 = train_surrogate(
            surrogate2,
            classifier2,
            train_loader,
            val_loader,
            epochs=1,
            lr=1e-4,
            device=device,
            classifier_device=None,
            save_dir=None,
            use_amp=False,
        )
        assert abs(h1["val_loss"][0] - h2["val_loss"][0]) < 1e-6


# ---------------------------------------------------------------------------
# Binary surrogate training tests
# ---------------------------------------------------------------------------


class TestBinaryTrainOneEpochSurrogate:
    def test_returns_valid_metrics(self, device):
        surrogate = build_vit_surrogate(_TINY_MODEL, num_classes=1)
        classifier = build_vit_classifier(
            _TINY_MODEL, num_classes=1, pretrained=False
        )
        loader = DataLoader(
            _make_binary_dataset(), batch_size=_BATCH_SIZE, shuffle=True
        )
        optimizer = torch.optim.AdamW(surrogate.parameters(), lr=1e-4)
        metrics = train_one_epoch_surrogate(
            surrogate, classifier, loader, optimizer, device,
            target_type="binary",
        )
        assert "loss" in metrics and "acc" in metrics
        assert isinstance(metrics["loss"], float) and metrics["loss"] > 0.0
        assert 0.0 <= metrics["acc"] <= 1.0

    def test_surrogate_weights_update(self, device):
        surrogate = build_vit_surrogate(_TINY_MODEL, num_classes=1)
        classifier = build_vit_classifier(
            _TINY_MODEL, num_classes=1, pretrained=False
        )
        loader = DataLoader(
            _make_binary_dataset(), batch_size=_BATCH_SIZE, shuffle=True
        )
        optimizer = torch.optim.AdamW(surrogate.parameters(), lr=1e-3)
        before = {n: p.clone() for n, p in surrogate.named_parameters()}
        train_one_epoch_surrogate(
            surrogate, classifier, loader, optimizer, device,
            target_type="binary",
        )
        after = dict(surrogate.named_parameters())
        changed = any(not torch.allclose(before[n], after[n]) for n in before)
        assert changed, "No surrogate parameter changed after binary training"


class TestBinaryEvaluateSurrogate:
    def test_returns_valid_metrics(self, device):
        surrogate = build_vit_surrogate(_TINY_MODEL, num_classes=1)
        classifier = build_vit_classifier(
            _TINY_MODEL, num_classes=1, pretrained=False
        )
        loader = DataLoader(
            _make_binary_dataset(num_samples=4), batch_size=_BATCH_SIZE
        )
        metrics = evaluate_surrogate(
            surrogate, classifier, loader, device, target_type="binary",
        )
        assert "loss" in metrics and "acc" in metrics
        assert isinstance(metrics["loss"], float) and metrics["loss"] > 0.0
        assert 0.0 <= metrics["acc"] <= 1.0

    def test_val_loss_reproducible(self, device):
        surrogate = build_vit_surrogate(_TINY_MODEL, num_classes=1)
        classifier = build_vit_classifier(
            _TINY_MODEL, num_classes=1, pretrained=False
        )
        loader = DataLoader(
            _make_binary_dataset(num_samples=4), batch_size=_BATCH_SIZE
        )
        m1 = evaluate_surrogate(
            surrogate, classifier, loader, device,
            val_seed=7, target_type="binary",
        )
        m2 = evaluate_surrogate(
            surrogate, classifier, loader, device,
            val_seed=7, target_type="binary",
        )
        assert m1["loss"] == m2["loss"]


class TestBinaryTrainSurrogate:
    def test_returns_history_keys(self, device):
        surrogate = build_vit_surrogate(_TINY_MODEL, num_classes=1)
        classifier = build_vit_classifier(
            _TINY_MODEL, num_classes=1, pretrained=False
        )
        train_ldr = DataLoader(
            _make_binary_dataset(), batch_size=_BATCH_SIZE, shuffle=True
        )
        val_ldr = DataLoader(
            _make_binary_dataset(num_samples=4), batch_size=_BATCH_SIZE
        )
        history = train_surrogate(
            surrogate, classifier, train_ldr, val_ldr,
            epochs=1, lr=1e-4, device=device, save_dir=None, use_amp=False,
            target_type="binary",
        )
        for key in ("train_loss", "val_loss", "val_acc", "best_val_loss", "best_epoch"):
            assert key in history, f"Missing key '{key}'"

    def test_checkpoint_loadable(self, tmp_path, device):
        surrogate = build_vit_surrogate(_TINY_MODEL, num_classes=1)
        classifier = build_vit_classifier(
            _TINY_MODEL, num_classes=1, pretrained=False
        )
        train_ldr = DataLoader(
            _make_binary_dataset(), batch_size=_BATCH_SIZE, shuffle=True
        )
        val_ldr = DataLoader(
            _make_binary_dataset(num_samples=4), batch_size=_BATCH_SIZE
        )
        train_surrogate(
            surrogate, classifier, train_ldr, val_ldr,
            epochs=1, lr=1e-4, device=device, save_dir=tmp_path, use_amp=False,
            target_type="binary",
        )
        ckpt = torch.load(
            tmp_path / "best_surrogate.pth", map_location="cpu", weights_only=True
        )
        assert "model_state_dict" in ckpt
        fresh = build_vit_surrogate(_TINY_MODEL, num_classes=1)
        fresh.load_state_dict(ckpt["model_state_dict"])
