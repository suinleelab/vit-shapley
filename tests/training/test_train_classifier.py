"""Tests for vit_shapley.training.train_classifier.

Uses tiny synthetic datasets (no ImageNette download needed) and
vit_tiny_patch16_224 (no pretrained weights) for speed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from vit_shapley.models.classifier import build_vit_classifier
from vit_shapley.training.train_classifier import evaluate, train_classifier, train_one_epoch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TINY_MODEL = "vit_tiny_patch16_224"
_IMAGE_SIZE = 224
_NUM_CLASSES = 3
_BATCH_SIZE = 2
_NUM_SAMPLES = 6  # divisible by batch size


def _make_synthetic_dataset(
    num_samples: int = _NUM_SAMPLES,
    num_classes: int = _NUM_CLASSES,
    image_size: int = _IMAGE_SIZE,
) -> TensorDataset:
    images = torch.randn(num_samples, 3, image_size, image_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(images, labels)


@pytest.fixture
def tiny_model():
    return build_vit_classifier(_TINY_MODEL, num_classes=_NUM_CLASSES, pretrained=False)


@pytest.fixture
def train_loader():
    dataset = _make_synthetic_dataset()
    return DataLoader(dataset, batch_size=_BATCH_SIZE, shuffle=True)


@pytest.fixture
def val_loader():
    dataset = _make_synthetic_dataset(num_samples=4)
    return DataLoader(dataset, batch_size=_BATCH_SIZE, shuffle=False)


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def criterion():
    return nn.CrossEntropyLoss()


# ---------------------------------------------------------------------------
# train_one_epoch tests
# ---------------------------------------------------------------------------

class TestTrainOneEpoch:
    def test_returns_loss_and_acc(self, tiny_model, train_loader, device, criterion):
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
        metrics = train_one_epoch(
            tiny_model, train_loader, optimizer, criterion, device, scaler=None
        )
        assert "loss" in metrics
        assert "acc" in metrics

    def test_loss_is_positive_float(self, tiny_model, train_loader, device, criterion):
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
        metrics = train_one_epoch(
            tiny_model, train_loader, optimizer, criterion, device, scaler=None
        )
        assert isinstance(metrics["loss"], float)
        assert metrics["loss"] > 0.0

    def test_acc_in_valid_range(self, tiny_model, train_loader, device, criterion):
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
        metrics = train_one_epoch(
            tiny_model, train_loader, optimizer, criterion, device, scaler=None
        )
        assert 0.0 <= metrics["acc"] <= 1.0

    def test_model_updates_weights(self, tiny_model, train_loader, device, criterion):
        """Weights should change after one training step."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        # snapshot weights before
        before = {
            name: param.clone()
            for name, param in tiny_model.named_parameters()
            if param.requires_grad
        }
        train_one_epoch(tiny_model, train_loader, optimizer, criterion, device, scaler=None)
        after = dict(tiny_model.named_parameters())
        changed = any(
            not torch.allclose(before[n], after[n]) for n in before
        )
        assert changed, "No parameter changed after train_one_epoch"


# ---------------------------------------------------------------------------
# evaluate tests
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_returns_loss_and_acc(self, tiny_model, val_loader, device, criterion):
        metrics = evaluate(tiny_model, val_loader, criterion, device)
        assert "loss" in metrics
        assert "acc" in metrics

    def test_loss_positive(self, tiny_model, val_loader, device, criterion):
        metrics = evaluate(tiny_model, val_loader, criterion, device)
        assert isinstance(metrics["loss"], float)
        assert metrics["loss"] > 0.0

    def test_acc_in_valid_range(self, tiny_model, val_loader, device, criterion):
        metrics = evaluate(tiny_model, val_loader, criterion, device)
        assert 0.0 <= metrics["acc"] <= 1.0

    def test_no_gradient_updates(self, tiny_model, val_loader, device, criterion):
        """evaluate() must not modify model weights."""
        before = {
            name: param.clone()
            for name, param in tiny_model.named_parameters()
        }
        evaluate(tiny_model, val_loader, criterion, device)
        after = dict(tiny_model.named_parameters())
        for name in before:
            assert torch.allclose(before[name], after[name]), (
                f"Parameter '{name}' changed during evaluate()"
            )

    def test_deterministic_result(self, tiny_model, val_loader, device, criterion):
        """Calling evaluate twice on same inputs/weights should give same result."""
        m1 = evaluate(tiny_model, val_loader, criterion, device)
        m2 = evaluate(tiny_model, val_loader, criterion, device)
        assert abs(m1["loss"] - m2["loss"]) < 1e-6
        assert abs(m1["acc"] - m2["acc"]) < 1e-6


# ---------------------------------------------------------------------------
# train_classifier tests
# ---------------------------------------------------------------------------

class TestTrainClassifier:
    def test_returns_history_keys(self, tiny_model, train_loader, val_loader, device):
        history = train_classifier(
            tiny_model, train_loader, val_loader,
            epochs=1, lr=1e-4, device=device, save_dir=None, use_amp=False
        )
        for key in ("train_loss", "train_acc", "val_loss", "val_acc",
                    "best_val_acc", "best_epoch"):
            assert key in history, f"Missing key '{key}' in history"

    def test_history_length_matches_epochs(self, tiny_model, train_loader, val_loader, device):
        epochs = 2
        history = train_classifier(
            tiny_model, train_loader, val_loader,
            epochs=epochs, lr=1e-4, device=device, save_dir=None, use_amp=False
        )
        for key in ("train_loss", "train_acc", "val_loss", "val_acc"):
            assert len(history[key]) == epochs, (
                f"Expected {epochs} entries for '{key}', got {len(history[key])}"
            )

    def test_best_val_acc_is_max(self, tiny_model, train_loader, val_loader, device):
        history = train_classifier(
            tiny_model, train_loader, val_loader,
            epochs=2, lr=1e-4, device=device, save_dir=None, use_amp=False
        )
        assert history["best_val_acc"] == max(history["val_acc"])

    def test_checkpoint_saved(self, tmp_path, train_loader, val_loader, device):
        model = build_vit_classifier(_TINY_MODEL, num_classes=_NUM_CLASSES, pretrained=False)
        history = train_classifier(
            model, train_loader, val_loader,
            epochs=2, lr=1e-4, device=device, save_dir=tmp_path, use_amp=False
        )
        ckpt_path = tmp_path / "best_classifier.pth"
        assert ckpt_path.exists(), "best_classifier.pth not created"

    def test_checkpoint_loadable(self, tmp_path, train_loader, val_loader, device):
        model = build_vit_classifier(_TINY_MODEL, num_classes=_NUM_CLASSES, pretrained=False)
        train_classifier(
            model, train_loader, val_loader,
            epochs=1, lr=1e-4, device=device, save_dir=tmp_path, use_amp=False
        )
        ckpt_path = tmp_path / "best_classifier.pth"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert "model_state_dict" in ckpt
        assert "val_acc" in ckpt
        assert "epoch" in ckpt

        # Verify state dict can be loaded back into a fresh model
        fresh_model = build_vit_classifier(
            _TINY_MODEL, num_classes=_NUM_CLASSES, pretrained=False
        )
        fresh_model.load_state_dict(ckpt["model_state_dict"])

    def test_no_save_dir_skips_checkpoint(self, tiny_model, train_loader, val_loader, device, tmp_path):
        """No checkpoint should appear when save_dir=None."""
        train_classifier(
            tiny_model, train_loader, val_loader,
            epochs=1, lr=1e-4, device=device, save_dir=None, use_amp=False
        )
        # Ensure nothing was written in cwd
        assert not (Path(".") / "best_classifier.pth").exists()

    def test_val_loss_positive(self, tiny_model, train_loader, val_loader, device):
        history = train_classifier(
            tiny_model, train_loader, val_loader,
            epochs=1, lr=1e-4, device=device, save_dir=None, use_amp=False
        )
        assert all(loss > 0.0 for loss in history["val_loss"])

    def test_default_no_label_smoothing(self, train_loader, val_loader, device):
        """train_classifier must use plain CrossEntropyLoss (no label smoothing)."""
        import inspect
        from vit_shapley.training.train_classifier import train_classifier as tc
        src = inspect.getsource(tc)
        assert "label_smoothing" not in src, (
            "train_classifier should not use label_smoothing (removed in fix 4)"
        )

    def test_default_hyperparameters(self):
        """Defaults must match paper: lr=1e-5, weight_decay=1e-5, epochs=25, warmup_steps=500."""
        import inspect
        sig = inspect.signature(train_classifier)
        params = sig.parameters
        assert params["lr"].default == 1e-5, f"lr default: {params['lr'].default}"
        assert params["weight_decay"].default == 1e-5, f"wd default: {params['weight_decay'].default}"
        assert params["epochs"].default == 25, f"epochs default: {params['epochs'].default}"
        assert params["warmup_steps"].default == 500, f"warmup_steps default: {params['warmup_steps'].default}"


# ---------------------------------------------------------------------------
# _cosine_with_warmup tests
# ---------------------------------------------------------------------------

class TestCosineWithWarmup:
    def test_import(self):
        from vit_shapley.training.train_classifier import _cosine_with_warmup
        assert callable(_cosine_with_warmup)

    def test_zero_at_step_zero(self):
        from vit_shapley.training.train_classifier import _cosine_with_warmup
        assert _cosine_with_warmup(0, warmup_steps=500, total_steps=1000) == 0.0

    def test_one_at_warmup_end(self):
        from vit_shapley.training.train_classifier import _cosine_with_warmup
        val = _cosine_with_warmup(500, warmup_steps=500, total_steps=1000)
        assert abs(val - 1.0) < 1e-6

    def test_linear_in_warmup(self):
        from vit_shapley.training.train_classifier import _cosine_with_warmup
        warmup, total = 100, 200
        for step in range(1, warmup):
            expected = step / warmup
            got = _cosine_with_warmup(step, warmup, total)
            assert abs(got - expected) < 1e-6, f"step={step}: expected {expected}, got {got}"

    def test_near_zero_at_end(self):
        from vit_shapley.training.train_classifier import _cosine_with_warmup
        val = _cosine_with_warmup(1000, warmup_steps=100, total_steps=1000)
        assert val < 0.01  # cosine decay reaches near 0

    def test_monotone_decrease_after_warmup(self):
        from vit_shapley.training.train_classifier import _cosine_with_warmup
        warmup, total = 100, 500
        vals = [_cosine_with_warmup(s, warmup, total) for s in range(warmup, total + 1)]
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)), (
            "LR multiplier must be monotonically non-increasing after warmup"
        )

    def test_scheduler_steps_per_batch(self, tiny_model, train_loader, val_loader, device):
        """Scheduler must be stepped once per batch, not per epoch."""
        from vit_shapley.training.train_classifier import _cosine_with_warmup
        steps_per_epoch = len(train_loader)
        epochs = 2
        total_steps = epochs * steps_per_epoch
        warmup = 0  # disable warmup to see clear cosine behaviour
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda s: _cosine_with_warmup(s, warmup, total_steps),
        )
        lrs = []
        criterion = nn.CrossEntropyLoss()
        from vit_shapley.training.train_classifier import train_one_epoch
        tiny_model.train()
        for _ in range(epochs):
            for images, labels in train_loader:
                optimizer.zero_grad()
                loss = criterion(tiny_model(images), labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                lrs.append(optimizer.param_groups[0]["lr"])
        # LR should vary across batches (not be constant within an epoch)
        assert len(set(round(lr, 10) for lr in lrs)) > 1, (
            "LR did not change across batches — scheduler may not be stepping per batch"
        )
