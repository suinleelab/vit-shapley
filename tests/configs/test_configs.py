"""Tests for individual Pydantic config classes."""

import pytest
from pydantic import ValidationError

from vit_shapley.configs import (
    ClassifierConfig,
    ExplainerConfig,
    PlotConfig,
    SurrogateConfig,
    VisualizeConfig,
)

# ---------------------------------------------------------------------------
# ClassifierConfig
# ---------------------------------------------------------------------------


class TestClassifierConfig:
    def test_defaults(self):
        cfg = ClassifierConfig()
        assert cfg.target_type == "multiclass"
        assert cfg.dataset == "imagenette"
        assert cfg.model_name == "vit_base_patch16_224"
        assert cfg.pretrained is True
        assert cfg.epochs == 25
        assert cfg.batch_size == 64
        assert abs(cfg.lr - 1e-5) < 1e-12
        assert abs(cfg.weight_decay - 1e-5) < 1e-12
        assert cfg.warmup_steps == 500
        assert cfg.num_workers == 4
        assert cfg.image_size == 224
        assert cfg.save_dir == "checkpoints/classifier"
        assert cfg.use_amp is True
        assert cfg.device == ""

    def test_target_type_binary(self):
        cfg = ClassifierConfig(target_type="binary")
        assert cfg.target_type == "binary"

    def test_dataset_pet(self):
        cfg = ClassifierConfig(dataset="pet")
        assert cfg.dataset == "pet"

    def test_full_construction(self):
        cfg = ClassifierConfig(
            dataset="pet",
            data_root="/data",
            model_name="vit_tiny_patch16_224",
            pretrained=False,
            epochs=10,
            batch_size=32,
            lr=1e-3,
            weight_decay=1e-2,
            warmup_steps=100,
            num_workers=2,
            image_size=224,
            save_dir="checkpoints/test",
            use_amp=False,
            device="cpu",
        )
        assert cfg.dataset == "pet"
        assert cfg.model_name == "vit_tiny_patch16_224"
        assert cfg.pretrained is False
        assert cfg.epochs == 10
        assert cfg.device == "cpu"

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError, match="extra_field"):
            ClassifierConfig(extra_field="oops")

    def test_extra_field_model_validate_raises(self):
        with pytest.raises(ValidationError, match="typo_key"):
            ClassifierConfig.model_validate({"typo_key": 42})


# ---------------------------------------------------------------------------
# SurrogateConfig
# ---------------------------------------------------------------------------


class TestSurrogateConfig:
    def test_requires_classifier_ckpt(self):
        with pytest.raises(ValidationError):
            SurrogateConfig()

    def test_defaults(self):
        cfg = SurrogateConfig(classifier_ckpt="ckpt.pth")
        assert cfg.target_type == "multiclass"
        assert cfg.dataset == "imagenette"
        assert cfg.model_name == "vit_base_patch16_224"
        assert cfg.masking_strategy == "attn_mask"
        assert cfg.epochs == 50
        assert cfg.batch_size == 64
        assert abs(cfg.lr - 1e-5) < 1e-12
        assert cfg.warmup_steps == 500
        assert cfg.save_dir == "checkpoints/surrogate"
        assert cfg.use_amp is True
        assert cfg.device == ""
        assert cfg.classifier_device == ""

    def test_target_type_binary(self):
        cfg = SurrogateConfig(classifier_ckpt="ckpt.pth", target_type="binary")
        assert cfg.target_type == "binary"

    def test_model_validate_with_required_field(self):
        data = {"classifier_ckpt": "ckpt.pth", "epochs": 10}
        cfg = SurrogateConfig.model_validate(data)
        assert cfg.epochs == 10
        assert cfg.classifier_ckpt == "ckpt.pth"

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError, match="unknown_param"):
            SurrogateConfig(classifier_ckpt="ckpt.pth", unknown_param="oops")


# ---------------------------------------------------------------------------
# ExplainerConfig
# ---------------------------------------------------------------------------


class TestExplainerConfig:
    def test_requires_surrogate_ckpt(self):
        with pytest.raises(ValidationError):
            ExplainerConfig()

    def test_defaults(self):
        cfg = ExplainerConfig(surrogate_ckpt="ckpt.pth")
        assert cfg.target_type == "multiclass"
        assert cfg.dataset == "imagenette"
        assert cfg.model_name == "vit_base_patch16_224"
        assert cfg.epochs == 100
        assert cfg.batch_size == 64
        assert abs(cfg.lr - 1e-4) < 1e-12
        assert abs(cfg.weight_decay - 1e-5) < 1e-12
        assert cfg.warmup_steps == 500
        assert cfg.num_mask_samples == 32
        assert cfg.paired_masks is True
        assert cfg.num_workers == 4
        assert cfg.image_size == 224
        assert cfg.save_dir == "checkpoints/explainer"
        assert cfg.use_amp is True
        assert cfg.device == ""
        assert cfg.surrogate_device == ""

    def test_paired_masks_false(self):
        cfg = ExplainerConfig(surrogate_ckpt="ckpt.pth", paired_masks=False)
        assert cfg.paired_masks is False

    def test_num_mask_samples_override(self):
        cfg = ExplainerConfig(surrogate_ckpt="ckpt.pth", num_mask_samples=4)
        assert cfg.num_mask_samples == 4

    def test_masking_strategy_default(self):
        cfg = ExplainerConfig(surrogate_ckpt="ckpt.pth")
        assert cfg.masking_strategy == "attn_mask"

    def test_masking_strategy_override(self):
        cfg = ExplainerConfig(surrogate_ckpt="ckpt.pth", masking_strategy="zero_input")
        assert cfg.masking_strategy == "zero_input"

    def test_target_type_binary(self):
        cfg = ExplainerConfig(surrogate_ckpt="ckpt.pth", target_type="binary")
        assert cfg.target_type == "binary"

    def test_surrogate_device_custom(self):
        cfg = ExplainerConfig(surrogate_ckpt="ckpt.pth", surrogate_device="cuda:1")
        assert cfg.surrogate_device == "cuda:1"

    def test_model_validate_missing_required_raises(self):
        with pytest.raises(ValidationError):
            ExplainerConfig.model_validate({"epochs": 50})

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError, match="bad_key"):
            ExplainerConfig(surrogate_ckpt="ckpt.pth", bad_key=True)


# ---------------------------------------------------------------------------
# PlotConfig
# ---------------------------------------------------------------------------


class TestPlotConfig:
    def test_requires_all_three_ckpts(self):
        with pytest.raises(ValidationError):
            PlotConfig()

    def test_requires_attn_surrogate_ckpt(self):
        with pytest.raises(ValidationError):
            PlotConfig(
                classifier_ckpt="clf.pth",
                zero_surrogate_ckpt="zero.pth",
            )

    def test_requires_zero_surrogate_ckpt(self):
        with pytest.raises(ValidationError):
            PlotConfig(
                classifier_ckpt="clf.pth",
                attn_surrogate_ckpt="attn.pth",
            )

    def test_defaults(self):
        cfg = PlotConfig(
            classifier_ckpt="clf.pth",
            attn_surrogate_ckpt="attn.pth",
            zero_surrogate_ckpt="zero.pth",
        )
        assert cfg.target_type == "multiclass"
        assert cfg.dataset == "imagenette"
        assert cfg.model_name == "vit_base_patch16_224"
        assert cfg.num_images == 50
        assert cfg.num_masks == 50
        assert cfg.step == 10
        assert cfg.output == "figures/surrogate_kl.png"
        assert cfg.device == ""

    def test_full_construction(self):
        cfg = PlotConfig(
            classifier_ckpt="clf.pth",
            attn_surrogate_ckpt="attn.pth",
            zero_surrogate_ckpt="zero.pth",
            num_images=100,
            step=5,
            output="out/fig.png",
        )
        assert cfg.num_images == 100
        assert cfg.step == 5
        assert cfg.output == "out/fig.png"

    def test_target_type_binary(self):
        cfg = PlotConfig(
            classifier_ckpt="clf.pth",
            attn_surrogate_ckpt="attn.pth",
            zero_surrogate_ckpt="zero.pth",
            target_type="binary",
        )
        assert cfg.target_type == "binary"

    def test_model_validate_missing_ckpt_raises(self):
        with pytest.raises(ValidationError):
            PlotConfig.model_validate({"classifier_ckpt": "clf.pth"})

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError, match="nonsense"):
            PlotConfig(
                classifier_ckpt="clf.pth",
                attn_surrogate_ckpt="attn.pth",
                zero_surrogate_ckpt="zero.pth",
                nonsense=99,
            )


# ---------------------------------------------------------------------------
# VisualizeConfig
# ---------------------------------------------------------------------------


class TestVisualizeConfig:
    def test_requires_surrogate_and_explainer_ckpt(self):
        with pytest.raises(ValidationError):
            VisualizeConfig()

    def test_requires_explainer_ckpt(self):
        with pytest.raises(ValidationError):
            VisualizeConfig(surrogate_ckpt="surr.pth")

    def test_requires_surrogate_ckpt(self):
        with pytest.raises(ValidationError):
            VisualizeConfig(explainer_ckpt="exp.pth")

    def test_defaults(self):
        cfg = VisualizeConfig(surrogate_ckpt="surr.pth", explainer_ckpt="exp.pth")
        assert cfg.target_type == "multiclass"
        assert cfg.dataset == "imagenette"
        assert cfg.model_name == "vit_base_patch16_224"
        assert cfg.split == "val"
        assert cfg.sample_indices == [0, 1, 2, 3]
        assert cfg.class_indices is None
        assert cfg.output == "figures/shapley_heatmaps.png"
        assert cfg.image_size == 224
        assert cfg.device == ""

    def test_sample_indices_custom(self):
        cfg = VisualizeConfig(
            surrogate_ckpt="surr.pth",
            explainer_ckpt="exp.pth",
            sample_indices=[0, 5, 10, 15],
        )
        assert cfg.sample_indices == [0, 5, 10, 15]

    def test_class_indices_custom(self):
        cfg = VisualizeConfig(
            surrogate_ckpt="surr.pth",
            explainer_ckpt="exp.pth",
            class_indices=[2, 8],
        )
        assert cfg.class_indices == [2, 8]

    def test_class_indices_none(self):
        cfg = VisualizeConfig(
            surrogate_ckpt="surr.pth",
            explainer_ckpt="exp.pth",
            class_indices=None,
        )
        assert cfg.class_indices is None

    def test_masking_strategy_default(self):
        cfg = VisualizeConfig(surrogate_ckpt="surr.pth", explainer_ckpt="exp.pth")
        assert cfg.masking_strategy == "attn_mask"

    def test_masking_strategy_override(self):
        cfg = VisualizeConfig(
            surrogate_ckpt="surr.pth",
            explainer_ckpt="exp.pth",
            masking_strategy="zero_input",
        )
        assert cfg.masking_strategy == "zero_input"

    def test_split_train(self):
        cfg = VisualizeConfig(
            surrogate_ckpt="surr.pth",
            explainer_ckpt="exp.pth",
            split="train",
        )
        assert cfg.split == "train"

    def test_target_type_binary(self):
        cfg = VisualizeConfig(
            surrogate_ckpt="surr.pth",
            explainer_ckpt="exp.pth",
            target_type="binary",
        )
        assert cfg.target_type == "binary"

    def test_model_validate_missing_ckpts_raises(self):
        with pytest.raises(ValidationError):
            VisualizeConfig.model_validate({"split": "val"})

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError, match="wrong_key"):
            VisualizeConfig(
                surrogate_ckpt="surr.pth",
                explainer_ckpt="exp.pth",
                wrong_key="oops",
            )
