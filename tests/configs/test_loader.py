"""Tests for vit_shapley.configs.loader."""

import textwrap
from pathlib import Path

import pytest

from vit_shapley.configs import ClassifierConfig, SurrogateConfig
from vit_shapley.configs.loader import (
    _VAR_RE,
    _load_env,
    _parse_value,
    _resolve_variables,
    load_config,
)

# ---------------------------------------------------------------------------
# _parse_value
# ---------------------------------------------------------------------------


class TestParseValue:
    def test_int(self):
        assert _parse_value("42") == 42
        assert isinstance(_parse_value("42"), int)

    def test_negative_int(self):
        assert _parse_value("-10") == -10

    def test_float(self):
        assert _parse_value("1.5") == 1.5
        assert isinstance(_parse_value("1.5"), float)

    def test_scientific_notation(self):
        result = _parse_value("1e-3")
        assert abs(result - 1e-3) < 1e-10
        assert isinstance(result, float)

    def test_bool_true(self):
        assert _parse_value("true") is True
        assert _parse_value("True") is True
        assert _parse_value("TRUE") is True

    def test_bool_false(self):
        assert _parse_value("false") is False
        assert _parse_value("False") is False
        assert _parse_value("FALSE") is False

    def test_string_passthrough(self):
        assert _parse_value("hello") == "hello"
        assert _parse_value("vit_tiny_patch16_224") == "vit_tiny_patch16_224"


# ---------------------------------------------------------------------------
# _load_env
# ---------------------------------------------------------------------------


class TestLoadEnv:
    def test_basic_key_value(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("data_dir=/my/data\ncheckpoint_dir=checkpoints\n")
        result = _load_env(env_file)
        assert result == {"data_dir": "/my/data", "checkpoint_dir": "checkpoints"}

    def test_skips_blank_lines(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("key1=val1\n\n\nkey2=val2\n")
        result = _load_env(env_file)
        assert result == {"key1": "val1", "key2": "val2"}

    def test_skips_comments(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# this is a comment\nkey=val\n# another\n")
        result = _load_env(env_file)
        assert result == {"key": "val"}

    def test_strips_double_quotes(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('data_dir="/my/data"\n')
        result = _load_env(env_file)
        assert result == {"data_dir": "/my/data"}

    def test_strips_single_quotes(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("data_dir='/my/data'\n")
        result = _load_env(env_file)
        assert result == {"data_dir": "/my/data"}

    def test_value_with_equals_sign(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("formula=a=b\n")
        result = _load_env(env_file)
        assert result == {"formula": "a=b"}

    def test_strips_whitespace(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("  key  =  val  \n")
        result = _load_env(env_file)
        assert result == {"key": "val"}

    def test_empty_file(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("")
        result = _load_env(env_file)
        assert result == {}

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_env(tmp_path / "missing.env")

    def test_line_without_equals_skipped(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("no_equals_here\nkey=val\n")
        result = _load_env(env_file)
        assert result == {"key": "val"}


# ---------------------------------------------------------------------------
# load_config — basic YAML loading
# ---------------------------------------------------------------------------


class TestLoadConfigBasic:
    def test_loads_all_classifier_defaults(self, tmp_path):
        cfg_file = tmp_path / "classifier.yaml"
        cfg_file.write_text(
            textwrap.dedent("""\
            data_root: /data
            model_name: vit_tiny_patch16_224
            pretrained: true
            epochs: 5
            batch_size: 16
            lr: 1.0e-4
            weight_decay: 1.0e-5
            warmup_steps: 100
            num_workers: 2
            image_size: 224
            save_dir: checkpoints/test
            use_amp: false
            device: cpu
        """)
        )
        cfg = load_config(ClassifierConfig, cfg_file)
        assert cfg.model_name == "vit_tiny_patch16_224"
        assert cfg.epochs == 5
        assert cfg.batch_size == 16
        assert abs(cfg.lr - 1e-4) < 1e-10
        assert cfg.use_amp is False
        assert cfg.device == "cpu"

    def test_missing_optional_fields_use_defaults(self, tmp_path):
        """YAML with only a subset of keys -> missing keys use Pydantic defaults."""
        cfg_file = tmp_path / "partial.yaml"
        cfg_file.write_text("epochs: 3\n")
        cfg = load_config(ClassifierConfig, cfg_file)
        assert cfg.epochs == 3
        assert cfg.model_name == "vit_base_patch16_224"  # default
        assert cfg.pretrained is True  # default

    def test_empty_yaml_uses_all_defaults(self, tmp_path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        cfg = load_config(ClassifierConfig, cfg_file)
        assert cfg.epochs == 25
        assert cfg.batch_size == 64

    def test_accepts_path_object(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("epochs: 7\n")
        cfg = load_config(ClassifierConfig, Path(cfg_file))
        assert cfg.epochs == 7

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(ClassifierConfig, tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# load_config — overrides
# ---------------------------------------------------------------------------


class TestLoadConfigOverrides:
    def test_override_int(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("epochs: 5\n")
        cfg = load_config(ClassifierConfig, cfg_file, overrides=["epochs=20"])
        assert cfg.epochs == 20

    def test_override_float(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("")
        cfg = load_config(ClassifierConfig, cfg_file, overrides=["lr=1e-3"])
        assert abs(cfg.lr - 1e-3) < 1e-10

    def test_override_bool(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("use_amp: true\n")
        cfg = load_config(ClassifierConfig, cfg_file, overrides=["use_amp=false"])
        assert cfg.use_amp is False

    def test_override_string(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("")
        cfg = load_config(
            ClassifierConfig, cfg_file, overrides=["model_name=vit_tiny_patch16_224"]
        )
        assert cfg.model_name == "vit_tiny_patch16_224"

    def test_override_takes_precedence_over_yaml(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("epochs: 5\n")
        cfg = load_config(ClassifierConfig, cfg_file, overrides=["epochs=99"])
        assert cfg.epochs == 99

    def test_multiple_overrides(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("")
        cfg = load_config(
            ClassifierConfig, cfg_file, overrides=["epochs=3", "batch_size=8"]
        )
        assert cfg.epochs == 3
        assert cfg.batch_size == 8

    def test_none_overrides_is_ok(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("")
        cfg = load_config(ClassifierConfig, cfg_file, overrides=None)
        assert cfg.epochs == 25


# ---------------------------------------------------------------------------
# load_config — validation errors
# ---------------------------------------------------------------------------


class TestLoadConfigValidation:
    def test_unknown_key_raises(self, tmp_path):
        """Pydantic v2 by default raises on extra fields."""
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("nonexistent_field: 123\n")
        # Pydantic v2 with default model_config ignores extra fields;
        # but we rely on model_validate — unknown keys are silently ignored
        # by default. This test documents that behavior (no error expected).
        cfg = load_config(ClassifierConfig, cfg_file)
        assert not hasattr(cfg, "nonexistent_field")

    def test_required_field_missing_raises(self, tmp_path):
        """SurrogateConfig.classifier_ckpt has no default — missing raises."""
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("epochs: 10\n")
        with pytest.raises(Exception):  # pydantic ValidationError
            load_config(SurrogateConfig, cfg_file)

    def test_required_field_provided_via_override(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("epochs: 10\n")
        cfg = load_config(
            SurrogateConfig,
            cfg_file,
            overrides=["classifier_ckpt=checkpoints/best.pth"],
        )
        assert cfg.classifier_ckpt == "checkpoints/best.pth"
        assert cfg.epochs == 10

    def test_wrong_type_raises(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("epochs: not_an_int\n")
        with pytest.raises(Exception):
            load_config(ClassifierConfig, cfg_file)


# ---------------------------------------------------------------------------
# _resolve_variables — $var and ${var} syntax
# ---------------------------------------------------------------------------


class TestResolveVariables:
    @pytest.mark.parametrize("syntax", ["$data_dir", "${data_dir}"])
    def test_resolves_variable(self, syntax):
        data = {"data_root": syntax}
        defaults = {"data_dir": "/my/data"}
        result = _resolve_variables(data, defaults)
        assert result["data_root"] == "/my/data"

    def test_resolves_nested_path(self):
        data = {"save_dir": "$checkpoint_dir/surrogate"}
        defaults = {"checkpoint_dir": "checkpoints"}
        result = _resolve_variables(data, defaults)
        assert result["save_dir"] == "checkpoints/surrogate"

    def test_resolves_multiple_vars_in_one_value(self):
        data = {"path": "$data_dir/${checkpoint_dir}/file.pth"}
        defaults = {"data_dir": "/data", "checkpoint_dir": "ckpt"}
        result = _resolve_variables(data, defaults)
        assert result["path"] == "/data/ckpt/file.pth"

    @pytest.mark.parametrize("syntax", ["$unknown_var/data", "${unknown_var}/data"])
    def test_unresolved_var_raises(self, syntax):
        data = {"path": syntax}
        with pytest.raises(ValueError, match="unknown_var"):
            _resolve_variables(data, {"data_dir": "/my/data"})

    @pytest.mark.parametrize(
        "syntax,expected",
        [
            ("$MY_TEST_DIR/stuff", "/env/path/stuff"),
            ("${MY_TEST_DIR}/stuff", "/env/path/stuff"),
        ],
    )
    def test_falls_back_to_env_var(self, monkeypatch, syntax, expected):
        monkeypatch.setenv("MY_TEST_DIR", "/env/path")
        data = {"data_root": syntax}
        result = _resolve_variables(data, {})
        assert result["data_root"] == expected

    def test_defaults_take_priority_over_env(self, monkeypatch):
        monkeypatch.setenv("data_dir", "/from_env")
        data = {"data_root": "$data_dir"}
        result = _resolve_variables(data, {"data_dir": "/from_defaults"})
        assert result["data_root"] == "/from_defaults"

    def test_non_string_values_unchanged(self):
        data = {"epochs": 10, "use_amp": True, "lr": 1e-4}
        result = _resolve_variables(data, {"data_dir": "/data"})
        assert result["epochs"] == 10
        assert result["use_amp"] is True
        assert result["lr"] == 1e-4

    def test_no_vars_unchanged(self):
        data = {"data_root": "/absolute/path", "model": "vit_base"}
        result = _resolve_variables(data, {"data_dir": "/data"})
        assert result["data_root"] == "/absolute/path"
        assert result["model"] == "vit_base"

    def test_none_defaults_unresolved_raises(self):
        data = {"path": "$data_dir/stuff"}
        with pytest.raises(ValueError, match="data_dir"):
            _resolve_variables(data, None)

    def test_empty_data(self):
        result = _resolve_variables({}, {"data_dir": "/data"})
        assert result == {}

    def test_defaults_value_is_not_string(self):
        """Non-string defaults values are cast via str()."""
        data = {"port": "$port_num"}
        result = _resolve_variables(data, {"port_num": 8080})
        assert result["port"] == "8080"


# ---------------------------------------------------------------------------
# load_config — env_path integration
# ---------------------------------------------------------------------------


class TestLoadConfigEnv:
    def test_env_path_none_backward_compatible(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("epochs: 5\n")
        cfg = load_config(ClassifierConfig, cfg_file, env_path=None)
        assert cfg.epochs == 5

    def test_resolves_vars_from_env_file(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("data_dir=/resolved/path\ncheckpoint_dir=ckpts\n")
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("data_root: $data_dir\nsave_dir: $checkpoint_dir/cls\n")
        cfg = load_config(ClassifierConfig, cfg_file, env_path=env_file)
        assert cfg.data_root == "/resolved/path"
        assert cfg.save_dir == "ckpts/cls"

    def test_overrides_take_precedence_over_resolved(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("data_dir=/resolved/path\n")
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("data_root: $data_dir\n")
        cfg = load_config(
            ClassifierConfig,
            cfg_file,
            overrides=["data_root=/override/path"],
            env_path=env_file,
        )
        assert cfg.data_root == "/override/path"

    def test_missing_env_file_skipped_when_no_vars(self, tmp_path):
        """Missing .env is silently skipped when config has no $variables."""
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("epochs: 5\n")
        cfg = load_config(ClassifierConfig, cfg_file, env_path=tmp_path / "missing.env")
        assert cfg.epochs == 5

    def test_missing_env_file_raises_when_vars_unresolved(self, tmp_path):
        """Missing .env raises ValueError when $variables can't be resolved."""
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("data_root: $data_dir\n")
        with pytest.raises(ValueError, match="data_dir"):
            load_config(ClassifierConfig, cfg_file, env_path=tmp_path / "missing.env")

    def test_end_to_end_with_surrogate_config(self, tmp_path):
        """Full integration: .env resolves classifier_ckpt $variable."""
        env_file = tmp_path / ".env"
        env_file.write_text("checkpoint_dir=/ckpts\ndata_dir=/mydata\n")
        cfg_file = tmp_path / "surrogate.yaml"
        cfg_file.write_text(
            "classifier_ckpt: $checkpoint_dir/classifier/best.pth\n"
            "data_root: $data_dir\n"
            "save_dir: ${checkpoint_dir}/surrogate\n"
        )
        cfg = load_config(SurrogateConfig, cfg_file, env_path=env_file)
        assert cfg.classifier_ckpt == "/ckpts/classifier/best.pth"
        assert cfg.data_root == "/mydata"
        assert cfg.save_dir == "/ckpts/surrogate"

    def test_shell_env_fallback_in_load_config(self, tmp_path, monkeypatch):
        """Without .env file, $VAR resolves from environment."""
        monkeypatch.setenv("VSHAP_DATA", "/env/data")
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("data_root: $VSHAP_DATA\n")
        cfg = load_config(ClassifierConfig, cfg_file)
        assert cfg.data_root == "/env/data"

    def test_no_env_file_no_shell_env_raises(self, tmp_path):
        """Without .env file and without matching env, raises ValueError."""
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("data_root: $nonexistent_var\nepochs: 5\n")
        with pytest.raises(ValueError, match="nonexistent_var"):
            load_config(ClassifierConfig, cfg_file)


# ---------------------------------------------------------------------------
# _resolve_variables — self-referencing (config keys reference other keys)
# ---------------------------------------------------------------------------


class TestSelfReferencing:
    """Tests for config values that reference other config keys."""

    def test_basic_self_ref(self):
        """A value can reference another key in the same config dict."""
        data = {"dataset": "imagenette", "save_dir": "checkpoints/${dataset}"}
        result = _resolve_variables(data)
        assert result["save_dir"] == "checkpoints/imagenette"
        assert result["dataset"] == "imagenette"

    def test_basic_self_ref_dollar_syntax(self):
        """$var syntax also works for self-referencing."""
        data = {"dataset": "pet", "save_dir": "checkpoints/$dataset"}
        result = _resolve_variables(data)
        assert result["save_dir"] == "checkpoints/pet"

    def test_multiple_self_refs_in_one_value(self):
        """Multiple self-references in a single value."""
        data = {
            "dataset": "imagenette",
            "stage": "classifier",
            "save_dir": "ckpts/${stage}_${dataset}",
        }
        result = _resolve_variables(data)
        assert result["save_dir"] == "ckpts/classifier_imagenette"

    def test_skip_same_key(self):
        """A key does not resolve from itself (avoids infinite loop)."""
        data = {"save_dir": "${save_dir}/sub"}
        # save_dir references itself — falls through to external lookup
        with pytest.raises(ValueError, match="save_dir"):
            _resolve_variables(data)

    def test_transitive_chain(self):
        """A → B → C transitive resolution across multiple passes."""
        data = {
            "base": "root",
            "mid": "${base}/middle",
            "leaf": "${mid}/end",
        }
        result = _resolve_variables(data)
        assert result["base"] == "root"
        assert result["mid"] == "root/middle"
        assert result["leaf"] == "root/middle/end"

    def test_self_ref_priority_over_env_defaults(self):
        """Config data dict takes priority over .env defaults."""
        data = {"dataset": "imagenette", "save_dir": "ckpts/${dataset}"}
        defaults = {"dataset": "pet_from_env"}
        result = _resolve_variables(data, defaults)
        assert result["save_dir"] == "ckpts/imagenette"

    def test_falls_back_to_env_defaults_when_not_in_data(self):
        """If not in config dict, falls back to .env defaults."""
        data = {"save_dir": "ckpts/${dataset}"}
        defaults = {"dataset": "pet"}
        result = _resolve_variables(data, defaults)
        assert result["save_dir"] == "ckpts/pet"

    def test_mixed_self_ref_and_env(self):
        """Mix of self-referencing and .env variables."""
        data = {
            "dataset": "imagenette",
            "save_dir": "$checkpoint_dir/classifier_${dataset}",
        }
        defaults = {"checkpoint_dir": "checkpoints"}
        result = _resolve_variables(data, defaults)
        assert result["save_dir"] == "checkpoints/classifier_imagenette"

    def test_cycle_detection_raises(self):
        """Circular references raise ValueError."""
        data = {"a": "${b}", "b": "${a}"}
        with pytest.raises(ValueError):
            _resolve_variables(data)

    def test_non_string_self_ref(self):
        """Non-string config values are cast to str when referenced."""
        data = {"num_classes": 10, "label": "classes_${num_classes}"}
        result = _resolve_variables(data)
        assert result["label"] == "classes_10"

    def test_override_before_resolution_end_to_end(self, tmp_path):
        """--set dataset=pet modifies config BEFORE ${dataset} is resolved."""
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(
            "dataset: imagenette\n"
            "save_dir: checkpoints/classifier_${dataset}\n"
        )
        cfg = load_config(
            ClassifierConfig, cfg_file, overrides=["dataset=pet"]
        )
        assert cfg.dataset == "pet"
        assert cfg.save_dir == "checkpoints/classifier_pet"

    def test_override_before_resolution_with_env(self, tmp_path):
        """--set + .env + self-ref all work together."""
        env_file = tmp_path / ".env"
        env_file.write_text("checkpoint_dir=ckpts\n")
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(
            "dataset: imagenette\n"
            "save_dir: $checkpoint_dir/classifier_${dataset}\n"
        )
        cfg = load_config(
            ClassifierConfig,
            cfg_file,
            overrides=["dataset=pet"],
            env_path=env_file,
        )
        assert cfg.dataset == "pet"
        assert cfg.save_dir == "ckpts/classifier_pet"

    def test_var_re_pattern_exported(self):
        """_VAR_RE is the compiled pattern used for variable matching."""
        assert _VAR_RE.search("${foo}") is not None
        assert _VAR_RE.search("$foo") is not None
        assert _VAR_RE.search("no_vars") is None
