"""YAML config loader with --set override and $variable resolution support."""

import os
import re
from pathlib import Path
from typing import Type, TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _parse_value(s: str):
    """Try int -> float -> bool -> str."""
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    return s


def _load_env(path: str | Path) -> dict[str, str]:
    """Parse a ``.env`` file (``KEY=VALUE`` lines) into a dict.

    Blank lines and lines starting with ``#`` are skipped.
    Values may optionally be wrapped in single or double quotes.
    """
    env: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            if not _:
                continue  # skip lines without '='
            key = key.strip()
            value = value.strip()
            # Strip matching surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            env[key] = value
    return env


_VAR_RE = re.compile(r"\$\{(\w+)\}|\$(\w+)")


def _resolve_variables(data: dict, defaults: dict | None = None) -> dict:
    """Replace ``$var`` / ``${var}`` placeholders in string values.

    Resolution order for each variable name:

    1. The *data* dict itself (self-referencing: ``${dataset}`` resolves from
       the ``dataset`` key in the same config).
    2. *defaults* dict (from a ``.env`` file).
    3. Environment variables (``os.environ``).

    Multi-pass resolution (up to 10 passes) handles transitive chains where
    one variable's value depends on another variable that is also being
    resolved (e.g. ``A → B → C``).  A variable does not resolve from its
    own key (``save_dir: ${save_dir}`` would look elsewhere).

    Raises :class:`ValueError` if any variables remain unresolved after all
    passes, including cycle detection.

    Parameters
    ----------
    data:
        Config dictionary whose string values may contain ``$var`` references.
    defaults:
        Optional mapping of variable names to replacement values.

    Returns
    -------
    dict
        A new dictionary with variables resolved.

    Raises
    ------
    ValueError
        If any ``$var`` references could not be resolved.
    """
    ext_lookup = defaults or {}
    resolved = dict(data)
    max_passes = 10

    for _ in range(max_passes):
        still_has_vars = False

        for key, value in resolved.items():
            if not isinstance(value, str):
                continue
            if not _VAR_RE.search(value):
                continue

            def _replace(match: re.Match, _key: str = key) -> str:
                var_name = match.group(1) or match.group(2)
                # Skip self-referencing same key
                if var_name == _key:
                    return match.group(0)
                # 1. Config data dict itself (highest priority)
                if var_name in resolved:
                    candidate = resolved[var_name]
                    if isinstance(candidate, str) and _VAR_RE.search(candidate):
                        # Still unresolved — leave for next pass
                        return match.group(0)
                    return str(candidate)
                # 2. .env defaults
                if var_name in ext_lookup:
                    return str(ext_lookup[var_name])
                # 3. Environment variables
                env_val = os.environ.get(var_name)
                if env_val is not None:
                    return env_val
                return match.group(0)

            resolved[key] = _VAR_RE.sub(_replace, value)

            if _VAR_RE.search(resolved[key]):
                still_has_vars = True

        if not still_has_vars:
            break

    # Check for unresolved variables
    unresolved: set[str] = set()
    for key, value in resolved.items():
        if isinstance(value, str):
            for m in _VAR_RE.finditer(value):
                unresolved.add(m.group(1) or m.group(2))

    if unresolved:
        names = ", ".join(sorted(unresolved))
        raise ValueError(
            f"Unresolved config variables: {names}. "
            f"Either pass --env <.env file> or set them as environment variables."
        )

    return resolved


def load_config(
    config_cls: Type[T],
    config_path: str | Path,
    overrides: list[str] | None = None,
    env_path: str | Path | None = None,
) -> T:
    """Load a Pydantic config from a YAML file with optional KEY=VALUE overrides.

    Parameters
    ----------
    config_cls:
        Pydantic model class to instantiate.
    config_path:
        Path to a YAML config file.
    overrides:
        List of ``KEY=VALUE`` strings that override YAML values.
        Values are auto-cast: int, float, bool (true/false), then str.
        Overrides are applied **before** variable resolution so that e.g.
        ``--set dataset=pet`` participates in resolving ``${dataset}``
        in other values.
    env_path:
        Optional path to a ``.env`` file (``KEY=VALUE`` lines).  Variables
        (``$var`` / ``${var}``) in config values are resolved using:
        config self-references first, then the env file, then real
        environment variables.

    Returns
    -------
    T
        Validated config instance.
    """
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    # Overrides apply BEFORE variable resolution so that e.g.
    # --set dataset=pet participates in resolving ${dataset} in other values.
    for kv in overrides or []:
        key, val = kv.split("=", 1)
        data[key] = _parse_value(val)

    # Resolve $variable placeholders (config self-ref > .env file > env vars)
    defaults = None
    if env_path is not None and Path(env_path).is_file():
        defaults = _load_env(env_path)
    data = _resolve_variables(data, defaults)

    return config_cls.model_validate(data)
