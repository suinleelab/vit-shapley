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


def _resolve_variables(data: dict, defaults: dict | None = None) -> dict:
    """Replace ``$var`` / ``${var}`` placeholders in string values.

    Resolution order for each variable name:

    1. *defaults* dict (from a ``.env`` file).
    2. Environment variables (``os.environ``).

    Raises :class:`ValueError` if any variables remain unresolved.

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
        If any ``$var`` references could not be resolved from *defaults* or
        the environment.
    """
    lookup = defaults or {}
    unresolved: set[str] = set()

    def _replace(match: re.Match) -> str:
        var_name = match.group(1) or match.group(2)
        if var_name in lookup:
            return str(lookup[var_name])
        env_val = os.environ.get(var_name)
        if env_val is not None:
            return env_val
        unresolved.add(var_name)
        return match.group(0)

    resolved = {}
    for key, value in data.items():
        if isinstance(value, str):
            # Match ${var_name} or $var_name (word chars only)
            resolved[key] = re.sub(r"\$\{(\w+)\}|\$(\w+)", _replace, value)
        else:
            resolved[key] = value

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
    env_path:
        Optional path to a ``.env`` file (``KEY=VALUE`` lines).  If provided,
        ``$var`` and ``${var}`` placeholders in config string values are
        resolved using the env file, with real environment variables as
        fallback.  ``--set`` overrides apply last.

    Returns
    -------
    T
        Validated config instance.
    """
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    # Resolve $variable placeholders (.env file > env vars > error)
    defaults = None
    if env_path is not None and Path(env_path).is_file():
        defaults = _load_env(env_path)
    # Always resolve: even without an env file, env vars are checked.
    data = _resolve_variables(data, defaults)

    # Overrides apply last (raw values, no variable resolution)
    for kv in overrides or []:
        key, val = kv.split("=", 1)
        data[key] = _parse_value(val)

    return config_cls.model_validate(data)
