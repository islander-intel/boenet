# utils/config_utils.py
"""
config_utils.py

Shared helpers to:
  • parse booleans robustly (str2bool),
  • load a YAML config file (load_yaml_config),
  • deep-merge YAML with CLI args while respecting parser defaults
    (merge_yaml_cli / merge_config_dicts),
  • update an argparse.Namespace in-place with the merged config
    (apply_merged_config_to_namespace).

Design goals
------------
- If --config is provided, we load YAML first, then override with any CLI flags
  that the user *explicitly* set (i.e., differ from parser defaults).
- Unknown YAML keys are kept in the merged dict (harmless to trainers that
  ignore them) for reproducibility.
- Nested dicts are merged recursively (deep merge).
- No hard dependency on PyYAML when not used; we import lazily.

Typical usage (in your trainer)
-------------------------------
>>> import argparse
>>> from utils.config_utils import (
...     load_yaml_config, detect_cli_overrides, merge_config_dicts,
...     apply_merged_config_to_namespace
... )

>>> parser = build_arg_parser()        # your existing parser
>>> parser.add_argument("--config", type=str, default=None,
...     help="Path to a YAML config. CLI flags override values in this file.")

>>> args = parser.parse_args()
>>> yaml_cfg = load_yaml_config(args.config)  # {} if None or not found
>>> # parser defaults used to decide which CLI flags were explicitly set:
>>> defaults = vars(parser.parse_args([]))
>>> cli_overrides = detect_cli_overrides(cli_dict=vars(args), defaults=defaults)

>>> merged = merge_config_dicts(base=yaml_cfg, overlay=cli_overrides)
>>> # Keep args in sync with merged values (so downstream code can keep using args.*)
>>> apply_merged_config_to_namespace(args, merged)

>>> # From here on:
>>> # - read trainer knobs from `args` (now reflecting config+CLI),
>>> # - or, if you prefer, use the `merged` dict directly.

Notes
-----
- If your parser uses `action="store_true"` style flags, a bare "False"
  on the command line (e.g., `--use_pruning False`) will be rejected by argparse.
  For value-carrying booleans, define flags like:
      parser.add_argument("--use_pruning", type=str2bool, default=False)
  and remove the store_true/store_false pattern.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Mapping, MutableMapping
import argparse
import os


# --------------------------------------------------------------------------- #
#                              Boolean parsing                                #
# --------------------------------------------------------------------------- #

def str2bool(v: Any) -> bool:
    """
    Robust boolean parser:
      - Pass-through if already a bool.
      - Accepts: "true","t","yes","y","1","on"  -> True
                 "false","f","no","n","0","off" -> False
      - Case-insensitive; strips whitespace.
      - Integers: 0 -> False; nonzero -> True.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, (int,)):
        return v != 0
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in {"true", "t", "yes", "y", "1", "on"}:
        return True
    if s in {"false", "f", "no", "n", "0", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from: {v!r}")


# --------------------------------------------------------------------------- #
#                              YAML utilities                                 #
# --------------------------------------------------------------------------- #

def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    """
    Load a YAML config file. If `path` is None or the file does not exist,
    returns an empty dict. Raises on YAML syntax errors.

    Lazy-imports PyYAML so this module does not hard-depend on it.

    Args
    ----
    path : str | None

    Returns
    -------
    cfg : dict
    """
    if path is None:
        return {}
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        # Silently return {} to allow CLI-only runs.
        return {}
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required to load config files. Install with `pip install pyyaml`."
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)  # can be None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML must be a mapping (dict). Got: {type(data).__name__}")
    return data


# --------------------------------------------------------------------------- #
#                        Deep-merge & override detection                       #
# --------------------------------------------------------------------------- #

def deep_merge(base: MutableMapping[str, Any], upd: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """
    Recursively merge mapping `upd` into mapping `base` in-place and return `base`.

    Rules
    -----
    - If both sides have a dict at a key: recurse.
    - Otherwise: the value from `upd` overwrites `base`.
    - Lists/tuples are replaced (not concatenated).

    Example
    -------
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    upd  = {"b": {"y": 99, "z": 3}, "c": 7}
    -> {"a": 1, "b": {"x": 1, "y": 99, "z": 3}, "c": 7}
    """
    for k, v in upd.items():
        if (
            k in base
            and isinstance(base[k], Mapping)
            and isinstance(v, Mapping)
        ):
            deep_merge(base[k], v)  # type: ignore[arg-type]
        else:
            base[k] = v
    return base


def detect_cli_overrides(
    cli_dict: Mapping[str, Any],
    defaults: Optional[Mapping[str, Any]] = None,
    *,
    skip_none: bool = True,
) -> Dict[str, Any]:
    """
    Return only those CLI key/value pairs that should override config.

    Behavior
    --------
    - If `defaults` is provided, a CLI key is considered "set by user" when
      `cli_value != defaults[key]`. Otherwise, any non-None (or any value if
      skip_none=False) is considered set.
    - The returned dict is *flat* (no nesting). Trainers typically use a flat
      arg space; unknown keys are harmless.

    Args
    ----
    cli_dict : mapping of parsed args, usually vars(args)
    defaults : mapping of parser defaults, e.g., vars(parser.parse_args([]))
    skip_none : drop keys whose CLI value is None

    Returns
    -------
    overrides : dict
    """
    overrides: Dict[str, Any] = {}
    for k, v in cli_dict.items():
        if k == "config":
            # Never propagate the path itself as a trainer option
            continue
        if defaults is not None:
            if k not in defaults:
                # Unknown to the parser; treat as override if not None
                if (v is not None) or (not skip_none):
                    overrides[k] = v
                continue
            # Only override when the user supplied a value different from default
            if v != defaults[k]:
                if (v is not None) or (not skip_none):
                    overrides[k] = v
        else:
            if (v is not None) or (not skip_none):
                overrides[k] = v
    return overrides


def merge_config_dicts(
    base: Optional[Mapping[str, Any]],
    overlay: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Deep-merge two (possibly None) dict-like objects into a new dict.
    `overlay` takes precedence.

    Returns a *new* dict (does not modify inputs).
    """
    out: Dict[str, Any] = {}
    if base:
        deep_merge(out, dict(base))
    if overlay:
        deep_merge(out, dict(overlay))
    return out


def merge_yaml_cli(
    yaml_cfg: Optional[Mapping[str, Any]],
    cli_args: argparse.Namespace | Mapping[str, Any],
    parser: Optional[argparse.ArgumentParser] = None,
) -> Dict[str, Any]:
    """
    One-shot convenience: given yaml config, CLI args, and the parser, return
    a single merged dict where CLI flags that differ from parser defaults
    override YAML.

    Example
    -------
    >>> yaml_cfg = load_yaml_config(args.config)
    >>> merged = merge_yaml_cli(yaml_cfg, args, parser)

    Returns
    -------
    merged : dict
    """
    cli_dict = dict(cli_args) if isinstance(cli_args, Mapping) else vars(cli_args)
    defaults = vars(parser.parse_args([])) if parser is not None else None
    cli_overrides = detect_cli_overrides(cli_dict=cli_dict, defaults=defaults)
    return merge_config_dicts(base=yaml_cfg, overlay=cli_overrides)


# --------------------------------------------------------------------------- #
#                   Namespace ↔ dict synchronization helpers                   #
# --------------------------------------------------------------------------- #

def apply_merged_config_to_namespace(ns: argparse.Namespace, cfg: Mapping[str, Any]) -> None:
    """
    Update an argparse.Namespace `ns` in-place from `cfg`.

    - If a key exists in `cfg`, we set `ns.<key>` to that value.
    - Keys present in `cfg` but not originally defined in the parser will
      still be injected into the namespace—this is intentional so downstream
      code can access config-only fields if desired.
    """
    for k, v in cfg.items():
        setattr(ns, k, v)


def namespace_to_dict(ns: argparse.Namespace) -> Dict[str, Any]:
    """Shallow convert argparse.Namespace → dict."""
    return dict(vars(ns))


# --------------------------------------------------------------------------- #
#                         Optional: simple round-trip                          #
# --------------------------------------------------------------------------- #

def dump_yaml_config(cfg: Mapping[str, Any], path: str) -> None:
    """
    Save a config dictionary to YAML. Requires PyYAML.

    Useful for writing out the *effective* (merged) config alongside checkpoints:
      dump_yaml_config(merged, "checkpoints/run_cfg.yaml")
    """
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyYAML is required to dump YAML. Install with `pip install pyyaml`.") from e

    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False)
