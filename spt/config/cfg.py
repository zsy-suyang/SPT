"""Portable YAML loading without changing scientific configuration values.

Relative base configurations are resolved beside the declaring YAML first and
then from the project root. Runtime paths are relative to the project root.
"""

import argparse
from datetime import datetime
from pathlib import Path
import time

import yaml


PATH_FIELDS = {
    "data_file", "split_file", "ckpt", "save_folder", "save_log",
    "metrics_output", "prediction_output_dir", "dat_output_dir",
}


def is_none_like(value):
    return value is None or (
        isinstance(value, str) and value.strip().lower() in {"none", "null", ""}
    )


def _project_root(config_path, explicit_root=None):
    if explicit_root is not None:
        return Path(explicit_root).expanduser().resolve()
    for candidate in (config_path.parent, *config_path.parent.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    return config_path.parent


def _read_mapping(path):
    with path.open(encoding="utf-8-sig") as handle:
        mapping = yaml.safe_load(handle)
    if not isinstance(mapping, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    return mapping


def _resolve_base(value, declaring_path, project_root):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Each __base__ entry must be a file path: {declaring_path}")
    candidate = Path(value).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [
        declaring_path.parent / candidate, project_root / candidate,
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"Base configuration {value!r} declared by {declaring_path} was not found. "
        "Use portable relative paths or provide --project-root."
    )


def _load_mapping(path, project_root, ancestors=()):
    path = path.resolve()
    if path in ancestors:
        chain = " -> ".join(str(item) for item in (*ancestors, path))
        raise ValueError(f"Cyclic __base__ configuration inheritance: {chain}")
    current = _read_mapping(path)
    bases = current.pop("__base__", [])
    if bases is None:
        bases = []
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list):
        raise ValueError(f"__base__ must be a path or a list of paths: {path}")
    merged = {}
    for base in bases:
        base_path = _resolve_base(base, path, project_root)
        merged.update(_load_mapping(base_path, project_root, (*ancestors, path)))
    # Preserve the original shallow override semantics, including task mappings.
    merged.update(current)
    return merged


def load_config(config_path, project_root=None, overrides=None):
    """Return a Namespace without creating outputs or loading model weights.

    Overrides are applied after YAML values. A Python None is a real override
    and may be used to disable checkpoint loading for a fresh training run.
    """
    path = Path(config_path).expanduser().resolve()
    root = _project_root(path, project_root)
    values = _load_mapping(path, root)
    if overrides:
        values.update(overrides)
    for key, value in list(values.items()):
        if is_none_like(value):
            values[key] = None
    for key in PATH_FIELDS:
        value = values.get(key)
        if value is not None:
            runtime_path = Path(value).expanduser()
            if not runtime_path.is_absolute():
                runtime_path = root / runtime_path
            values[key] = str(runtime_path.resolve())
    values["config"] = str(path)
    values["project_root"] = str(root)
    values.setdefault("time_delay", 0)
    values.setdefault("num_workers", 0)
    values.setdefault("device", "auto")
    values.setdefault("gpu_id", 0)
    values.setdefault("dataset_mode", "test")
    values.setdefault("test_set", "test")
    exp_param = values.get("exp_param")
    if exp_param is not None:
        parameters = [exp_param] if isinstance(exp_param, str) else exp_param
        if not isinstance(parameters, (list, tuple)):
            raise ValueError("exp_param must be a parameter name or a list of names")
        for key in parameters:
            if key not in values:
                raise ValueError(f"exp_param refers to missing configuration key: {key}")
            values["exp_name"] = f"{values.get('exp_name', 'run')}-{key}={values[key]}"
    return argparse.Namespace(**values)


def argument_parser(description=None, evaluation=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Model YAML configuration")
    parser.add_argument("--project-root", help="Root for relative data/output paths")
    parser.add_argument("--data-file", help="Override the input CSV")
    parser.add_argument("--split-file", help="Override the frozen split JSON")
    parser.add_argument("--checkpoint", help="Checkpoint override; 'none' starts fresh training")
    parser.add_argument("--output-dir", help="Override run/evaluation output directory")
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--num-workers", type=int, help="DataLoader workers (0 is portable)")
    if evaluation:
        parser.add_argument("--split", choices=("val", "dev", "test"), default="test")
    return parser


def parse_config(argv=None, *, evaluation=False, description=None):
    parsed = argument_parser(description, evaluation=evaluation).parse_args(argv)
    overrides = {}
    for source, target in (
        ("data_file", "data_file"), ("split_file", "split_file"),
        ("checkpoint", "ckpt"), ("device", "device"), ("num_workers", "num_workers"),
    ):
        value = getattr(parsed, source)
        if value is not None:
            overrides[target] = value
    if parsed.num_workers is not None and parsed.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if evaluation:
        overrides.update(dataset_mode=parsed.split, test_set=parsed.split)
    if parsed.output_dir is not None:
        output = Path(parsed.output_dir)
        if evaluation:
            overrides.update(
                metrics_output=str(output / "metrics.json"),
                prediction_output_dir=str(output / "predictions"),
                dat_output_dir=str(output / "arrays"),
            )
        else:
            overrides.update(save_folder=str(output / "checkpoints"), save_log=str(output / "logs"))
    args = load_config(parsed.config, parsed.project_root, overrides)
    if evaluation and is_none_like(getattr(args, "ckpt", None)):
        raise ValueError("Evaluation requires --checkpoint or a ckpt path in the YAML")
    return args


def arg2str(args):
    lines = ["run_time: " + datetime.now().strftime("%b%d_%H-%M-%S")]
    lines.extend(f"{key}: {value}" for key, value in sorted(vars(args).items()))
    return "\n".join(lines) + "\n"


class BaseConfig:
    """Compatibility adapter for the original configuration entry point."""

    def __init__(self, config=None):
        self.config = config

    def initialize(self, config=None, argv=None):
        selected = config or self.config
        args = load_config(selected) if selected else parse_config(argv)
        delay = float(getattr(args, "time_delay", 0) or 0)
        if delay < 0:
            raise ValueError("time_delay must be non-negative")
        if delay:
            time.sleep(delay)
        return args
