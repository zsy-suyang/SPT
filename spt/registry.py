"""Explicit model and artifact identities for the reported experiments."""
from __future__ import annotations

import json
import os
from pathlib import Path


def project_root(value=None):
    if value is not None:
        root = Path(value).expanduser().resolve()
    elif os.environ.get("SPT_PROJECT_ROOT"):
        root = Path(os.environ["SPT_PROJECT_ROOT"]).expanduser().resolve()
    else:
        candidates = [Path.cwd(), *Path(__file__).resolve().parents]
        root = next((p for p in candidates if (p / "configs/registry.json").is_file()), None)
        if root is None:
            raise FileNotFoundError("Cannot locate configs/registry.json. Pass --project-root to the SPT checkout.")
    if not (root / "configs/registry.json").is_file():
        raise FileNotFoundError(f"No model registry in {root}")
    return root


def load_registry(root=None):
    root = project_root(root)
    registry = json.loads((root / "configs/registry.json").read_text(encoding="utf-8"))
    if registry["schema_version"] != 1:
        raise ValueError("Unsupported model registry version")
    return root, registry


def get_model(model_id, root=None):
    root, registry = load_registry(root)
    matches = [entry for entry in registry["models"] if entry["id"] == model_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous model ID: {model_id}. Use 'spt list-models'.")
    return root, matches[0]


def verify_asset(path, expected_bytes=None):
    """Check that an asset exists and, when recorded, has the expected size."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing asset: {path}. See docs/checkpoints.md and docs/datasets.md.")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(f"Asset size mismatch: {path.name}")
    return path
