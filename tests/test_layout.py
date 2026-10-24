"""Repository and registry contracts, without loading research assets."""

import json
from pathlib import Path, PurePosixPath

from spt.registry import load_registry, project_root


def test_flat_package_layout():
    root = project_root()
    assert (root / "spt/__init__.py").is_file()
    assert (root / "spt/__main__.py").is_file()
    assert not (root / "src").exists()


def test_root_discovery_from_nested_directory(monkeypatch):
    root = project_root()
    monkeypatch.delenv("SPT_PROJECT_ROOT", raising=False)
    monkeypatch.chdir(root / "spt/models")
    assert project_root() == root
    assert project_root(root) == root


def test_registered_weights_match_asset_manifest():
    root, registry = load_registry()
    assets = json.loads((root / "data/asset_manifest.json").read_text(encoding="utf-8"))["assets"]
    by_path = {item["path"]: item for item in assets}
    assert len(by_path) == len(assets)
    for model in registry["models"]:
        path = PurePosixPath(model["checkpoint"])
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert path.parts[0] == "checkpoints"
        assert all(a != b for a, b in zip(path.parts, path.parts[1:]))
        asset = by_path[str(path)]
        assert asset["bytes"] == model["checkpoint_bytes"]
        assert (root / model["config"]).is_file()


def test_optional_component_license_is_retained():
    root = project_root()
    license_path = root / "spt/baselines/amformer/LICENSE"
    assert "GNU GENERAL PUBLIC LICENSE" in license_path.read_text(encoding="utf-8")
