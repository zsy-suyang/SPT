"""Small, explicit command surface; heavy libraries load only when needed."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    commands = {"train", "evaluate", "predict", "application", "experiment", "list-models", "verify-assets"}
    parser = argparse.ArgumentParser(prog="spt", description="SPT: electronic and geometric descriptors from XANES")
    parser.add_argument("command", nargs="?", choices=sorted(commands))
    parser.add_argument("--version", action="store_true")
    if not argv or argv[0] in {"-h", "--help", "--version"}:
        args = parser.parse_args(argv)
        if args.version:
            from . import __version__
            print(__version__)
        else:
            parser.print_help()
        return
    command, rest = argv[0], argv[1:]
    if command not in commands:
        parser.error(f"unknown command {command!r}")
    if command in {"list-models", "verify-assets"}:
        from .registry import load_registry, verify_asset
        sub = argparse.ArgumentParser(prog=f"spt {command}")
        sub.add_argument("--project-root")
        sub.add_argument("--family")
        sub.add_argument("--site", choices=["metal", "oxygen"])
        sub.add_argument("--model", action="append", dest="models")
        sub.add_argument("--json", action="store_true")
        if command == "verify-assets":
            sub.add_argument("--all-assets", action="store_true",
                             help="Verify every file in data/asset_manifest.json, including application data")
        args = sub.parse_args(rest)
        root, registry = load_registry(args.project_root)
        entries = [e for e in registry["models"]
                   if (not args.family or e["family"] == args.family)
                   and (not args.site or e["site"] == args.site)
                   and (not args.models or e["id"] in args.models)]
        if args.models and set(args.models) != {e['id'] for e in entries}:
            sub.error("one or more model IDs do not exist or do not match the filters")
        if command == "verify-assets":
            if args.all_assets:
                manifest = json.loads((root / "data/asset_manifest.json").read_text(encoding="utf-8"))
                assets = manifest["assets"] if isinstance(manifest, dict) else manifest
                for asset in assets:
                    path = (root / asset["path"]).resolve()
                    if root.resolve() not in path.parents:
                        raise ValueError(f"Asset path is outside the checkout: {asset['path']}")
                    verify_asset(path, asset.get("bytes"))
                    print(f"OK {asset['path']}")
                return 0
            for entry in entries:
                verify_asset(root / entry["checkpoint"], entry.get("checkpoint_bytes"))
                print(f"OK {entry['id']}")
            for site in sorted({e["site"] for e in entries}):
                for asset in registry["datasets"][site].values():
                    verify_asset(root / asset["path"], asset.get("bytes"))
            return
        if args.json:
            print(json.dumps(entries, indent=2))
        else:
            print(f"{'MODEL':42} {'FAMILY':12} {'TASK':8} METRIC")
            for entry in entries:
                print(f"{entry['id']:42} {entry['family']:12} {entry['display_task']:8} {entry.get('primary_metric', '')}")
        return
    if command == "experiment":
        from .experiments import main as run
        result = run(rest)
        return result if isinstance(result, int) else 0
    if command == "predict":
        from .predict import main as run
        result = run(rest)
        return result if isinstance(result, int) else 0
    if command == "application":
        from .applications import main as run
        result = run(rest)
        return result if isinstance(result, int) else 0
    # Preserve the independently usable module CLIs for train and evaluate.
    import runpy
    old_argv = sys.argv
    try:
        sys.argv = [f"spt {command}", *rest]
        runpy.run_module(f"spt.{command}", run_name="__main__")
    finally:
        sys.argv = old_argv
