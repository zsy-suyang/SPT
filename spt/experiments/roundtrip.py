"""Unchanged-coefficient AFLD reconstruction on captured frozen-model features."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (fresh_output, load_config, load_dataset, load_model,
                     save_json, set_seed, add_model_arguments, resolve_model_arguments)


def transform_errors(transform, inputs):
    import torch
    with torch.no_grad():
        coefficients = transform.analyze(inputs)
        reconstructed = transform.synthesize(coefficients)
    error = reconstructed - inputs
    norm = inputs.flatten(1).norm(dim=1)
    if (norm == 0).any():
        raise ValueError("Relative reconstruction error is undefined for a zero-norm input")
    return error.abs().flatten(1).amax(1), error.flatten(1).norm(dim=1) / norm


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--element", default="Co")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    cli = resolve_model_arguments(parser.parse_args(argv), parser)
    if cli.samples < 1:
        parser.error("samples must be positive")
    import torch
    from spt.models.specnet import AdaptiveFourierLiftingSpectralOperator
    set_seed(cli.seed)
    torch.set_num_threads(1)
    args = load_config(cli.config)
    dataset = load_dataset(args)
    reference = pd.read_csv(cli.reference_predictions)
    if reference.row_index.duplicated().any():
        raise ValueError("Duplicate reference row identity")
    if set(reference.row_index) != set(dataset.indices):
        raise ValueError("Reference and frozen test partition differ")
    source = pd.read_csv(args.data_file, comment="#")
    candidates = source.iloc[reference.row_index.to_numpy(int)]
    chosen = candidates[candidates.center_atom_type.eq(cli.element)].drop_duplicates("material_id").iloc[:cli.samples]
    if len(chosen) != cli.samples:
        raise ValueError("Insufficient distinct-material test spectra for requested element")
    lookup = {int(row): index for index, row in enumerate(dataset.indices)}
    selected = [lookup[int(row)] for row in chosen.index]
    cate = torch.as_tensor(np.asarray(dataset.cate)[selected], dtype=torch.long)
    cont = torch.as_tensor(np.asarray(dataset.cont)[selected], dtype=torch.float32)
    model = load_model(args, cli.checkpoint, torch.device("cpu"))
    captured, hooks = {}, []
    modules = dict(model.named_modules())
    for name, module in modules.items():
        if isinstance(module, AdaptiveFourierLiftingSpectralOperator):
            def capture(_module, inputs, name=name):
                captured[name] = inputs[0].detach().clone()
            hooks.append(module.register_forward_pre_hook(capture))
    try:
        with torch.no_grad():
            model(cate, cont, None)
    finally:
        for hook in hooks:
            hook.remove()
    if not captured:
        raise ValueError("Selected model contains no AFLD operators")
    records = []
    for block, (name, inputs) in enumerate(captured.items(), 1):
        for dtype in (torch.float32, torch.float64):
            transform = copy.deepcopy(modules[name].transform).to(dtype=dtype)
            absolute, relative = transform_errors(transform, inputs.to(dtype=dtype))
            for row, material, a, r in zip(chosen.index, chosen.material_id, absolute, relative):
                records.append({"block": block, "module": name, "dtype": str(dtype).split(".")[-1],
                                "row_index": int(row), "material_id": str(material),
                                "max_absolute_error": float(a), "relative_frobenius_error": float(r)})
    output = fresh_output(cli.output_dir)
    table = pd.DataFrame(records)
    table.to_csv(output / "afld_roundtrip_samples.csv", index=False)
    summary = table.groupby(["block", "module", "dtype"], sort=False).agg(
        n_samples=("row_index", "count"), max_absolute_error=("max_absolute_error", "max"),
        max_relative_frobenius_error=("relative_frobenius_error", "max"),
        mean_relative_frobenius_error=("relative_frobenius_error", "mean")).reset_index()
    summary.to_csv(output / "afld_roundtrip_summary.csv", index=False)
    save_json(output / "metadata.json", {"seed": cli.seed, "device": "cpu", "samples": cli.samples,
              "checkpoint": str(cli.checkpoint), "config": str(cli.config),
              "selection": f"first distinct-material {cli.element} test rows in saved prediction order",
              "row_indices": chosen.index.astype(int).tolist(),
              "float64": "promoted float32 learned parameters and captured features, not a new trained model",
              "scope": "analyze then synthesize with unchanged coefficients, not the full learned spectral branch"})
