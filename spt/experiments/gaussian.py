"""Frozen inference for clean spectra and six clipped Gaussian-noise amplitudes."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (fresh_output, load_config, load_dataset, load_model,
                     metric_values, predict, raw_to_cont, registry_job, save_json, set_seed, verify_reference)
from spt.utils.execution import inference_backend

AMPLITUDES = (.05, .10, .15, .20, .25, .30)
METHODS = ("spt", "random_forest", "direct_nn", "amformer")


def original_noise_vector(limit, seed=42):
    """The historical shared 256-point raw-intensity vector (sigma=A/3)."""
    limit = float(limit)
    if not np.isfinite(limit) or limit < 0:
        raise ValueError("Noise amplitude must be finite and nonnegative")
    rng = random.Random(seed)
    return np.asarray([max(-limit, min(rng.gauss(0, limit / 3), limit))
                       for _ in range(256)], dtype=np.float32)


def add_original_noise(intensities, limit, seed=42):
    vector = original_noise_vector(limit, seed)
    values = np.asarray(intensities, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 256 or not np.isfinite(values).all():
        raise ValueError("Gaussian evaluation requires finite 256-point spectra")
    return values + vector, vector


def verify_matched_inputs(cache, key, arrays):
    """Compare the actual cohort arrays across models, including row order."""
    if key not in cache:
        cache[key] = {name: np.asarray(value).copy() for name, value in arrays.items()}
        return
    reference = cache[key]
    if set(reference) != set(arrays) or any(
        not np.array_equal(reference[name], np.asarray(arrays[name])) for name in reference
    ):
        raise ValueError(f"Model inputs, row order or labels differ for {key}")


def run_case(entry, output, *, device="cuda:0", batch_size=128, seed=42, matched_inputs=None):
    args = load_config(entry["config"])
    with inference_backend(args, device) as backend:
        return _run_case(entry, output, device=device, batch_size=batch_size,
                         seed=seed, matched_inputs=matched_inputs, backend=backend)


def _run_case(entry, output, *, device, batch_size, seed, matched_inputs, backend):
    import torch
    from spt.utils.data_load.data_des import parse_column_to_numpy
    method = entry["method"]
    if method not in METHODS:
        raise ValueError(f"Unsupported Gaussian comparator: {method}")
    args = load_config(entry["config"])
    if entry.get("root"):
        for key in ("data_file", "split_file"):
            value = Path(getattr(args, key))
            setattr(args, key, str((Path(entry["root"]) / value).resolve()))
    if entry.get("task") and entry["task"] != args.task:
        raise ValueError("Manifest task differs from model config")
    if method != "random_forest":
        expected = "spt" if method == "spt" else method
        if getattr(args, "architecture", "spt") != expected:
            raise ValueError("Manifest method differs from configuration architecture")
    args.batch_size = batch_size
    set_seed(seed)
    dataset = load_dataset(args)
    if isinstance(getattr(args, "multi_task", None), list) and args.multi_task:
        raise ValueError("Gaussian task comparison requires one descriptor per model")
    raw = pd.read_csv(args.data_file, comment="#",
                      usecols=["XAS_Relative_energy", "XAS_Intensity"]).iloc[dataset.indices]
    energy = parse_column_to_numpy(raw.XAS_Relative_energy)
    intensity = parse_column_to_numpy(raw.XAS_Intensity)
    clean = np.asarray(dataset.cont, dtype=np.float32).copy()
    if matched_inputs is not None:
        verify_matched_inputs(matched_inputs, (entry.get("site", args.data_name), args.task), {
            "rows": np.asarray(dataset.indices, dtype=np.int64),
            "energy": np.asarray(energy, dtype=np.float32),
            "intensity": np.asarray(intensity, dtype=np.float32),
            "labels": np.asarray(dataset.label, dtype=np.float64),
            "continuous": clean,
        })
    if not np.array_equal(raw_to_cont(energy, intensity, args), clean):
        raise ValueError("Reconstructed inputs differ from training preprocessing")
    weights = Path(entry["checkpoint"])
    from spt.registry import verify_asset
    verify_asset(weights, entry.get("checkpoint_bytes"))
    if method == "random_forest":
        import joblib
        model = joblib.load(weights)
        model.set_params(n_jobs=1)
    else:
        model = load_model(args, weights, torch.device(device))
    output.mkdir(parents=True, exist_ok=True)
    records, clean_score, verification = [], None, None
    mode = getattr(args, args.task)["mode"]
    primary = "F1" if mode == "Cls" else "R2"
    for amplitude in (0.,) + AMPLITUDES:
        values, _ = add_original_noise(intensity, amplitude, seed)
        dataset.cont = raw_to_cont(energy, values, args)
        if not np.array_equal(dataset.cont[:, 0::2], clean[:, 0::2]):
            raise ValueError("The energy input changed during noise injection")
        if method == "random_forest":
            features = np.concatenate([dataset.cate.astype(np.float32), dataset.cont.astype(np.float32)], axis=1)
            frame = pd.DataFrame({"row_index": dataset.indices, "label": np.asarray(dataset.label),
                                  "prediction": model.predict(features)})
            scores = metric_values(mode, frame.label, frame.prediction)
        else:
            scores, frame = predict(model, dataset, torch.device(device), batch_size)
        if amplitude == 0:
            verification = verify_reference(frame, entry["reference_predictions"])
            clean_score = scores[primary]
        condition = "clean" if amplitude == 0 else f"original_noise_A{amplitude:.2f}"
        frame.to_csv(output / f"{condition}_predictions.csv", index=False)
        records.append({"method": method, "site": entry.get("site", args.data_name), "descriptor": args.task,
                        "task_type": "classification" if mode == "Cls" else "regression",
                        "condition": condition, "noise_amplitude": amplitude, "gaussian_sigma": amplitude / 3,
                        "n_spectra": len(frame), "primary_metric": primary, "primary_score": scores[primary],
                        "clean_primary_score": clean_score, "score_loss": clean_score - scores[primary], **scores})
    save_json(output / "metadata.json", {"method": method, "task": args.task, "batch_size": batch_size,
              "seed": seed, "device": device, "checkpoint": str(weights),
              "config": str(entry["config"]), "data": str(args.data_file),
              "split_file": str(args.split_file), "clean_verification": verification,
              "inference_backend": backend,
              "row_order": np.asarray(dataset.indices).astype(int).tolist(),
              "noise_rule": "one shared raw float32 vector per amplitude, clipped Gaussian sigma=A/3; then training normalization",
              "metrics": "float64 sklearn metrics from row-level predictions"})
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="JSON with jobs: id, method, config, checkpoint, reference_predictions")
    parser.add_argument("--model-id", nargs="+", help="One or more IDs in the public registry")
    parser.add_argument("--root", type=Path, help="Checkout root for registry assets")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference-predictions", type=Path)
    parser.add_argument("--method", choices=METHODS, default="spt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    cli = parser.parse_args(argv)
    if cli.batch_size < 1:
        parser.error("batch size must be positive")
    if cli.model_id:
        if any((cli.manifest, cli.config, cli.checkpoint, cli.reference_predictions)):
            parser.error("Use registry IDs, a manifest, or explicit paths, not a mixture")
        jobs = [registry_job(name, cli.root) for name in cli.model_id]
    elif cli.manifest:
        if any((cli.config, cli.checkpoint, cli.reference_predictions)):
            parser.error("Use a manifest or single-model paths, not both")
        jobs = json.loads(cli.manifest.read_text(encoding="utf-8"))["jobs"]
    else:
        if not all((cli.config, cli.checkpoint, cli.reference_predictions)):
            parser.error("config, checkpoint and reference-predictions are all required")
        jobs = [{"id": cli.method, "method": cli.method, "config": str(cli.config),
                 "checkpoint": str(cli.checkpoint), "reference_predictions": str(cli.reference_predictions)}]
    if not jobs or len({item["id"] for item in jobs}) != len(jobs):
        raise ValueError("Nonempty, unique job IDs are required")
    for job in jobs:
        if Path(job["id"]).name != job["id"] or job["id"] in (".", ".."):
            raise ValueError("Job IDs must be simple directory names")
        for key in ("config", "checkpoint", "reference_predictions"):
            if not Path(job[key]).is_file():
                raise FileNotFoundError(job[key])
    output = fresh_output(cli.output_dir)
    rows = []
    matched = {}
    for job in jobs:
        result = run_case(job, output / job["id"], device=cli.device,
                          batch_size=cli.batch_size, seed=cli.seed, matched_inputs=matched)
        rows.extend(result)
    pd.DataFrame(rows).to_csv(output / "gaussian_results.csv", index=False)
    save_json(output / "protocol.json", {"seed": cli.seed, "batch_size": cli.batch_size,
              "amplitudes": [0.] + list(AMPLITUDES), "n_jobs": len(jobs),
              "n_task_conditions": len(rows), "is_paper_seed": cli.seed == 42})
