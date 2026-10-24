"""Shared frozen-model input, prediction and provenance functions."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


def project_path(value):
    """Resolve explicit CLI/config paths from the working directory."""
    return Path(value).expanduser().resolve()


def load_config(path):
    """Load an expanded public YAML without running a training CLI."""
    from spt.config.cfg import load_config as read_config
    return read_config(path)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def set_seed(seed=42):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_spectrum(energy, intensity):
    energy, intensity = np.asarray(energy, float), np.asarray(intensity, float)
    if (energy.ndim != 1 or intensity.shape != energy.shape or len(energy) < 2
            or not np.isfinite(energy).all() or not np.isfinite(intensity).all()
            or np.any(np.diff(energy) <= 0)):
        raise ValueError("Expected finite paired samples with strictly increasing energies")
    return energy, intensity


def raw_to_cont(energies, intensities, args):
    """Use the original float32 training-time normalization and interleaving."""
    from spt.utils.data_load.data_des import X_log_normalize_optimized, Y_log_normalize_optimized
    energies = [np.asarray(x, dtype=np.float32) for x in energies]
    intensities = [np.asarray(x, dtype=np.float32) for x in intensities]
    x = X_log_normalize_optimized(energies, args.min_log_data, args.max_log_data, log_base=np.e)
    y = Y_log_normalize_optimized(intensities)
    rows = []
    for a, b in zip(x, y):
        if len(a) != len(b):
            raise ValueError("Energy and intensity length mismatch")
        row = np.empty(len(a) * 2, dtype=np.float32)
        row[0::2], row[1::2] = a, b
        rows.append(row)
    return np.stack(rows)


def load_dataset(args, split="test"):
    from spt.utils import data_load
    if not Path(args.split_file).is_file():
        raise FileNotFoundError(f"Frozen split is required: {args.split_file}")
    if not Path(args.data_file).is_file():
        raise FileNotFoundError(args.data_file)
    if str(getattr(args, "add_noise", None)).lower() != "none":
        raise ValueError("Frozen input configuration must have add_noise=None")
    args.dataset_mode, args.num_workers = split, 0
    args.allow_lockbox_access = False
    return getattr(data_load, args.data_name.lower())(dataset=split, args=args)


def load_model(args, checkpoint, device):
    import torch
    from spt.models import build_model
    args.ckpt = str(checkpoint)
    model = build_model(args, device).to(device)
    # Checkpoints and joblib estimators must come from a trusted source.
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["net_state_dict"], strict=True)
    model.eval()
    return model


def metric_values(mode, labels, predictions):
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if mode == "Cls":
        return {"Acc": float(accuracy_score(labels, predictions)),
                "F1": float(f1_score(labels, predictions, average="weighted"))}
    mse = mean_squared_error(labels, predictions)
    return {"R2": float(r2_score(labels, predictions)), "MSE": float(mse),
            "RMSE": float(np.sqrt(mse)), "MAE": float(mean_absolute_error(labels, predictions))}


def predict(model, dataset, device, batch_size):
    """Deterministic full single-task inference in saved loader row order."""
    import torch
    if batch_size < 1 or not len(dataset):
        raise ValueError("Batch size and evaluation row count must be positive")
    if isinstance(getattr(dataset.args, "multi_task", None), list) and dataset.args.multi_task:
        raise ValueError("This experiment requires a single-task model")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    values, labels = [], []
    with torch.inference_mode():
        for cate, cont, label in loader:
            output, _ = model(cate.to(device), cont.to(device), label.to(device))
            values.append(output.detach().cpu())
            labels.append(label.detach().cpu())
    values, labels = torch.cat(values), torch.cat(labels)
    if not torch.isfinite(values).all():
        raise FloatingPointError("Non-finite model output")
    mode = getattr(dataset.args, dataset.args.task)["mode"]
    frame = pd.DataFrame({"row_index": dataset.indices, "label": labels.numpy().reshape(-1)})
    if mode == "Cls":
        probability = torch.softmax(values, dim=1).numpy()
        frame["prediction"] = probability.argmax(axis=1)
        frame["confidence"] = probability.max(axis=1)
        for k in range(probability.shape[1]):
            frame[f"probability_{k}"] = probability[:, k]
    else:
        frame["prediction"] = values.numpy().reshape(-1)
    metrics = metric_values(mode, frame.label, frame.prediction)
    if mode == "Cls":
        from sklearn.metrics import f1_score
        metrics["macro_F1"] = float(f1_score(frame.label, frame.prediction,
            labels=np.arange(probability.shape[1]), average="macro", zero_division=0))
    return metrics, frame


def verify_reference(frame, path, partial=False):
    """Require identical row identity/labels and numerically matching clean predictions."""
    reference = path.copy() if isinstance(path, pd.DataFrame) else pd.read_csv(path)
    required = {"row_index", "label", "prediction"}
    if not required.issubset(reference) or not required.issubset(frame):
        raise ValueError("Predictions require row_index, label, prediction")
    if reference.row_index.duplicated().any() or frame.row_index.duplicated().any():
        raise ValueError("Duplicate prediction row identity")
    if not partial and set(reference.row_index) != set(frame.row_index):
        raise ValueError("Clean test row indices differ from reference")
    if not set(frame.row_index).issubset(set(reference.row_index)):
        raise ValueError("Evaluated rows absent from reference")
    reference = reference.set_index("row_index").loc[frame.row_index]
    if not np.allclose(reference.label, frame.label, rtol=1e-6, atol=1e-6):
        raise ValueError("Clean labels differ from reference")
    columns = [c for c in frame if c == "prediction" or c.startswith("probability_")]
    for column in columns:
        if column not in reference:
            raise ValueError(f"Reference is missing {column}")
        if not np.allclose(reference[column], frame[column], rtol=1e-5, atol=5e-6):
            raise ValueError(f"Clean {column} differs from reference")
    return {"passed": True, "n_rows": len(frame),
            "absolute_tolerance": 5e-6, "relative_tolerance": 1e-5}


def fresh_output(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def registry_job(model_id, root=None):
    """Resolve a public model registry record, without changing its configuration."""
    from spt.registry import get_model
    directory, entry = get_model(model_id, root)
    job = dict(entry)
    architecture = entry["architecture"]
    job["method"] = "spt" if architecture == "spt" else architecture
    for key in ("config", "checkpoint", "reference_predictions"):
        if job.get(key):
            job[key] = str((Path(directory) / job[key]).resolve())
    job["root"] = str(directory)
    return job


def add_model_arguments(parser, *, reference=True):
    """Expose either a registered model or explicit, user-provided asset paths."""
    parser.add_argument("--model-id", help="Frozen model ID in configs/registry.json")
    parser.add_argument("--root", type=Path, help="Checkout root for registry assets")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", "--ckpt", dest="checkpoint", type=Path)
    if reference:
        parser.add_argument("--reference-predictions", type=Path)


def resolve_model_arguments(cli, parser, *, reference=True):
    keys = ["config", "checkpoint"] + (["reference_predictions"] if reference else [])
    if cli.model_id:
        if any(getattr(cli, key, None) is not None for key in keys):
            parser.error("Use --model-id or explicit model paths, not both")
        entry = registry_job(cli.model_id, cli.root)
        for key in keys:
            if not entry.get(key):
                parser.error(f"Registry record {cli.model_id} has no {key}")
            setattr(cli, key, Path(entry[key]))
        from spt.registry import verify_asset
        verify_asset(cli.checkpoint, entry.get("checkpoint_bytes"))
    elif any(getattr(cli, key, None) is None for key in keys):
        parser.error("Specify --model-id, or all required model paths: " + ", ".join(keys))
    for key in keys:
        path = project_path(getattr(cli, key))
        if not path.is_file():
            raise FileNotFoundError(path)
        setattr(cli, key, path)
    return cli
