"""Evaluate registered frozen neural and tree models on their saved test partitions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .common import (fresh_output, load_config, load_dataset, load_model,
                     metric_values, predict, registry_job, save_json, set_seed, verify_reference)
from spt.utils.execution import inference_backend


def joint_metric_values(mode, values, labels, device):
    """Use the original Joint float32 metric reductions on the selected device."""
    import torch
    from torchmetrics import Accuracy, F1Score, MeanAbsoluteError, MeanSquaredError, R2Score
    values = values.to(device=device, dtype=torch.float32)
    target = torch.as_tensor(labels, device=device).reshape(-1)
    if mode == "Cls":
        target = target.long()
        classes = values.shape[-1]
        return {
            "Acc": Accuracy(task="multiclass", num_classes=classes, top_k=1).to(device)(values, target).item(),
            "F1": F1Score(task="multiclass", num_classes=classes, average="weighted").to(device)(values, target).item(),
        }
    values, target = values.reshape(-1), target.float()
    mse = MeanSquaredError().to(device)(values, target)
    return {
        "R2": R2Score().to(device)(values, target).item(),
        "MSE": mse.item(), "RMSE": torch.sqrt(mse).item(),
        "MAE": MeanAbsoluteError().to(device)(values, target).item(),
    }


def joint_predictions(model, dataset, device, batch_size):
    import torch
    tasks = dataset.args.multi_task
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    outputs, labels = {task: [] for task in tasks}, {task: [] for task in tasks}
    with torch.inference_mode():
        for cate, cont, truth in loader:
            values, _ = model(cate.to(device), cont.to(device), None)
            for task in tasks:
                outputs[task].append(values[task].cpu())
                labels[task].append(truth[task].cpu())
    result = {}
    for task in tasks:
        valid = np.asarray(dataset.task_valid_masks[task], dtype=bool)
        values = torch.cat(outputs[task])[valid]
        truth = torch.cat(labels[task])[valid].numpy().reshape(-1)
        if not len(truth) or not torch.isfinite(values).all():
            raise ValueError(f"Empty or non-finite Joint task output: {task}")
        frame = pd.DataFrame({"row_index": np.asarray(dataset.indices)[valid], "label": truth})
        mode = getattr(dataset.args, task)["mode"]
        if mode == "Cls":
            probability = values.softmax(dim=-1).numpy()
            frame["prediction"] = probability.argmax(-1)
            frame["confidence"] = probability.max(-1)
            for k in range(probability.shape[1]):
                frame[f"probability_{k}"] = probability[:, k]
        else:
            frame["prediction"] = values.numpy().reshape(-1)
        scores = joint_metric_values(mode, values, truth, device)
        audit = metric_values(mode, np.asarray(frame.label, dtype=np.float64),
                              np.asarray(frame.prediction, dtype=np.float64))
        scores.update({f"audit_float64_{key}": value for key, value in audit.items()})
        result[task] = (scores, frame)
    return result


def tree_predictions(args, dataset, checkpoint):
    import joblib
    architecture = args.architecture
    if architecture == "xgboost":
        try:
            import xgboost  # noqa: F401, required by trusted joblib deserialization
        except ImportError as error:
            raise ImportError("XGBoost replay requires: pip install -e '.[trees]'") from error
    model = joblib.load(checkpoint)
    if "n_jobs" in model.get_params(deep=False):
        model.set_params(n_jobs=1)
    features = np.concatenate([np.asarray(dataset.cate, np.float32), np.asarray(dataset.cont, np.float32)], axis=1)
    values = np.asarray(model.predict(features))
    mode = getattr(args, args.task)["mode"]
    if architecture == "xgboost" and mode == "Cls":
        training = load_dataset(args, "train")
        classes = np.unique(np.asarray(training.label))
        if np.any(values != np.floor(values)) or np.any(values < 0) or np.any(values >= len(classes)):
            raise ValueError("XGBoost returned a class outside its frozen training encoding")
        values = classes[values.astype(int)]
    if not np.isfinite(values).all():
        raise FloatingPointError("Non-finite tree prediction")
    frame = pd.DataFrame({"row_index": dataset.indices, "label": np.asarray(dataset.label), "prediction": values})
    return metric_values(mode, frame.label, frame.prediction), frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", nargs="+", required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-missing-reference", action="store_true",
                        help="Evaluate without a missing frozen prediction reference; mark it unverified")
    cli = parser.parse_args(argv)
    if cli.batch_size < 1 or len(set(cli.model_id)) != len(cli.model_id):
        parser.error("Positive batch size and unique model IDs are required")
    jobs = [registry_job(identity, cli.root) for identity in cli.model_id]
    for job in jobs:
        for field in ("config", "checkpoint"):
            if not Path(job[field]).is_file():
                raise FileNotFoundError(job[field])
        if not Path(job.get("reference_predictions", "")).is_file() and not cli.allow_missing_reference:
            raise FileNotFoundError(f"Frozen prediction reference missing for {job['id']}")
    output = fresh_output(cli.output_dir)
    import torch
    device = torch.device(cli.device)
    summary = []
    for job in jobs:
        set_seed(cli.seed)
        from spt.registry import verify_asset
        verify_asset(job["checkpoint"], job.get("checkpoint_bytes"))
        args = load_config(job["config"])
        with inference_backend(args, device) as backend:
            dataset = load_dataset(args)
            is_joint = isinstance(getattr(args, "multi_task", None), list) and bool(args.multi_task)
            if args.architecture in ("decision_tree", "extra_trees", "random_forest", "xgboost"):
                if is_joint:
                    raise ValueError("Joint tree estimators are not part of this benchmark")
                results = {args.task: tree_predictions(args, dataset, job["checkpoint"])}
            else:
                model = load_model(args, job["checkpoint"], device)
                results = joint_predictions(model, dataset, device, cli.batch_size) if is_joint else {
                    args.task: predict(model, dataset, device, cli.batch_size)}
                del model
        destination = output / job["id"]
        destination.mkdir()
        reference_path = Path(job.get("reference_predictions", ""))
        reference = pd.read_csv(reference_path) if reference_path.is_file() else None
        metrics_path = Path(job["root"]) / job["reference_metrics"] if job.get("reference_metrics") else None
        archived_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))[job["id"]]["metrics"] if metrics_path else {}
        checks = {}
        for task, (scores, frame) in results.items():
            if reference is not None:
                if is_joint:
                    if "task" not in reference:
                        raise ValueError("A Joint reference must be a long table with a task column")
                    subset = reference.loc[reference.task.eq(task)]
                else:
                    subset = reference
                checks[task] = verify_reference(frame, subset)
            else:
                checks[task] = {"passed": False, "reason": "reference not supplied"}
            frame.to_csv(destination / f"{task}_predictions.csv", index=False)
            primary = "F1" if getattr(args, task)["mode"] == "Cls" else "R2"
            expected = (archived_metrics.get(f"{task}_{primary.lower()}") if is_joint
                        else job.get("expected_score"))
            summary.append({"model_id": job["id"], "site": job["site"], "task": task,
                            "n_spectra": len(frame), "primary_metric": primary,
                            "expected_score": expected,
                            "score_difference_from_registered": scores[primary]-expected if expected is not None else None,
                            "reference_verified": checks[task]["passed"], **scores})
        save_json(destination / "metadata.json", {"model_id": job["id"], "device": str(device),
            "seed": cli.seed, "batch_size": cli.batch_size, "checkpoint": str(job["checkpoint"]),
            "config": str(job["config"]), "data": str(args.data_file),
            "split_file": str(args.split_file), "checks": checks,
            "inference_backend": backend,
            "reference_predictions": str(reference_path) if reference is not None else None,
            "reference_metrics": str(metrics_path) if metrics_path else None,
            "metrics": ("Joint: float32 torchmetrics on the selected device; audit_float64_*: sklearn float64"
                        if is_joint else "sklearn metrics from row predictions")})
        pd.DataFrame(summary).to_csv(output / "benchmark_results.csv", index=False)
    return 0
