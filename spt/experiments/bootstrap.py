"""Pointwise paired crystal-structure bootstrap from saved row-level predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import save_json


def metric_value(name, labels, predictions):
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
    if name == "accuracy":
        return accuracy_score(labels, predictions)
    if name == "f1":
        return f1_score(labels, predictions, average="weighted")
    if name == "mae":
        return mean_absolute_error(labels, predictions)
    if name == "r2":
        return r2_score(labels, predictions)
    raise ValueError(name)


def validate_predictions(frame, data, group_column):
    if not {"row_index", "label", "prediction"}.issubset(frame):
        raise ValueError("Predictions require row_index, label, prediction")
    if group_column not in data:
        raise ValueError(f"Missing structure column: {group_column}")
    values = frame[["row_index", "label", "prediction"]].to_numpy(float)
    rows = values[:, 0]
    if (not len(frame) or not np.isfinite(values).all() or frame.row_index.duplicated().any()
            or np.any(rows != np.floor(rows)) or np.any(rows < 0) or np.any(rows >= len(data))):
        raise ValueError("Invalid/duplicate positional row identity or non-finite predictions")
    groups = data.iloc[rows.astype(int)][group_column]
    if groups.isna().any():
        raise ValueError("Missing parent identity")
    return groups.astype(str).to_numpy()


def cluster_bootstrap(predictions, data, *, reference=None, group_column="material_id",
                      metric="f1", iterations=2000, seed=42):
    if iterations < 1:
        raise ValueError("iterations must be positive")
    predictions = predictions.copy()
    if reference is not None:
        predictions = predictions.sort_values("row_index").reset_index(drop=True)
        reference = reference.sort_values("row_index").reset_index(drop=True)
        validate_predictions(reference, data, group_column)
        if not np.array_equal(predictions.row_index, reference.row_index):
            raise ValueError("Paired predictions have different row identities")
        if not np.allclose(predictions.label, reference.label, rtol=1e-6, atol=1e-7):
            raise ValueError("Paired predictions have different labels")
    group_values = validate_predictions(predictions, data, group_column)
    groups = pd.unique(group_values)
    if len(groups) < 2:
        raise ValueError("At least two independent parent structures are required")
    group_rows = {group: np.flatnonzero(group_values == group) for group in groups}
    labels, values = predictions.label.to_numpy(), predictions.prediction.to_numpy()
    references = None if reference is None else reference.prediction.to_numpy()
    estimates, differences = [], []
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        rows = np.concatenate([group_rows[group] for group in sampled])
        estimate = metric_value(metric, labels[rows], values[rows])
        other = None if references is None else metric_value(metric, labels[rows], references[rows])
        if not np.isfinite(estimate) or (other is not None and not np.isfinite(other)):
            continue
        estimates.append(estimate)
        if other is not None:
            differences.append(estimate - other)
    if not estimates:
        raise ValueError("No finite bootstrap replicates")
    point = float(metric_value(metric, labels, values))
    result = {"metric": metric, "point_estimate": point,
              "ci_95_low": float(np.quantile(estimates, .025)),
              "ci_95_high": float(np.quantile(estimates, .975)),
              "bootstrap_iterations": len(estimates), "requested_iterations": iterations,
              "seed": seed, "n_parent_structures": len(groups), "n_spectra": len(labels),
              "interval": "pointwise 95% percentile structure-cluster bootstrap",
              "scope": "fixed predictions and comparator; not training-seed or simultaneous uncertainty"}
    if references is not None:
        other = float(metric_value(metric, labels, references))
        result.update(reference_point_estimate=other, paired_difference=point - other,
                      paired_difference_ci_95_low=float(np.quantile(differences, .025)),
                      paired_difference_ci_95_high=float(np.quantile(differences, .975)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--reference-predictions", type=Path)
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--group-column", default="material_id")
    parser.add_argument("--metric", choices=["accuracy", "f1", "mae", "r2"], required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args(argv)
    if cli.output.exists():
        raise FileExistsError(cli.output)
    result = cluster_bootstrap(pd.read_csv(cli.predictions), pd.read_csv(cli.data_file, comment="#"),
                               reference=pd.read_csv(cli.reference_predictions) if cli.reference_predictions else None,
                               group_column=cli.group_column, metric=cli.metric,
                               iterations=cli.iterations, seed=cli.seed)
    result["inputs"] = {"predictions": str(cli.predictions), "data": str(cli.data_file)}
    if cli.reference_predictions:
        result["inputs"]["reference"] = str(cli.reference_predictions)
    save_json(cli.output, result)
    print(json.dumps(result, indent=2))
