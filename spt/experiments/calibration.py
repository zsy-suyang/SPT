"""Recompute validation-calibrated coverage from frozen MC-dropout predictions."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .common import fresh_output, save_json
from spt.registry import load_registry


def calibration_quantile(scores, coverage=0.95):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Calibration requires a nonempty finite score vector")
    if not 0 < coverage < 1:
        raise ValueError("Nominal coverage must be between zero and one")
    rank = min(len(scores), math.ceil(coverage * (len(scores) + 1)))
    return float(np.partition(scores, rank - 1)[rank - 1])


def _validate_frame(frame):
    required = ["row_index", "label", "prediction", "uncertainty"]
    if not set(required).issubset(frame) or not len(frame):
        raise ValueError(f"Expected nonempty MC table with {required}")
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite MC values")
    if frame.row_index.duplicated().any():
        raise ValueError("Duplicate spectrum row indices")
    if (frame.uncertainty < 0).any():
        raise ValueError("Uncertainty must be nonnegative")


def _probabilities(frame):
    values = np.asarray([json.loads(x) for x in frame.probabilities], dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 1)
            or not np.allclose(values.sum(1), 1, atol=1e-5, rtol=0)):
        raise ValueError("Expected finite normalized class-probability vectors")
    labels = frame.label.to_numpy(dtype=float)
    if np.any(labels != np.floor(labels)) or np.any(labels < 0) or np.any(labels >= values.shape[1]):
        raise ValueError("Label outside the encoded probability columns")
    if not np.array_equal(values.argmax(1), frame.prediction.to_numpy()):
        raise ValueError("Saved class predictions differ from mean-score argmax")
    return values, labels.astype(int)


def _auroc(target, values):
    return float(roc_auc_score(target, values)) if len(np.unique(target)) == 2 else None


def summarize_pair(validation, test, coverage=0.95):
    """Use the SI order statistic, including empty classification sets."""
    _validate_frame(validation)
    _validate_frame(test)
    if set(validation.row_index) & set(test.row_index):
        raise ValueError("Validation and test rows overlap")
    classification = "probabilities" in validation
    if classification != ("probabilities" in test):
        raise ValueError("Validation and test task types differ")
    result = {"n_validation": len(validation), "n_test": len(test), "nominal_coverage": coverage}
    if classification:
        vp, vy = _probabilities(validation)
        tp, ty = _probabilities(test)
        if vp.shape[1] != tp.shape[1]:
            raise ValueError("Validation and test class spaces differ")
        threshold = calibration_quantile(1 - vp[np.arange(len(vy)), vy], coverage)
        included = 1 - tp <= threshold
        sizes = included.sum(1)
        covered = included[np.arange(len(ty)), ty]
        truth = np.eye(tp.shape[1])[ty]
        result.update(
            covered_count=int(covered.sum()), empirical_coverage=float(covered.mean()),
            mean_prediction_set_size=float(sizes.mean()), empty_set_count=int((sizes == 0).sum()),
            empty_set_fraction=float((sizes == 0).mean()),
            brier_score=float(np.square(tp - truth).sum(1).mean()),
            mean_predictive_entropy=float(test.uncertainty.mean()),
            error_detection_auroc=_auroc(tp.argmax(1) != ty, test.uncertainty),
            calibration_threshold=threshold,
        )
        return "classification", result
    validation_errors = np.abs(validation.label.to_numpy() - validation.prediction.to_numpy())
    errors = np.abs(test.label.to_numpy() - test.prediction.to_numpy())
    threshold = calibration_quantile(validation_errors, coverage)
    covered = errors <= threshold
    result.update(
        covered_count=int(covered.sum()), empirical_coverage=float(covered.mean()),
        mean_interval_width=2 * threshold, half_width=threshold,
        mean_mc_standard_deviation=float(test.uncertainty.mean()),
        high_error_detection_auroc=_auroc(errors >= np.quantile(errors, 0.75), test.uncertainty),
    )
    return "regression", result


def _check_reference(results, expected, tolerance=1e-10):
    keys = ["site", "descriptor"]
    actual = results.set_index(keys)
    reference = expected.set_index(keys)
    if actual.index.duplicated().any() or reference.index.duplicated().any():
        raise ValueError("Duplicate reference task")
    if not actual.index.isin(reference.index).all():
        raise ValueError("Task missing from the expected summary")
    columns = [c for c in actual.columns if c in reference and pd.api.types.is_numeric_dtype(actual[c])]
    for column in columns:
        a = actual[column].to_numpy(dtype=float)
        b = reference.loc[actual.index, column].to_numpy(dtype=float)
        if not np.allclose(a, b, atol=tolerance, rtol=0, equal_nan=True):
            raise ValueError(f"Calibration reference mismatch in {column}: max difference {np.nanmax(np.abs(a-b))}")
    return {"passed": True, "tasks": len(actual), "numeric_columns": columns, "absolute_tolerance": tolerance}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--model-id", nargs="+")
    parser.add_argument("--validation", type=Path, help="Custom validation MC table")
    parser.add_argument("--test", type=Path, help="Custom test MC table")
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--output-dir", required=True, type=Path)
    cli = parser.parse_args(argv)
    if bool(cli.validation) != bool(cli.test):
        parser.error("Supply both --validation and --test")
    if cli.validation and cli.model_id:
        parser.error("Custom tables and --model-id are mutually exclusive")
    if not 0 < cli.coverage < 1:
        parser.error("--coverage must be between zero and one")
    rows = {"classification": [], "regression": []}
    inputs = []
    if cli.validation:
        kind, result = summarize_pair(pd.read_csv(cli.validation), pd.read_csv(cli.test), cli.coverage)
        rows[kind].append(result)
        root = None
        inputs.append({"validation": str(cli.validation), "test": str(cli.test)})
    else:
        root, registry = load_registry(cli.root)
        mapping = {x["id"]: x for x in registry["models"]}
        experiments = json.loads((root / "configs/experiments.json").read_text(encoding="utf-8"))
        ids = cli.model_id or experiments["experiments"]["uncertainty"]["model_ids"]
        if len(ids) != len(set(ids)):
            parser.error("Repeated model ID")
        for identity in ids:
            model = mapping.get(identity)
            if model is None or model["family"] != "uncertainty":
                parser.error(f"Expected an uncertainty model ID, got {identity}")
            site, task = model["site"], model["task"]
            base = root / "reference_predictions/uncertainty" / site / task
            kind, result = summarize_pair(pd.read_csv(base / "val.csv"), pd.read_csv(base / "test.csv"), cli.coverage)
            rows[kind].append({"site": site, "descriptor": task, **result})
            inputs.append({"model_id": identity, "validation": str((base / "val.csv").relative_to(root)),
                           "test": str((base / "test.csv").relative_to(root))})
    frames = {k: pd.DataFrame(v) for k, v in rows.items() if v}
    checks = {}
    if root is not None and cli.coverage == 0.95:
        for kind, frame in frames.items():
            expected = pd.read_csv(root / f"reference_results/uncertainty_{kind}.csv")
            checks[kind] = _check_reference(frame, expected)
    output = fresh_output(cli.output_dir)
    for kind, frame in frames.items():
        frame.to_csv(output / f"{kind}.csv", index=False)
    save_json(output / "metadata.json", {"nominal_coverage": cli.coverage, "inputs": inputs,
              "checks": checks, "threshold": "min(n, ceil(coverage*(n+1))) order statistic",
              "classification_empty_sets": "retained as non-coverage",
              "regression_width": "twice the validation absolute-residual threshold"})
    print(f"Wrote {sum(map(len, frames.values()))} calibration summaries to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
