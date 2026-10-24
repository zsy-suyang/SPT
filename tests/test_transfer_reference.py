"""SPT transfer keeps hard results exact while exposing numerical score drift."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from spt.experiments import transfer


@pytest.fixture
def prediction_table():
    return pd.DataFrame({
        "source_row": [4, 8], "label": [2, 3], "prediction": [2, 3],
        "class_values": ["[2, 3]", "[2, 3]"],
        "probabilities": ["[0.8, 0.2]", "[0.1, 0.9]"],
    })


def test_identical_reference_and_reordered_rows(prediction_table):
    result = transfer.verify_reference(prediction_table, prediction_table.iloc[::-1])
    assert result["row_identities_equal"]
    assert result["labels_equal"]
    assert result["hard_predictions_equal"]
    assert result["f1_weighted_equal"]
    assert result["reference_f1_weighted"] == 1.
    assert result["probabilities_within_tolerance"]
    assert result["max_absolute_probability_difference"] == 0.
    assert result["strict_probabilities"] is False


def test_score_drift_is_recorded_by_default_and_rejected_only_when_strict(prediction_table):
    actual = prediction_table.copy()
    actual.loc[0, "probabilities"] = "[0.7995, 0.2005]"
    result = transfer.verify_reference(actual, prediction_table)
    assert result["hard_predictions_equal"] and result["f1_weighted_equal"]
    assert result["probabilities_within_tolerance"] is False
    assert result["max_absolute_probability_difference"] == pytest.approx(0.0005)
    assert result["probability_atol"] == 5e-6
    assert result["probability_rtol"] == 1e-5
    with pytest.raises(ValueError, match="Frozen reference probabilities differ"):
        transfer.verify_reference(actual, prediction_table, strict_probabilities=True)
    result = transfer.verify_reference(actual, prediction_table, atol=0.001,
                                       strict_probabilities=True)
    assert result["strict_probabilities"] and result["probabilities_within_tolerance"]


@pytest.mark.parametrize("column", ["label", "prediction"])
def test_discrete_changes_always_fail(prediction_table, column):
    actual = prediction_table.copy()
    actual.loc[0, column] = 3
    with pytest.raises(ValueError, match=f"Frozen reference {column} differs"):
        transfer.verify_reference(actual, prediction_table)


@pytest.mark.parametrize("which", ["actual", "reference"])
@pytest.mark.parametrize("identities", [[4, 4], [4, None], [4, 9]])
def test_invalid_row_identities_fail(prediction_table, which, identities):
    actual, reference = prediction_table.copy(), prediction_table.copy()
    (actual if which == "actual" else reference)["source_row"] = identities
    with pytest.raises(ValueError, match="row identities"):
        transfer.verify_reference(actual, reference)


@pytest.mark.parametrize("which", ["actual", "reference"])
@pytest.mark.parametrize("column,value", [
    ("label", np.nan), ("prediction", np.inf),
    ("probabilities", "[NaN, 0.2]"),
    ("probabilities", "[0.8, Infinity]"),
    ("class_values", "[2, NaN]"),
])
def test_nonfinite_values_fail_even_when_not_strict(prediction_table, which, column, value):
    actual, reference = prediction_table.copy(), prediction_table.copy()
    changed = actual if which == "actual" else reference
    if column in ("label", "prediction"):
        changed[column] = changed[column].astype(float)
    changed.loc[0, column] = value
    with pytest.raises(ValueError, match="Non-finite"):
        transfer.verify_reference(actual, reference)


@pytest.mark.parametrize("column", ["probabilities", "class_values"])
@pytest.mark.parametrize("which", ["actual", "reference"])
def test_missing_score_schema_is_not_silently_skipped(prediction_table, column, which):
    actual, reference = prediction_table.copy(), prediction_table.copy()
    if which == "actual":
        actual = actual.drop(columns=column)
    else:
        reference = reference.drop(columns=column)
    with pytest.raises(ValueError, match="require probabilities and class_values"):
        transfer.verify_reference(actual, reference)


@pytest.mark.parametrize("value", ["oops", "0.8", "[]", "[0.7, 0.2]", "[1.1, -0.1]"])
def test_malformed_or_invalid_probabilities_fail(prediction_table, value):
    actual = prediction_table.copy()
    actual["probabilities"] = value
    with pytest.raises(ValueError):
        transfer.verify_reference(actual, prediction_table)


def test_shape_mismatch_fails(prediction_table):
    actual = prediction_table.copy()
    actual["class_values"] = "[2, 3, 4]"
    actual["probabilities"] = ["[0.8, 0.1, 0.1]", "[0.1, 0.8, 0.1]"]
    with pytest.raises(ValueError, match="probability shapes differ"):
        transfer.verify_reference(actual, prediction_table)


def test_class_order_mismatch_fails(prediction_table):
    actual = prediction_table.copy()
    actual["class_values"] = "[3, 2]"
    actual["probabilities"] = ["[0.2, 0.8]", "[0.9, 0.1]"]
    with pytest.raises(ValueError, match="class order differs"):
        transfer.verify_reference(actual, prediction_table)


def test_probabilities_must_agree_with_recorded_hard_classes(prediction_table):
    actual = prediction_table.copy()
    actual.loc[0, "probabilities"] = "[0.2, 0.8]"
    with pytest.raises(ValueError, match="disagree with hard predictions"):
        transfer.verify_reference(actual, prediction_table)


@pytest.mark.parametrize("atol,rtol", [(-1., 0.), (0., np.nan), (np.inf, 0.)])
def test_invalid_tolerances_fail(prediction_table, atol, rtol):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        transfer.verify_reference(prediction_table, prediction_table, atol=atol, rtol=rtol)


def test_scoreless_tables_remain_exact_but_cannot_request_strict_scores(prediction_table):
    frame = prediction_table.drop(columns=["probabilities", "class_values"])
    result = transfer.verify_reference(frame, frame)
    assert result["hard_predictions_equal"]
    assert result["probabilities_within_tolerance"] is None
    with pytest.raises(ValueError, match="requires probability tables"):
        transfer.verify_reference(frame, frame, strict_probabilities=True)


def test_labels_only_reference_preserves_classification_checks(prediction_table):
    reference = prediction_table.drop(columns=["probabilities", "class_values"])
    result = transfer.verify_reference(prediction_table, reference, classification=True)
    assert result["f1_weighted_equal"] and result["reference_f1_weighted"] == 1.
    assert result["probability_reference_available"] is False
    assert result["probabilities_within_tolerance"] is None
    assert result["max_absolute_probability_difference"] is None
    with pytest.raises(ValueError, match="requires probability tables"):
        transfer.verify_reference(prediction_table, reference, classification=True,
                                  strict_probabilities=True)
    changed = prediction_table.copy()
    changed.loc[0, "probabilities"] = "[NaN, 0.2]"
    with pytest.raises(ValueError, match="Non-finite"):
        transfer.verify_reference(changed, reference, classification=True)


@pytest.mark.parametrize("strict", [False, True])
def test_cli_forwards_strict_mode(monkeypatch, strict):
    monkeypatch.setattr(transfer, "evaluate", lambda args: args)
    argv = ["evaluate", "--config", "model.yaml", "--checkpoint", "model.pth",
            "--data-file", "data.csv", "--expected-rows", "2",
            "--reference-predictions", "reference.csv", "--output-dir", "unused"]
    if strict:
        argv.append("--strict-probabilities")
    assert transfer.main(argv).strict_probabilities is strict


def test_cli_rejects_strict_mode_without_reference():
    with pytest.raises(SystemExit):
        transfer.main(["evaluate", "--config", "model.yaml", "--checkpoint", "model.pth",
                       "--data-file", "data.csv", "--expected-rows", "2",
                       "--strict-probabilities", "--output-dir", "unused"])


@pytest.mark.parametrize("mode", ["default", "strict", "bad_f1", "labels_only"])
def test_evaluate_writes_only_valid_hard_results_and_records_execution(
        tmp_path, monkeypatch, prediction_table, mode):
    config, checkpoint = tmp_path / "config.yaml", tmp_path / "checkpoint.pth"
    config.touch()
    checkpoint.touch()
    data_file = tmp_path / "data.csv"
    pd.DataFrame({"source_row": [4, 8], "transfer_group": ["a", "b"],
                  "CN": [2, 3]}).to_csv(data_file)
    reference_file = tmp_path / "reference.csv"
    reference = (prediction_table.drop(columns=["probabilities", "class_values"])
                 if mode == "labels_only" else prediction_table)
    reference.to_csv(reference_file, index=False)
    model_args = SimpleNamespace(architecture="spt", multi_task=None, task="CN", seed=42,
                                 CN={"mode": "Cls", "value": [2, 3], "label": [0, 1]})
    monkeypatch.setattr("spt.config.cfg.load_config", lambda *args: model_args)
    monkeypatch.setattr("spt.utils.runtime.resolve_device", lambda args: torch.device("cpu"))
    monkeypatch.setattr(transfer, "set_seed", lambda seed: None)
    monkeypatch.setattr(transfer, "prepare_inputs", lambda *args: (
        torch.zeros(2, 1, dtype=torch.long), torch.zeros(2, 4)))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"net_state_dict": {}})

    class FrozenModel(torch.nn.Module):
        def forward(self, categories, continuous, labels):
            return torch.tensor([[0.7995, 0.2005], [0.1, 0.9]]).log(), None

    monkeypatch.setattr("spt.models.build_model", lambda *args: FrozenModel())
    if mode == "bad_f1":
        score_predictions = transfer.score_predictions

        def changed_metrics(*args):
            metrics, table, groups = score_predictions(*args)
            metrics["f1_weighted"] = 0.5
            return metrics, table, groups

        monkeypatch.setattr(transfer, "score_predictions", changed_metrics)
    # evaluate() changes these global execution settings as part of its protocol.
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    previous_precision = torch.get_float32_matmul_precision()
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0")
    output = tmp_path / "output"
    argv = [
        "evaluate", "--config", str(config), "--checkpoint", str(checkpoint),
        "--data-file", str(data_file), "--expected-rows", "2",
        "--reference-predictions", str(reference_file), "--output-dir", str(output),
    ]
    if mode == "strict":
        argv.append("--strict-probabilities")
    try:
        if mode in ("strict", "bad_f1"):
            message = "probabilities differ" if mode == "strict" else "weighted F1 differs"
            with pytest.raises(ValueError, match=message):
                transfer.main(argv)
            assert not output.exists()
            return
        result = transfer.main(argv)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous_tf32
        torch.set_float32_matmul_precision(previous_precision)
    assert result["reference_comparison"]["probabilities_within_tolerance"] is (
        None if mode == "labels_only" else False)
    assert result["reference_comparison"]["f1_weighted_equal"]
    assert result["metrics"]["f1_weighted"] == 1.
    saved = json.loads((output / "metrics.json").read_text())
    assert saved["execution"]["device"] == "cpu"
    assert saved["execution"]["torch_num_threads"] == torch.get_num_threads()
    actual = pd.read_csv(output / "predictions.csv")
    assert json.loads(actual.probabilities.iloc[0])[0] == pytest.approx(0.7995)
