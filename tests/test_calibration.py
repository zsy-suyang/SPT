import json

import numpy as np
import pandas as pd
import pytest

from spt.experiments.calibration import calibration_quantile, summarize_pair


def classification(indices, probabilities, labels):
    p = np.asarray(probabilities)
    return pd.DataFrame({"row_index": indices, "probabilities": [json.dumps(x) for x in probabilities],
                         "label": labels, "prediction": p.argmax(1),
                         "uncertainty": -(p*np.log(np.maximum(p, 1e-12))).sum(1)})


def test_order_statistic_and_small_sample_cap():
    assert calibration_quantile(np.arange(100), .95) == 95
    assert calibration_quantile([.1, .3, .2], .95) == .3
    with pytest.raises(ValueError):
        calibration_quantile([])
    with pytest.raises(ValueError):
        calibration_quantile([1], 1)


def test_empty_sets_are_retained():
    val = classification([1, 2], [[.95, .05], [.1, .9]], [0, 1])
    test = classification([3, 4], [[.6, .4], [.95, .05]], [0, 0])
    kind, result = summarize_pair(val, test)
    assert kind == "classification"
    assert result["empty_set_count"] == 1
    assert result["empirical_coverage"] == .5
    assert result["mean_prediction_set_size"] == .5
    assert result["error_detection_auroc"] is None


def test_common_regression_interval_and_inclusive_endpoint():
    val = pd.DataFrame({"row_index": [0, 1], "label": [0., 0.], "prediction": [.1, .2], "uncertainty": [.3, .1]})
    test = pd.DataFrame({"row_index": [2, 3], "label": [0., 0.], "prediction": [.2, .4], "uncertainty": [.1, .2]})
    _, result = summarize_pair(val, test)
    assert result["half_width"] == .2
    assert result["mean_interval_width"] == .4
    assert result["empirical_coverage"] == .5
    assert result["high_error_detection_auroc"] == 1


def test_reject_overlap_and_invalid_probability():
    val = classification([1], [[.9, .1]], [0])
    with pytest.raises(ValueError, match="overlap"):
        summarize_pair(val, val)
    test = classification([2], [[.9, .1]], [0])
    test.loc[0, "probabilities"] = "[0.5, 0.2]"
    with pytest.raises(ValueError, match="normalized"):
        summarize_pair(val, test)


def test_reference_calibration_if_installed(tmp_path):
    from spt.registry import project_root
    from spt.experiments.calibration import main
    root = project_root()
    if not (root / "reference_predictions/uncertainty/metal/OS/val.csv").is_file():
        pytest.skip("Calibration reference assets are external")
    assert main(["--root", str(root), "--output-dir", str(tmp_path / "summaries")]) == 0
