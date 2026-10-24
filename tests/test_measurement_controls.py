"""Input transforms for the paper's existing controls, without model training."""
import json

import numpy as np
import pandas as pd
import pytest

from spt.experiments.measurement_controls import (
    _compare, co_fixed_grid, computed_perturbation, lco_input, transform_prepared,
    reported_computed_metric,
)
from spt.registry import project_root
from spt.utils.execution import NI_GEOMETRY_ATOL


def test_computed_shift_keeps_coordinates_and_clamps_edges():
    energy = np.arange(5.)
    intensity = np.array([0., 1., 3., 1., 0.])
    np.testing.assert_array_equal(computed_perturbation(energy, intensity), intensity)
    np.testing.assert_array_equal(computed_perturbation(energy, intensity, shift=1), [0., 0., 1., 3., 1.])
    np.testing.assert_allclose(computed_perturbation(energy, np.ones(5), sigma=.5), 1.)


def test_measured_coordinate_shift_does_not_shift_signal():
    frame = pd.DataFrame({'spectrum_id': 'a', 'energy_eV': np.arange(10.) + 7000,
                          'intensity': np.linspace(0, 1, 10), 'absorber': 'Fe'})
    result = transform_prepared(frame, offset=.2)
    np.testing.assert_allclose(result.energy_eV, frame.energy_eV + .2)
    np.testing.assert_array_equal(result.intensity, frame.intensity)
    drifted = transform_prepared(frame, drift=.1)
    np.testing.assert_allclose(drifted.intensity, frame.intensity * np.linspace(.9, 1.1, 10))
    pd.testing.assert_frame_equal(transform_prepared(frame), frame)


def test_lco_primary_matches_prepared_input():
    directory = project_root() / 'datasets/applications/lco_measured_charging'
    source = directory / 'source_context/supplied_processed_spectra.csv'
    if not source.is_file():
        pytest.skip('Install the prepared LCO data')
    actual = lco_input(pd.read_csv(source))
    prepared = pd.read_csv(directory / 'prepared.csv')
    assert actual.spectrum_id.tolist() == prepared.spectrum_id.tolist()
    np.testing.assert_allclose(actual.energy_eV, prepared.energy_eV, atol=1e-9, rtol=0)
    np.testing.assert_allclose(actual.intensity, prepared.intensity, atol=1e-9, rtol=0)


def test_co_grid_has_support_for_all_reported_shifts():
    path = project_root() / 'datasets/controls/co_calibrated_source.csv'
    if not path.is_file():
        pytest.skip('Install the Co reference data')
    source = pd.read_csv(path)
    primary = co_fixed_grid(source, 0.)
    assert len(primary) == 8 * 256
    for delta in (-1., -.5, -.2, .2, .5, 1.):
        result = co_fixed_grid(source, delta)
        np.testing.assert_array_equal(result.energy_eV, primary.energy_eV)
    with pytest.raises(ValueError, match='extrapolation'):
        co_fixed_grid(source, 1000.)


def test_ni_controls_only_contain_reported_endpoints_and_windows():
    path = project_root() / 'datasets/controls/ni_fixed_inputs.json'
    if not path.is_file():
        pytest.skip('Install the Ni control data')
    records = json.loads(path.read_text())
    assert len(records) == 8
    assert {row['state'] for row in records} == {'Dry', '4 h'}
    assert {row['condition'] for row in records} == {
        'provided_70_eV_window', 'provided_window_foil_shift_diagnostic',
        'train_only_median_grid', 'train_grid_foil_shift_diagnostic'}


def test_control_comparison_rejects_changed_or_missing_results():
    reference = pd.DataFrame({'task': ['OS', 'CN'], 'value': [1., .75]})
    assert _compare(reference.copy(), reference, ['task'], ['value'])['passed']
    changed = reference.copy()
    changed.loc[1, 'value'] = .8
    with pytest.raises(ValueError, match='differs'):
        _compare(changed, reference, ['task'], ['value'])
    with pytest.raises(ValueError, match='rows'):
        _compare(reference.iloc[:1], reference, ['task'], ['value'])


def test_computed_metric_retains_original_precision():
    import torch
    from torchmetrics import R2Score
    frame = pd.DataFrame({'label': [2.5, 2.51, 2.52, 2.6], 'prediction': [2.48, 2.51, 2.55, 2.59]})
    expected = R2Score()(torch.tensor(frame.prediction.to_numpy(), dtype=torch.float32),
                         torch.tensor(frame.label.to_numpy(), dtype=torch.float32)).item()
    assert reported_computed_metric(frame, False) == expected
    classes = pd.DataFrame({'label': [0, 1, 1], 'probability_0': [.9, .1, .4],
                            'probability_1': [.1, .9, .6]})
    assert reported_computed_metric(classes, True) == 1.


def test_ni_comparison_records_precision_without_relaxing_other_controls():
    reference = pd.DataFrame({'state': ['Dry'], 'prediction': [2.7]})
    actual = reference.copy()
    actual['prediction'] += 1.4e-6
    check = _compare(actual, reference, ['state'], ['prediction'],
                     atol=NI_GEOMETRY_ATOL, require_finite=True)
    assert check['passed']
    assert check['absolute_tolerance'] == 1e-5
    assert check['relative_tolerance'] == 0
    assert check['max_absolute_difference']['prediction'] == pytest.approx(1.4e-6)
    with pytest.raises(ValueError, match='differs'):
        _compare(actual, reference, ['state'], ['prediction'])
    actual['prediction'] = reference.prediction + 1.1 * NI_GEOMETRY_ATOL
    with pytest.raises(ValueError, match='differs'):
        _compare(actual, reference, ['state'], ['prediction'], atol=NI_GEOMETRY_ATOL)


@pytest.mark.parametrize('invalid', [np.nan, np.inf, -np.inf])
def test_ni_comparison_rejects_nonfinite_values(invalid):
    reference = pd.DataFrame({'state': ['Dry'], 'prediction': [invalid]})
    with pytest.raises(ValueError, match='Non-finite'):
        _compare(reference.copy(), reference, ['state'], ['prediction'],
                 atol=NI_GEOMETRY_ATOL, require_finite=True)


def test_ni_control_suite_uses_registered_precision(tmp_path, monkeypatch):
    from spt.experiments import measurement_controls as controls
    root = project_root()
    required = [root / 'datasets/controls/ni_fixed_inputs.json',
                root / 'reference_results/measurement_controls/ni.csv']
    if any(not path.is_file() for path in required):
        pytest.skip('Install the Ni control inputs and reference results')
    reference = pd.read_csv(required[1])

    def fake_infer(*args):
        result = reference.copy()
        result['spectrum_id'] = result.state
        result['prediction'] += 1.4e-6
        return result

    monkeypatch.setattr(controls, '_infer_conditions', fake_infer)
    check = controls.run_measured('ni', root, tmp_path)['ni']
    assert check['passed'] and check['rows'] == 16
    assert check['absolute_tolerance'] == NI_GEOMETRY_ATOL
    assert check['execution']['device'] == 'cpu'
    assert check['execution']['dtype'] == 'float32'
