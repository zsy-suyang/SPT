"""Replay the registered Ni–Fe weights; skip only when assets are not installed."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from spt.applications import main
from spt.registry import get_model
from spt.utils.execution import NI_GEOMETRY_ATOL


def test_eight_spectrum_reference(tmp_path):
    root, _ = get_model('app-ni_fe_activation-OCN')
    folder = root / 'datasets/applications/ni_fe_activation'
    tasks = ('OS', 'CN', 'OCN', 'NNRS', 'MOOD')
    required = [folder / name for name in (
        'prepared.csv', 'reference_predictions.csv', 'reference_predictions_long.csv',
        'reference_predictions_long.metadata.json')]
    required += [root / get_model(f'app-ni_fe_activation-{task}', root)[1]['checkpoint']
                 for task in tasks]
    if any(not path.is_file() for path in required):
        pytest.skip('Install the Ni–Fe application inputs and five registered weights.')
    original_threads = torch.get_num_threads()
    execution = json.loads((folder / 'reference_predictions_long.metadata.json').read_text())
    output = tmp_path / 'predictions.csv'
    try:
        torch.set_num_threads(16)
        main(['--cohort', 'ni_fe_activation', '--tasks', *tasks,
              '--project-root', str(root), '--device', 'cpu', '--output', str(output)])
        assert torch.get_num_threads() == 16
    finally:
        torch.set_num_threads(original_threads)
    actual = pd.read_csv(output)
    reference = pd.read_csv(folder / 'reference_predictions.csv').set_index('state')
    frozen = pd.read_csv(folder / 'reference_predictions_long.csv').set_index(
        ['spectrum_id', 'descriptor'])
    assert len(actual) == 8 * len(tasks)
    assert not actual.duplicated(['spectrum_id', 'descriptor']).any()
    for row in actual.to_dict('records'):
        task, state = row['descriptor'], row['spectrum_id']
        expected = reference.loc[state, task]
        if task in ('OS', 'CN', 'OCN'):
            assert row['prediction'] == expected
            _, entry = get_model(f'app-ni_fe_activation-{task}', root)
            from spt.config.cfg import load_config
            config = load_config(root / entry['config'], project_root=root)
            score_columns = [f'score_{value}' for value in getattr(config, task)['value']]
            scores = np.array([row[key] for key in score_columns])
            assert np.isclose(scores.sum(), 1, atol=1e-6)
            np.testing.assert_allclose(scores, frozen.loc[(state, task), score_columns],
                                       rtol=0, atol=5e-4)
        else:
            assert abs(row['prediction'] - expected) <= NI_GEOMETRY_ATOL
    metadata = json.loads(output.with_suffix('.metadata.json').read_text())
    assert root / metadata['input'] == folder / 'prepared.csv'
    assert metadata['models'] == {task: f'app-ni_fe_activation-{task}' for task in tasks}
    assert metadata['device'] == 'cpu'
    assert metadata['dtype'] == 'float32'
    assert metadata['torch_num_threads'] == execution['torch_num_threads'] == 4
    for asset in metadata['model_assets'].values():
        _, entry = get_model(asset['model_id'], root)
        assert asset['checkpoint'] == entry['checkpoint']
        assert asset['config'] == entry['config']
        assert (root / asset['checkpoint']).is_file()
        assert (root / asset['config']).is_file()
