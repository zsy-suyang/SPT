"""Application CPU settings are scoped to the registered inference call."""
import json
from unittest.mock import patch

import pytest
import torch

from spt.applications import _cpu_threads
from spt.registry import project_root


@pytest.mark.parametrize('fail', [False, True])
def test_cpu_threads_applied_and_restored(fail):
    previous = torch.get_num_threads()
    configured = 2 if previous != 2 else 4
    try:
        with _cpu_threads({'torch_num_threads': configured}, 'cpu') as actual:
            assert actual == configured == torch.get_num_threads()
            if fail:
                raise RuntimeError('inference failure')
    except RuntimeError as exc:
        assert fail and str(exc) == 'inference failure'
    assert torch.get_num_threads() == previous


@pytest.mark.parametrize('execution,device', [({}, 'cpu'), ({'torch_num_threads': 4}, 'cuda:0')])
def test_other_execution_settings_are_unchanged(execution, device):
    previous = torch.get_num_threads()
    with patch('torch.set_num_threads') as setter:
        with _cpu_threads(execution, device) as actual:
            assert actual == previous
        setter.assert_not_called()


@pytest.mark.parametrize('value', [0, -1, True, 4.0, '4'])
def test_invalid_cpu_threads_rejected(value):
    with patch('torch.set_num_threads') as setter:
        with pytest.raises(ValueError, match='positive integer'):
            with _cpu_threads({'torch_num_threads': value}, 'cpu'):
                pytest.fail('Invalid thread count entered inference')
        setter.assert_not_called()


def test_registered_ni_cpu_threads():
    cohorts = json.loads((project_root() / 'configs/applications.json').read_text())['cohorts']
    registered = {cohort['id']: cohort['execution']['torch_num_threads']
                  for cohort in cohorts if 'torch_num_threads' in cohort['execution']}
    assert registered == {'ni_fe_activation': 4, 'ni_first_sodiation': 4}
