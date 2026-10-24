"""Experiment wrappers scope backend settings across all inference calls."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from spt import evaluate
from spt.experiments import bands, energy_windows, uncertainty


@pytest.mark.parametrize('module', [bands, energy_windows, uncertainty])
def test_experiment_backend_covers_run_and_restores(module):
    original = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    settings = {'cuda_matmul_allow_tf32': not original[0], 'cudnn_allow_tf32': not original[1]}
    config = SimpleNamespace(inference_backend=settings)
    cli = SimpleNamespace(config='config.yaml', device='cpu', batch_size=128, seed=42, passes=30)

    def run(actual_cli, args, device, metadata):
        assert actual_cli is cli
        assert args is config
        assert device.type == 'cpu'
        assert metadata['configured'] == settings
        assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == (not original[0], not original[1])
        return 'completed'

    with patch.object(module, 'load_config', return_value=config), patch.object(module, '_run', side_effect=run):
        if module is bands:
            with patch.object(module, 'resolve_model_arguments', return_value=cli):
                assert module.main(['--output-dir', 'unused']) == 'completed'
        else:
            with patch.object(module, 'parse_args', return_value=cli):
                assert module.main([]) == 'completed'
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == original


@pytest.mark.parametrize('module', [bands, energy_windows, uncertainty])
def test_publication_route_rejects_wrong_device_before_run(module):
    cli = SimpleNamespace(config='config.yaml', device='cpu', batch_size=128, seed=42, passes=30)
    config = SimpleNamespace(reference_device='cuda')
    with patch.object(module, 'load_config', return_value=config), patch.object(module, '_run') as run:
        with pytest.raises(ValueError, match='publication-reference replay requires CUDA'):
            if module is bands:
                with patch.object(module, 'resolve_model_arguments', return_value=cli):
                    module.main(['--output-dir', 'unused'])
            else:
                with patch.object(module, 'parse_args', return_value=cli):
                    module.main([])
        run.assert_not_called()


def test_general_evaluate_honors_arithmetic_but_allows_cpu():
    original = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    settings = {'cuda_matmul_allow_tf32': not original[0], 'cudnn_allow_tf32': not original[1]}
    args = SimpleNamespace(reference_device='cuda', inference_backend=settings)

    def run(actual_args, device, metadata):
        assert actual_args is args
        assert device.type == 'cpu'
        assert metadata['reference_device'] is None
        assert metadata['configured'] == settings
        assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == (not original[0], not original[1])
        raise RuntimeError('test exit')

    with patch.object(evaluate, 'resolve_device', return_value=torch.device('cpu')), patch.object(evaluate, '_testing', side_effect=run):
        with pytest.raises(RuntimeError, match='test exit'):
            evaluate.testing(args)
    assert args.reference_device == 'cuda'
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == original
