import numpy as np
import pandas as pd
import pytest
import torch

from spt.config.cfg import load_config
from spt.models import build_model
from spt.predict import prepare_spectra, predict_frame
from spt.registry import get_model


@pytest.fixture
def setup_input():
    root, entry = get_model('expert-metal-OS')
    args = load_config(root/entry['config'],project_root=root)
    frame = pd.DataFrame({'energy_eV':np.linspace(7700,7780,256),
                          'intensity':np.linspace(0,1,256),'absorber':'Co'})
    return args, frame


def test_preparation_shape(setup_input):
    args, frame = setup_input
    ids, categories, values = prepare_spectra(frame,args)
    assert ids == ['spectrum_1']
    assert categories.shape == (1,1)
    assert values.shape == (1,512)
    assert values.dtype == np.float32
    assert np.allclose(values[0,1::2],np.linspace(0,1,256),atol=1e-6)


def test_reject_relative_energy(setup_input):
    args, frame = setup_input
    frame['energy_eV'] = np.linspace(-10,60,256)
    with pytest.raises(ValueError,match='positive'):
        prepare_spectra(frame,args)


def test_untrained_forward_is_only_software_smoke(setup_input):
    args, frame = setup_input
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = build_model(args,torch.device('cpu'))
    result = predict_frame(model,frame,args)
    assert len(result)==1
    assert result.descriptor.iloc[0]=='OS'
    scores = result.filter(like='score_').iloc[0]
    assert np.isclose(scores.sum(),1,atol=1e-5)


def test_registered_normalization_precision(setup_input):
    from spt.utils.data_load.data_des import X_log_normalize_optimized, Y_log_normalize_optimized
    args, frame = setup_input
    args.normalization_dtype = 'float64'
    _, _, actual = prepare_spectra(frame, args)
    energy = X_log_normalize_optimized([frame.energy_eV.to_numpy(dtype=np.float64)],
                                      args.min_log_data, args.max_log_data, log_base=np.e)
    intensity = Y_log_normalize_optimized([frame.intensity.to_numpy(dtype=np.float64)])
    np.testing.assert_array_equal(actual[:, 0::2], np.asarray(energy, dtype=np.float32))
    np.testing.assert_array_equal(actual[:, 1::2], np.asarray(intensity, dtype=np.float32))
    args.normalization_dtype = 'float16'
    with pytest.raises(ValueError, match='normalization_dtype'):
        prepare_spectra(frame, args)


def test_registered_block_discards_padding(setup_input):
    args, frame = setup_input
    args.inference_block_size = 34
    frame = pd.concat([frame.assign(spectrum_id=f'Ti-{i}') for i in range(9)], ignore_index=True)

    class Model(torch.nn.Module):
        def forward(self, categories, continuous, labels):
            assert categories.shape[0] == continuous.shape[0] == 34
            torch.testing.assert_close(continuous[9:], continuous[:1].repeat(25, 1))
            return torch.zeros(34, len(args.OS['value'])), None

    result = predict_frame(Model(), frame, args)
    assert len(result) == 9
    assert result.spectrum_id.tolist() == [f'Ti-{i}' for i in range(9)]
