from pathlib import Path

from spt.registry import get_model, load_registry, verify_asset


def test_registry_unique_and_portable():
    root, registry = load_registry()
    ids = [m['id'] for m in registry['models']]
    assert len(ids) == len(set(ids))
    assert len([m for m in registry['models'] if m['family']=='expert']) == 12
    assert len([m for m in registry['models'] if m['family']=='joint']) == 2
    for item in registry['models']:
        assert (root/item['config']).is_file()
        assert not Path(item['checkpoint']).is_absolute()
        assert item['checkpoint_bytes'] > 0


def test_registered_score():
    _, model = get_model('expert-metal-OS')
    assert abs(model['expected_score']-0.9664983153343201) < 1e-12


def test_missing_weight_is_not_silently_downloaded(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        verify_asset(tmp_path/'missing.pth',None)


def test_asset_size_is_checked_when_recorded(tmp_path):
    import pytest
    path = tmp_path / 'asset.bin'
    path.write_bytes(b'asset')
    assert verify_asset(path, 5) == path
    with pytest.raises(ValueError, match='size mismatch'):
        verify_asset(path, 4)


def test_asset_without_recorded_size_is_usable(tmp_path):
    path = tmp_path / 'asset.bin'
    path.write_bytes(b'asset')
    assert verify_asset(path) == path
