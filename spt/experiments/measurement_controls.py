"""Replay the reported energy-reference, broadening and input-support controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import fresh_output, save_json, set_seed
from spt.utils.execution import NI_GEOMETRY_ATOL


def computed_perturbation(energy, intensity, *, shift=0., sigma=0.):
    """Shift the signal on its fixed grid, or broaden it on a uniform auxiliary grid."""
    energy, intensity = np.asarray(energy, float), np.asarray(intensity, float)
    if sigma:
        step = min(np.diff(energy).min(), sigma / 3.)
        n = max(2, int(np.ceil((energy[-1] - energy[0]) / step)) + 1)
        grid = np.linspace(energy[0], energy[-1], n)
        values = np.interp(grid, energy, intensity)
        width = sigma / (grid[1] - grid[0])
        radius = max(1, int(np.ceil(4 * width)))
        positions = np.arange(-radius, radius + 1)
        kernel = np.exp(-positions**2 / (2 * width**2))
        kernel /= kernel.sum()
        values = np.convolve(np.pad(values, radius, mode='reflect'), kernel, mode='valid')
        return np.interp(energy, grid, values)
    return np.interp(energy - shift, energy, intensity)


def transform_prepared(frame, *, offset=0., sigma=0., drift=0.):
    """The measured-cohort controls retain each spectrum's provided sampling grid."""
    from scipy.ndimage import gaussian_filter1d
    groups = []
    for _, group in frame.groupby('spectrum_id', sort=False):
        group = group.copy()
        energy = group.energy_eV.to_numpy(float)
        values = group.intensity.to_numpy(float)
        if sigma:
            values = gaussian_filter1d(values, sigma / np.median(np.diff(energy)), mode='nearest')
        if drift:
            values = values * (1 + drift * np.linspace(-1, 1, len(values)))
        group['energy_eV'] = energy + offset
        group['intensity'] = values
        groups.append(group)
    return pd.concat(groups, ignore_index=True)


def co_fixed_grid(source, delta):
    grid = np.linspace(7712.4037160060025, 7774.945278034512, 256)
    rows = []
    for identifier, group in source.groupby('spectrum_id', sort=False):
        energy, values = group.energy_eV.to_numpy(), group.intensity.to_numpy()
        if grid[0] - delta < energy[0] or grid[-1] - delta > energy[-1]:
            raise ValueError('The fixed Co grid requires extrapolation')
        rows.append(pd.DataFrame({'spectrum_id': identifier, 'energy_eV': grid,
                                  'intensity': np.interp(grid - delta, energy, values), 'absorber': 'Co'}))
    return pd.concat(rows, ignore_index=True)


def lco_input(source, *, delta=0., padding='constant'):
    shift = 1.345
    start = float(source.iloc[0].energy_min_eV) + shift + delta
    grid = np.linspace(start, start + 7774.94294921875 - 7711.545, 256)
    rows = []
    for record in source.itertuples():
        energy, values = np.asarray(json.loads(record.spectrum_2x256_json), dtype=float)
        energy = energy + shift
        if grid[-1] > energy[-1] + 1e-8:
            raise ValueError('The LCO grid exceeds the supplied post-edge support')
        sampled = np.interp(grid, energy, values)
        before = grid < energy[0]
        if padding.startswith('mean'):
            sampled[before] = np.mean(values[:int(padding[4:])])
        elif padding == 'linear10':
            sampled[before] = np.polyval(np.polyfit(energy[:10], values[:10], 1), grid[before])
        elif padding == 'zero_control':
            sampled[before] = 0.
        elif padding != 'constant':
            raise ValueError(f'Unknown pre-edge extension: {padding}')
        percentage = float(str(record.percentage).rstrip('%'))
        rows.append(pd.DataFrame({'spectrum_id': f'LCO-SOC-{percentage:g}',
                                  'scan': record.scan, 'energy_eV': grid, 'intensity': sampled, 'absorber': 'Co'}))
    return pd.concat(rows, ignore_index=True)


def _settings(root, cohort):
    cohorts = json.loads((root / 'configs/applications.json').read_text())['cohorts']
    return next(c for c in cohorts if c['id'] == cohort)


def _infer_conditions(root, cohort, conditions, tasks=None):
    import torch
    from spt.config.cfg import load_config
    from spt.evaluate import _load_model
    from spt.predict import predict_frame
    from spt.registry import get_model, verify_asset
    from spt.utils.execution import inference_backend
    selected = _settings(root, cohort)
    execution = selected['execution']
    device = execution['device']
    set_seed(execution['seed'])
    rows = []
    for task, model_id in selected['models'].items():
        if tasks and task not in tasks:
            continue
        _, entry = get_model(model_id, root)
        verify_asset(root / entry['checkpoint'], entry.get('checkpoint_bytes'))
        args = load_config(root / entry['config'], project_root=root,
                           overrides={'ckpt': str(root / entry['checkpoint'])})
        args.normalization_dtype = execution['normalization_dtype']
        args.inference_backend = execution.get('inference_backend', {})
        args.reference_device = torch.device(device).type
        with inference_backend(args, device):
            model, _ = _load_model(args, torch.device(device))
            for condition, frame in conditions.items():
                result = predict_frame(model, frame, args, device, execution['batch_size'])
                result['task'] = task
                result['condition'] = condition
                result['model_id'] = model_id
                rows.append(result)
            del model
    return pd.concat(rows, ignore_index=True)


def _agreement(predictions):
    rows = []
    for task, subset in predictions.groupby('task', sort=False):
        clean = subset[subset.condition.eq('clean')].set_index('spectrum_id').prediction
        for condition, group in subset.groupby('condition', sort=False):
            base = clean.loc[group.spectrum_id].to_numpy()
            current = group.prediction.to_numpy()
            classification = task not in ('NNRS', 'MOOD', 'NNOSM')
            rows.append({'task': task, 'condition': condition, 'n_spectra': len(group),
                         'agreement_with_clean': float(np.mean(current == base)) if classification else np.nan,
                         'mean_abs_prediction_change': float(np.mean(abs(current - base))) if not classification else np.nan})
    return pd.DataFrame(rows)


def _compare(actual, reference, keys, columns, *, atol=1e-6, require_finite=False):
    if not np.isfinite(atol) or atol < 0:
        raise ValueError('Control comparison tolerance must be finite and nonnegative')
    if actual.duplicated(keys).any() or reference.duplicated(keys).any():
        raise ValueError('Duplicate control result keys')
    joined = reference.merge(actual, on=keys, suffixes=('_ref', '_run'), how='outer', indicator=True)
    if not joined['_merge'].eq('both').all():
        raise ValueError('Control rows do not match the supplied reference')
    deviations = {}
    for column in columns:
        a, b = joined[column + '_ref'].to_numpy(float), joined[column + '_run'].to_numpy(float)
        if np.isinf(a).any() or np.isinf(b).any() or (
                require_finite and not (np.isfinite(a).all() and np.isfinite(b).all())):
            raise ValueError(f'Non-finite control result: {column}')
        if not np.allclose(a, b, atol=atol, rtol=0, equal_nan=True):
            raise ValueError(f'Reported control differs: {column}; maximum difference '
                             f'{np.nanmax(abs(a-b)):.6g}, atol={atol:g}, rtol=0')
        deviations[column] = float(np.nanmax(abs(a-b))) if np.isfinite(a).any() else 0.
    return {'passed': True, 'rows': len(joined), 'max_absolute_difference': deviations,
            'absolute_tolerance': atol, 'relative_tolerance': 0.}


def reported_computed_metric(predictions, classification):
    """Use the float32 TorchMetrics aggregation of the reported shift/broadening tables."""
    import torch
    from torchmetrics import R2Score
    from torchmetrics.classification import F1Score
    if classification:
        columns = [column for column in predictions if column.startswith('probability_')]
        probability = torch.as_tensor(predictions[columns].to_numpy(dtype=np.float32))
        label = torch.as_tensor(predictions.label.to_numpy(dtype=np.int64))
        return F1Score(task='multiclass', num_classes=len(columns), average='weighted')(probability, label).item()
    prediction = torch.as_tensor(predictions.prediction.to_numpy(dtype=np.float32))
    label = torch.as_tensor(predictions.label.to_numpy(dtype=np.float32))
    return R2Score()(prediction, label).item()


def run_measured(suite, root, output):
    references = root / 'reference_results/measurement_controls'
    checks = {}
    if suite in ('co', 'fe'):
        cohort = 'co_lisa_compounds' if suite == 'co' else 'fe_charging'
        prepared = pd.read_csv(root / _settings(root, cohort)['prepared_input']['path'])
        conditions = {'clean': prepared}
        for sigma in (.5, 1.):
            conditions[f'broadening_{sigma:.1f}eV'] = transform_prepared(prepared, sigma=sigma)
        if suite == 'co':
            for drift in (-.10, -.05, .05, .10):
                conditions[f'drift_{drift:+.2f}'] = transform_prepared(prepared, drift=drift)
        else:
            for delta in (-1., -.5, -.2, .2, .5, 1.):
                conditions[f'shift_{delta:+.1f}eV'] = transform_prepared(prepared, offset=delta)
        predictions = _infer_conditions(root, cohort, conditions)
        predictions.to_csv(output / 'predictions.csv', index=False)
        summary = _agreement(predictions)
        summary.to_csv(output / 'summary.csv', index=False)
        reference = pd.read_csv(references / ('co_primary.csv' if suite == 'co' else 'fe.csv'))
        if suite == 'fe':
            reference = reference.rename(columns={'hard_class_agreement': 'agreement_with_clean'})
            reference.loc[~reference.task.isin(['NNRS', 'MOOD']), 'mean_abs_prediction_change'] = np.nan
        checks['primary'] = _compare(summary, reference, ['task', 'condition'],
                                      ['agreement_with_clean', 'mean_abs_prediction_change'])
        if suite == 'co':
            source = pd.read_csv(root / 'datasets/controls/co_calibrated_source.csv')
            conditions = {'clean': co_fixed_grid(source, 0)}
            for delta in (-.2, .2, -.5, .5, -1., 1.):
                conditions[f'signal_shift_{delta:+.1f}eV'] = co_fixed_grid(source, delta)
            predictions = _infer_conditions(root, cohort, conditions, ['OS', 'CN', 'OCN', 'CN2', 'PSGO'])
            predictions.to_csv(output / 'fixed_grid_predictions.csv', index=False)
            summary = _agreement(predictions)
            summary.to_csv(output / 'fixed_grid_summary.csv', index=False)
            checks['fixed_grid'] = _compare(summary, pd.read_csv(references / 'co_grid.csv'),
                                             ['task', 'condition'], ['agreement_with_clean'])
        else:
            source = pd.read_csv(root / 'datasets/applications/fe_charging/source_context/model_input_arrays.csv')
            edge = source.set_index('scan').edge_E0_eV
            neutral = prepared.copy()
            neutral['energy_eV'] = neutral.energy_eV - neutral['scan'].map(edge) + edge.median()
            result = _infer_conditions(root, cohort, {'E0_neutralized': neutral}, ['OS'])
            result['scan'] = result.spectrum_id.str.extract(r'(\d+)$').astype(int)
            result['OS_prediction'] = result.prediction
            result['OS_expected'] = sum(float(k[6:]) * result[k] for k in result if k.startswith('score_'))
            result.to_csv(output / 'neutralized_predictions.csv', index=False)
            checks['neutralized'] = _compare(result, pd.read_csv(references / 'fe_neutral.csv'),
                                              ['scan'], ['OS_prediction', 'OS_expected'])
    elif suite == 'lco':
        source = pd.read_csv(root / 'datasets/applications/lco_measured_charging/source_context/supplied_processed_spectra.csv')
        conditions = {'reference_calibrated_fixed_absolute_grid': lco_input(source)}
        for delta in (-.2, -.5, .2, .5):
            name = f'anchor_{"minus" if delta < 0 else "plus"}_{abs(delta):.1f}_eV'
            conditions[name] = lco_input(source, delta=delta)
        for padding in ('mean5', 'mean10', 'linear10', 'zero_control'):
            conditions[f'padding_{padding}'] = lco_input(source, padding=padding)
        result = _infer_conditions(root, 'lco_measured_charging', conditions, ['OS'])
        scans = conditions[next(iter(conditions))].drop_duplicates('spectrum_id').set_index('spectrum_id')['scan']
        result['scan'] = result.spectrum_id.map(scans)
        result['hard_os'] = result.prediction
        result['posterior_os_index'] = sum(float(k[6:]) * result[k] for k in result if k.startswith('score_'))
        result.to_csv(output / 'predictions.csv', index=False)
        checks['lco'] = _compare(result, pd.read_csv(references / 'lco.csv'),
                                  ['scan', 'condition'], ['hard_os', 'posterior_os_index'])
    elif suite == 'ni':
        inputs = json.loads((root / 'datasets/controls/ni_fixed_inputs.json').read_text())
        conditions = {}
        for row in inputs:
            frame = pd.DataFrame({'spectrum_id': row['state'], 'energy_eV': row['energy'],
                                  'intensity': row['intensity'], 'absorber': 'Ni'})
            conditions.setdefault(row['condition'], []).append(frame)
        conditions = {key: pd.concat(value, ignore_index=True) for key, value in conditions.items()}
        result = _infer_conditions(root, 'ni_fe_activation', conditions, ['NNRS', 'MOOD'])
        result['state'] = result.spectrum_id
        result.to_csv(output / 'predictions.csv', index=False)
        checks['ni'] = _compare(result, pd.read_csv(references / 'ni.csv'),
                                 ['state', 'condition', 'task'], ['prediction'],
                                 atol=NI_GEOMETRY_ATOL, require_finite=True)
        import platform
        import torch
        checks['ni']['execution'] = {
            **_settings(root, 'ni_fe_activation')['execution'],
            'dtype': 'float32', 'torch_version': str(torch.__version__),
            'numpy_version': np.__version__, 'python_version': platform.python_version(),
            'platform': platform.system(), 'torch_num_threads': torch.get_num_threads(),
        }
    return checks


def run_computed(root, output):
    import torch
    from spt.config.cfg import load_config
    from spt.registry import get_model
    from spt.utils.execution import inference_backend
    from spt.utils.data_load.data_des import parse_column_to_numpy
    from .common import load_dataset, load_model, predict, raw_to_cont, verify_reference
    tasks = {'metal': ['OS', 'CN', 'OCN', 'CN2', 'PSGO', 'NNRS', 'MOOD'],
             'oxygen': ['CN', 'CN2', 'PSGO', 'NNRS', 'NNOSM']}
    conditions = {'clean': {}, 'energy_shift_minus_0.5_eV': {'shift': -.5},
                  'energy_shift_plus_0.5_eV': {'shift': .5},
                  'gaussian_broadening_sigma_0.5_eV': {'sigma': .5},
                  'gaussian_broadening_sigma_1.0_eV': {'sigma': 1.}}
    rows = []
    for site, descriptors in tasks.items():
        for task in descriptors:
            _, entry = get_model(f'expert-{site}-{task}', root)
            args = load_config(root / entry['config'], project_root=root)
            set_seed(42)
            dataset = load_dataset(args)
            raw = pd.read_csv(args.data_file, comment='#', usecols=['XAS_Relative_energy', 'XAS_Intensity']).iloc[dataset.indices]
            energies = parse_column_to_numpy(raw.XAS_Relative_energy)
            intensities = parse_column_to_numpy(raw.XAS_Intensity)
            device = torch.device('cuda:0')
            directory = output / f'{site}-{task}'
            directory.mkdir()
            with inference_backend(args, device):
                model = load_model(args, root / entry['checkpoint'], device)
                for condition, options in conditions.items():
                    values = [computed_perturbation(e, y, **options) for e, y in zip(energies, intensities)]
                    dataset.cont = raw_to_cont(energies, values, args)
                    metrics, predictions = predict(model, dataset, device, 128)
                    if condition == 'clean':
                        verify_reference(predictions, root / entry['reference_predictions'])
                    predictions.to_csv(directory / f'{condition}.csv', index=False)
                    classification = getattr(args, task)['mode'] == 'Cls'
                    rows.append({'site': site, 'descriptor': task, 'n': len(predictions), 'condition': condition,
                                 'metric': 'weighted_f1' if classification else 'r2',
                                 'value': reported_computed_metric(predictions, classification)})
                del model
    result = pd.DataFrame(rows)
    result.to_csv(output / 'summary.csv', index=False)
    return {'computed': _compare(result, pd.read_csv(root / 'reference_results/measurement_controls/computed.csv'),
                                   ['site', 'descriptor', 'condition'], ['value'], atol=1e-6)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['computed', 'co', 'fe', 'lco', 'ni'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--project-root')
    cli = parser.parse_args(argv)
    from spt.registry import project_root
    root = project_root(cli.project_root)
    output = fresh_output(cli.output)
    checks = run_computed(root, output) if cli.suite == 'computed' else run_measured(cli.suite, root, output)
    save_json(output / 'verification.json', {'suite': cli.suite, 'reference_comparisons': checks})
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
