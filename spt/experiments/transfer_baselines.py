"""Evaluate the released NN-XANES, AMFormer and RF transfer checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import fresh_output, save_json, set_seed
from .transfer import descriptor_metrics, prepare_inputs, score_predictions


def load_cases(root):
    path = Path(root) / 'configs/transfer/baselines.json'
    collection = json.loads(path.read_text(encoding='utf-8'))
    cases = collection['models']
    if len({case['id'] for case in cases}) != len(cases):
        raise ValueError('Duplicate transfer baseline model identifiers')
    return collection, {case['id']: case for case in cases}


def target_frame(root, task, cohort, manifest_path):
    manifest = json.loads((Path(root) / manifest_path).read_text(encoding='utf-8'))
    frame = pd.read_csv(Path(root) / f'datasets/transfer/target/{task}.csv', index_col=0)
    if frame.source_row.duplicated().any():
        raise ValueError('Duplicate target source_row identifiers')
    expected = set(manifest['task_source_rows']['metal'][task])
    if set(frame.source_row) != expected:
        raise ValueError('Target rows differ from the frozen transfer manifest')
    group = manifest['group_column']
    assigned = frame[group].astype(str).map(manifest['single_split']['assignments'])
    if assigned.isna().any():
        raise ValueError('Unassigned target parent')
    if cohort == 'test':
        frame = frame.loc[assigned.eq('test')].copy()
    elif cohort != 'all':
        raise ValueError('cohort must be test or all')
    frame = frame.sort_values('source_row').reset_index(drop=True)
    frame['transfer_group'] = frame[group].astype(str)
    return frame


def verify_reference(actual, reference, atol=5e-6, rtol=1e-5, strict_probabilities=True):
    if (actual.source_row.duplicated().any() or reference.source_row.duplicated().any()
            or set(actual.source_row) != set(reference.source_row)):
        raise ValueError('Reference target row identities differ')
    reference = reference.set_index('source_row').loc[actual.source_row]
    for name in ('label', 'prediction'):
        if not np.array_equal(reference[name].to_numpy(), actual[name].to_numpy()):
            raise ValueError(f'Reference {name} differs')
    difference, close = None, None
    if 'probabilities' in reference and 'probabilities' in actual:
        expected = np.stack(reference.probabilities.map(json.loads))
        observed = np.stack(actual.probabilities.map(json.loads))
        if not np.array_equal(np.stack(reference.class_values.map(json.loads)),
                              np.stack(actual.class_values.map(json.loads))):
            raise ValueError('Reference probability class order differs')
        if expected.shape != observed.shape:
            raise ValueError('Reference probability shape differs')
        difference = float(np.abs(expected - observed).max())
        close = bool(np.allclose(expected, observed, rtol=rtol, atol=atol))
        if not close and strict_probabilities:
            raise ValueError(f'Reference probabilities differ (max difference {difference:.8g})')
    return {'row_identities_equal': True, 'labels_equal': True,
            'hard_predictions_equal': True, 'max_absolute_probability_difference': difference,
            'probabilities_within_tolerance': close, 'probability_atol': atol, 'probability_rtol': rtol}


def tree_predictions(estimator, cate, cont, frame, args):
    pieces = [cont.numpy()]
    for column, size in enumerate(args.categories):
        pieces.append(np.eye(size, dtype=np.float32)[cate[:, column].numpy()])
    features = np.concatenate(pieces, axis=1)
    prediction = estimator.predict(features)
    classes = estimator.classes_.tolist()
    probability = estimator.predict_proba(features)
    table = frame[['source_row', 'structure', 'transfer_group', 'site']].copy()
    table['label'], table['prediction'] = frame[args.task].to_numpy(), prediction
    table['class_values'] = json.dumps(classes)
    table['probabilities'] = [json.dumps(row.tolist()) for row in probability]
    table['label_supported_by_head'] = np.isin(table.label, classes)
    metrics = descriptor_metrics(table.label, prediction, classes)
    return metrics, table


def inference_blocks(frame, evaluation):
    """Retain the original evaluation batch membership without using labels."""
    parents = evaluation.get('inference_parent_blocks')
    if parents is None:
        return [np.arange(len(frame))]
    blocks = [np.flatnonzero(frame.transfer_group.isin(group).to_numpy()) for group in parents]
    joined = np.concatenate(blocks)
    if any(not len(block) for block in blocks) or not np.array_equal(np.sort(joined), np.arange(len(frame))):
        raise ValueError('Inference blocks must cover every target exactly once')
    return blocks


def evaluate_case(root, case, cohort, device='cpu', strict_probabilities=False):
    import torch
    from spt.config import load_config
    from spt.models import build_model
    root = Path(root).resolve()
    if cohort not in case['evaluations']:
        raise ValueError(f'{case["id"]} has no reported {cohort} evaluation')
    evaluation = case['evaluations'][cohort]
    frame = target_frame(root, case['task'], cohort, case['manifest'])
    if len(frame) != evaluation['n']:
        raise ValueError('Evaluation size differs from the frozen case')
    args = load_config(root / case['config'], root, {'device': device, 'ckpt': None})
    set_seed(int(args.seed))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    cate, cont = prepare_inputs(frame, args, 'metal')
    if case['architecture'] == 'random_forest':
        import joblib
        payload = joblib.load(root / case['checkpoint'])
        metrics, table = tree_predictions(payload['estimator'], cate, cont, frame, args)
    else:
        model = build_model(args, device).to(device)
        payload = torch.load(root / case['checkpoint'], map_location=device, weights_only=False)
        model.load_state_dict(payload['net_state_dict'], strict=True)
        model.eval()
        raw = np.empty((len(frame), args.out), dtype=np.float32)
        # AMFormer includes batch-dependent operations. Preserve both the saved
        # source_row order and batch limit; test is inferred independently of all.
        batch_size = int(case['evaluation_batch_size'])
        with torch.inference_mode():
            for block in inference_blocks(frame, evaluation):
                for start in range(0, len(block), batch_size):
                    indices = block[start:start + batch_size]
                    batch_cate = cate[indices].to(device)
                    batch_cont = cont[indices].to(device)
                    # The adapter computes an unused loss even at inference.
                    # Valid dummy labels do not affect its predictions.
                    dummy = torch.zeros(len(batch_cont), dtype=torch.long, device=device)
                    output, _ = model(batch_cate, batch_cont, dummy)
                    raw[indices] = output.detach().float().cpu().numpy()
        metrics, table, _ = score_predictions(raw, frame, args.task,
                                               getattr(args, args.task), 'transfer_group')
    reference = pd.read_csv(root / evaluation['reference_predictions'])
    comparison = verify_reference(table, reference, strict_probabilities=strict_probabilities)
    if not np.isclose(metrics['f1_weighted'], evaluation['expected_f1'], rtol=0, atol=1e-12):
        raise ValueError('Weighted F1 differs from the paper reference')
    result = {'id': case['id'], 'cohort': cohort, 'metrics': metrics,
              'reference_comparison': comparison,
              'checkpoint': case['checkpoint'], 'config': case['config'],
              'reference_predictions': evaluation['reference_predictions']}
    return result, table


def summarize(root, cases):
    """Recalculate reported scores from final per-spectrum predictions."""
    results = []
    for case in cases:
        for cohort, evaluation in case['evaluations'].items():
            frame = target_frame(root, case['task'], cohort, case['manifest'])
            table = pd.read_csv(Path(root) / evaluation['reference_predictions'])
            truth = frame[['source_row']].copy()
            truth['label'] = frame[case['task']].to_numpy()
            truth['prediction'] = table.set_index('source_row').loc[truth.source_row, 'prediction'].to_numpy()
            verify_reference(truth, table)
            classes = json.loads(table.class_values.iloc[0])
            metrics = descriptor_metrics(table.label, table.prediction, classes)
            if len(table) != evaluation['n'] or not np.isclose(
                    metrics['f1_weighted'], evaluation['expected_f1'], rtol=0, atol=1e-12):
                raise ValueError(f'Reference score differs: {case["id"]}/{cohort}')
            results.append({'id': case['id'], 'model': case['architecture'],
                            'task': case['task'], 'mode': case['training_mode'],
                            'training_fraction': case['training_fraction'],
                            'cohort': cohort, **metrics})
    return pd.DataFrame(results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['list', 'summarize', 'evaluate'])
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--model-ids', nargs='+', help='Defaults to all released baseline cases')
    parser.add_argument('--cohort', choices=['test', 'all'], default='test')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--strict-probabilities', action='store_true',
                        help='Also require probabilities within atol=5e-6, rtol=1e-5; '
                             'labels, hard predictions and paper F1 are always checked exactly')
    parser.add_argument('--output-dir')
    args = parser.parse_args(argv)
    collection, mapping = load_cases(args.project_root)
    ids = args.model_ids or list(mapping)
    if set(ids) - set(mapping):
        parser.error('Unknown model IDs: ' + ', '.join(sorted(set(ids) - set(mapping))))
    cases = [mapping[model_id] for model_id in ids]
    if args.operation == 'list':
        print('\n'.join(case['id'] for case in cases))
        return cases
    if not args.output_dir:
        parser.error('--output-dir is required for summarize/evaluate')
    if args.threads < 1:
        parser.error('--threads must be positive')
    output = fresh_output(args.output_dir)
    if args.operation == 'summarize':
        result = summarize(args.project_root, cases)
        result.to_csv(output / 'metrics.csv', index=False)
        print(f'Verified {len(result)} reported baseline scores')
        return result
    import torch
    torch.set_num_threads(args.threads)
    results = []
    for case in cases:
        result, table = evaluate_case(args.project_root, case, args.cohort, args.device,
                                      args.strict_probabilities)
        case_output = output / case['id']
        case_output.mkdir()
        table.to_csv(case_output / 'predictions.csv', index=False)
        save_json(case_output / 'metrics.json', result)
        results.append(result)
        print(f'{case["id"]}: n={len(table)}, F1={result["metrics"]["f1_weighted"]:.8f}')
    save_json(output / 'summary.json', results)
    return results


if __name__ == '__main__':
    main()
