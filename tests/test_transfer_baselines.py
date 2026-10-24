"""Frozen transfer baseline mapping and reference-prediction contracts."""
import json
from pathlib import Path
import unittest

import pandas as pd

from spt.experiments.transfer_baselines import inference_blocks, load_cases, summarize, verify_reference

ROOT = Path(__file__).resolve().parents[1]


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'source_row': [4, 8], 'label': [2, 3],
                                   'prediction': [2, 3],
                                   'class_values': ['[2, 3]'] * 2,
                                   'probabilities': ['[0.8, 0.2]', '[0.1, 0.9]']})

    def test_reference_rows_can_be_reordered(self):
        result = verify_reference(self.frame, self.frame.iloc[::-1])
        self.assertTrue(result['hard_predictions_equal'])
        self.assertEqual(result['max_absolute_probability_difference'], 0)

    def test_duplicate_identity_rejected(self):
        changed = self.frame.copy()
        changed['source_row'] = 4
        with self.assertRaisesRegex(ValueError, 'identities'):
            verify_reference(self.frame, changed)

    def test_hard_prediction_change_always_rejected(self):
        changed = self.frame.copy()
        changed.loc[0, 'prediction'] = 3
        with self.assertRaisesRegex(ValueError, 'prediction'):
            verify_reference(changed, self.frame, strict_probabilities=False)

    def test_soft_difference_reported_and_strict_mode_enforced(self):
        changed = self.frame.copy()
        changed.loc[0, 'probabilities'] = '[0.79, 0.21]'
        result = verify_reference(changed, self.frame, strict_probabilities=False)
        self.assertFalse(result['probabilities_within_tolerance'])
        with self.assertRaisesRegex(ValueError, 'probabilities differ'):
            verify_reference(changed, self.frame)

    def test_class_order_change_rejected(self):
        changed = self.frame.copy()
        changed['class_values'] = '[3, 2]'
        with self.assertRaisesRegex(ValueError, 'class order'):
            verify_reference(changed, self.frame)

    def test_inference_blocks_are_label_blind_and_complete(self):
        frame = pd.DataFrame({'transfer_group': ['a', 'b', 'a']})
        blocks = inference_blocks(frame, {'inference_parent_blocks': [['b'], ['a']]})
        self.assertEqual([block.tolist() for block in blocks], [[1], [0, 2]])
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            inference_blocks(frame, {'inference_parent_blocks': [['a'], ['a']]})


class CaseTests(unittest.TestCase):
    def test_complete_case_map_and_portable_config_paths(self):
        from spt.config import load_config
        _, cases = load_cases(ROOT)
        self.assertEqual(len(cases), 54)
        self.assertEqual(sum(len(case['evaluations']) for case in cases.values()), 63)
        for case in cases.values():
            with self.subTest(case=case['id']):
                args = load_config(ROOT / case['config'], ROOT)
                self.assertEqual(args.task, case['task'])
                self.assertEqual(args.architecture, case['architecture'])
                self.assertEqual(args.seed, 42)
                self.assertFalse(Path(case['checkpoint']).is_absolute())
                self.assertTrue((ROOT / case['manifest']).is_file())
                if case['training_mode'] == 'tl':
                    self.assertIsNotNone(case['initial_checkpoint'])
                    self.assertTrue((ROOT / case['source_config']).is_file())
                elif case['training_mode'] == 'scratch':
                    self.assertIsNone(args.ckpt)

    @unittest.skipUnless((ROOT / 'datasets/transfer/target/OS.csv').is_file()
                         and (ROOT / 'reference_predictions/transfer/baselines').is_dir(),
                         'Transfer data and reference prediction bundle not installed')
    def test_all_63_reported_score_cells(self):
        _, cases = load_cases(ROOT)
        result = summarize(ROOT, list(cases.values()))
        self.assertEqual(len(result), 63)
        self.assertEqual(set(result.n), {114, 399})

    @unittest.skipUnless((ROOT / 'datasets/transfer/train/OS/scratch_10.csv').is_file(),
                         'Transfer data bundle not installed')
    def test_target_training_rows_match_frozen_budgets(self):
        _, cases = load_cases(ROOT)
        manifest = json.loads((ROOT / 'data/splits/transfer/manifest.json').read_text())
        for case in cases.values():
            if case['training_mode'] == 'zeroshot':
                continue
            with self.subTest(case=case['id']):
                frame = pd.read_csv(ROOT / case['training_data'], index_col=0)
                budget = manifest['train_budgets'][str(case['training_fraction'])]
                self.assertEqual(set(frame.source_row), set(budget['train_source_rows']))
                self.assertEqual(len(frame), case['target_train_n'])
                self.assertFalse(set(frame.parent_group) & set(manifest['single_split']['test_groups']))


if __name__ == '__main__':
    unittest.main()
