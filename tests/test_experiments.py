"""Protocol tests use synthetic inputs; they do not stand in for weight replay."""
import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from spt.experiments import main as experiments_main
from spt.experiments.bootstrap import cluster_bootstrap, validate_predictions
from spt.experiments.common import fresh_output, verify_reference
from spt.experiments.gaussian import (
    AMPLITUDES, add_original_noise, original_noise_vector, verify_matched_inputs,
)
from spt.experiments.energy_windows import (
    build_element_train_medians, common_formal_mask, edge_energy,
    element_train_median_replace, endpoint_linear_inpaint,
    make_bootstrap_plan, make_region_aligned_windows, make_windows,
    paired_cluster_bootstrap, reference_grid, window_metadata,
)


class ProtocolTests(unittest.TestCase):
    def test_cli_turns_experiment_report_into_success_exit_code(self):
        from spt.cli import main
        with patch("spt.experiments.main", return_value={"status": "done"}):
            self.assertEqual(main(["experiment", "bootstrap"]), 0)
        with patch("spt.experiments.main", return_value=2):
            self.assertEqual(main(["experiment", "bootstrap"]), 2)

    def test_dispatcher_turns_report_into_success_exit_code(self):
        from types import SimpleNamespace
        with patch("importlib.import_module", return_value=SimpleNamespace(main=lambda argv: {"done": True})):
            self.assertEqual(experiments_main(["benchmark", "--model-id", "synthetic"]), 0)

    def test_asset_manifest_verification_checks_presence_and_size(self):
        from spt.cli import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            content = b"known asset bytes"
            (root / "asset.bin").write_bytes(content)
            manifest = {"assets": [{"path": "asset.bin", "bytes": len(content)}]}
            (root / "data/asset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch("spt.registry.load_registry", return_value=(root, {"models": []})):
                self.assertEqual(main(["verify-assets", "--all-assets"]), 0)
                manifest["assets"][0]["bytes"] += 1
                (root / "data/asset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "size mismatch"):
                    main(["verify-assets", "--all-assets"])

    def test_dispatcher_help_does_not_need_weights(self):
        self.assertEqual(experiments_main(["--help"]), 0)

    def test_original_gaussian_vector_is_exact_python_rng_protocol(self):
        self.assertEqual(AMPLITUDES, (.05, .10, .15, .20, .25, .30))
        for amplitude in (0.,) + AMPLITUDES:
            rng = random.Random(42)
            expected = np.array([max(-amplitude, min(rng.gauss(0, amplitude / 3), amplitude))
                                 for _ in range(256)], dtype=np.float32)
            np.testing.assert_array_equal(original_noise_vector(amplitude), expected)
            self.assertLessEqual(np.abs(expected).max(), amplitude + 1e-7)

    def test_noise_is_shared_across_rows_and_clean_is_unchanged(self):
        data = np.zeros((3, 256), dtype=np.float32)
        changed, vector = add_original_noise(data, .2)
        np.testing.assert_array_equal(changed, np.broadcast_to(vector, data.shape))
        values = np.arange(768, dtype=np.float32).reshape(3, 256)
        np.testing.assert_array_equal(add_original_noise(values, 0)[0], values)

    def test_invalid_noise_inputs_fail(self):
        for amplitude in (-.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                original_noise_vector(amplitude)
        with self.assertRaises(ValueError):
            add_original_noise(np.zeros((3, 255)), .1)

    def test_matched_gaussian_inputs_compare_arrays_and_order(self):
        cache = {}
        key = ('metal', 'OCN')
        arrays = {'rows': np.array([4, 2]), 'energy': np.ones((2, 256)),
                  'intensity': np.zeros((2, 256)), 'labels': np.array([0., 2.])}
        verify_matched_inputs(cache, key, arrays)
        verify_matched_inputs(cache, key, {name: value.copy() for name, value in arrays.items()})
        for name in arrays:
            changed = {field: value.copy() for field, value in arrays.items()}
            changed[name].flat[0] += 1
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, 'differ'):
                verify_matched_inputs(cache, key, changed)
        arrays['rows'][0] = 9
        self.assertEqual(cache[key]['rows'][0], 4)

    def test_clean_verification_aligns_rows_not_csv_order(self):
        frame = pd.DataFrame({"row_index": [4, 2], "label": [0, 1], "prediction": [0, 1],
                              "probability_0": [.9, .1], "probability_1": [.1, .9]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.csv"
            frame.iloc[::-1].to_csv(path, index=False)
            self.assertTrue(verify_reference(frame, path)["passed"])
            bad = frame.copy()
            bad.loc[0, "prediction"] = 1
            with self.assertRaisesRegex(ValueError, "prediction"):
                verify_reference(bad, path)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                verify_reference(pd.concat([frame, frame]), path)

    def test_outputs_are_never_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(fresh_output(root), root.resolve())
            (root / "keep.txt").write_text("existing output", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                fresh_output(root)

    def test_cluster_bootstrap_matches_explicit_parent_resampling(self):
        frame = pd.DataFrame({"row_index": [4, 0, 1, 3, 2], "label": [5., 0., 1., 3., 2.],
                              "prediction": [4., .5, 1., 2., 2.]})
        data = pd.DataFrame({"material_id": ["a", "a", "b", "c", "c"]})
        reference = frame.assign(prediction=frame.label + .25)
        # A transparent metric keeps this sampling test independent of sklearn.
        mae = lambda _name, y, p: float(np.mean(np.abs(y-p)))
        with patch("spt.experiments.bootstrap.metric_value", side_effect=mae):
            result = cluster_bootstrap(frame, data, reference=reference, metric="mae", iterations=100)
        ordered = frame.sort_values("row_index")
        rng = np.random.default_rng(42)
        groups = np.array(["a", "b", "c"])
        samples, differences = [], []
        for _ in range(100):
            selected = rng.choice(groups, len(groups), replace=True)
            rows = np.concatenate([np.flatnonzero(data.material_id.to_numpy() == g) for g in selected])
            value = mae("mae", ordered.label.to_numpy()[rows], ordered.prediction.to_numpy()[rows])
            samples.append(value)
            differences.append(value - .25)
        self.assertEqual(result["n_parent_structures"], 3)
        self.assertEqual(result["ci_95_low"], np.quantile(samples, .025))
        self.assertEqual(result["paired_difference_ci_95_high"], np.quantile(differences, .975))

    def test_bootstrap_identical_predictions_have_zero_paired_interval(self):
        frame = pd.DataFrame({"row_index": [0, 1, 2, 3], "label": [0., 1., 2., 3.],
                              "prediction": [.2, .9, 2., 2.]})
        data = pd.DataFrame({"material_id": ["a", "a", "b", "c"]})
        with patch("spt.experiments.bootstrap.metric_value",
                   side_effect=lambda _m, y, p: float(np.mean(np.abs(y-p)))):
            result = cluster_bootstrap(frame, data, reference=frame, metric="mae", iterations=30)
        for key in ("paired_difference", "paired_difference_ci_95_low", "paired_difference_ci_95_high"):
            self.assertEqual(result[key], 0.)

    def test_bootstrap_rejects_false_pairs_and_invalid_identity(self):
        frame = pd.DataFrame({"row_index": [0, 1, 2], "label": [0, 1, 0], "prediction": [0, 0, 0]})
        data = pd.DataFrame({"material_id": ["a", "a", "b"]})
        with self.assertRaisesRegex(ValueError, "labels"):
            cluster_bootstrap(frame, data, reference=frame.assign(label=[1, 1, 0]))
        with self.assertRaises(ValueError):
            validate_predictions(frame.assign(row_index=[0, 1, 1]), data, "material_id")
        with self.assertRaises(ValueError):
            validate_predictions(frame.assign(row_index=[0, 1, 9]), data, "material_id")


class EnergyWindowTests(unittest.TestCase):
    def test_reported_region_aligned_grid_has_22_windows(self):
        windows = make_region_aligned_windows(-15, 45, 5, 2.5)
        self.assertEqual(len(windows), 22)
        for window in windows:
            self.assertFalse(window.lo < -2 < window.hi)
            self.assertFalse(window.lo < 15 < window.hi)
        metadata = window_metadata(windows[0])
        self.assertEqual(metadata["overlap_pre_edge"], 1.)
        self.assertEqual(len(make_windows(-15, 45, 5, 2.5)), 23)

    def test_linear_inpaint_changes_only_bracketed_window(self):
        energy = np.arange(11.)
        intensity = energy ** 2
        window = make_windows(-1, 2, 3, 1)[0]
        changed, diagnostic = endpoint_linear_inpaint(energy - 5, intensity, window)
        mask = ((energy-5) >= -1) & ((energy-5) < 2)
        np.testing.assert_array_equal(changed[~mask], intensity[~mask])
        self.assertEqual(diagnostic["fallback"], "none")
        self.assertFalse(np.array_equal(changed[mask], intensity[mask]))

    def test_no_missing_flank_extrapolation(self):
        values = np.arange(10.)
        changed, diagnostic = endpoint_linear_inpaint(values, values**2, make_windows(-2, 3, 5, 1)[0])
        np.testing.assert_array_equal(changed, values**2)
        self.assertEqual(diagnostic["fallback"], "insufficient_bracketing_coverage_noop")

    def test_median_library_is_element_specific_and_window_local(self):
        energies = [np.linspace(edge-20, edge+50, 281) for edge in (100., 101., 500.)]
        intensities = [1/(1+np.exp(-(x-edge)/.7))+.04*np.sin((x-edge)/3)
                       for x, edge in zip(energies, (100., 101., 500.))]
        grid = reference_grid(-20., 50., .25)
        library, diagnostics = build_element_train_medians(energies, intensities, np.array(["A", "A", "B"]),
                                                           np.array(["m1", "m2", "m3"]), grid, 1)
        self.assertEqual(set(library), {"A", "B"})
        self.assertEqual(diagnostics["A"]["n_train_materials"], 2)
        e0, _, _ = edge_energy(energies[0], intensities[0])
        window = make_windows(-5, 5, 10, 1)[0]
        changed, diagnostic = element_train_median_replace(energies[0]-e0, intensities[0], "A", window, grid, library)
        mask = ((energies[0]-e0) >= -5) & ((energies[0]-e0) < 5)
        np.testing.assert_array_equal(changed[~mask], intensities[0][~mask])
        self.assertGreater(diagnostic["changed_points"], 0)

    def test_formal_rows_require_common_coverage_for_both_methods(self):
        a = pd.DataFrame({"has_window_samples": [1, 1, 0, 1], "complete_two_flank_coverage": [1, 0, 0, 1],
                          "fully_reference_supported": [1, 1, 0, 1]}, dtype=bool)
        b = a.copy()
        b.loc[0, "fully_reference_supported"] = False
        np.testing.assert_array_equal(common_formal_mask({"linear": a, "median": b}), [0, 0, 0, 1])

    def test_paired_energy_bootstrap_has_zero_drop_for_same_predictions(self):
        frame = pd.DataFrame({"row_index": np.arange(6), "label": [0, 1, 0, 1, 1, 0],
                              "prediction": [0, 1, 1, 1, 0, 0]})
        plan = make_bootstrap_plan(np.array(["a", "a", "b", "b", "c", "c"]), 30, 42, "test")
        result, drops = paired_cluster_bootstrap(frame, frame.copy(), plan, "Cls", 2)
        self.assertEqual(result["n_groups"], 3)
        np.testing.assert_allclose(drops, 0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for tensor protocol tests")
class TensorProtocolTests(unittest.TestCase):
    def test_band_boundaries_and_dc_preservation(self):
        import torch
        from spt.experiments.bands import band_metadata, apply_rfft_intervention
        self.assertEqual([(b["bin_start"], b["bin_stop_exclusive"]) for b in band_metadata(256, preserve_dc=True)],
                         [(1, 25), (25, 51), (51, 77), (77, 103), (103, 129)])
        x = torch.linspace(0, 1, 256)[None, :]
        for operation in ("mask", "perturb", "phase_randomize", "amplitude_attenuate"):
            changed = apply_rfft_intervention(x, 1, operation, preserve_dc=True)
            torch.testing.assert_close(changed.mean(-1), x.mean(-1))

    def test_interventions_do_not_change_energy_columns(self):
        import torch
        from spt.experiments.bands import intervene_interleaved_input
        x = torch.arange(1024, dtype=torch.float32).reshape(2, 512)
        changed = intervene_interleaved_input(x, 3, "mask", preserve_dc=True)
        torch.testing.assert_close(x[:, 0::2], changed[:, 0::2], rtol=0, atol=0)

    def test_phase_randomization_keeps_fft_amplitudes(self):
        import torch
        from spt.experiments.bands import apply_rfft_intervention
        x = torch.randn(3, 256, generator=torch.Generator().manual_seed(7))
        y = apply_rfft_intervention(x, 5, "phase_randomize", preserve_dc=True)
        torch.testing.assert_close(torch.fft.rfft(x).abs(), torch.fft.rfft(y).abs(), rtol=1e-4, atol=1e-5)

    def test_identity_roundtrip_has_zero_error(self):
        import torch
        from spt.experiments.roundtrip import transform_errors
        class Identity:
            def analyze(self, value): return value
            def synthesize(self, value): return value
        absolute, relative = transform_errors(Identity(), torch.ones(2, 4, 8))
        torch.testing.assert_close(absolute, torch.zeros(2))
        torch.testing.assert_close(relative, torch.zeros(2))

    def test_mc_dropout_keeps_batch_normalization_in_eval(self):
        import torch
        from spt.experiments.uncertainty import enable_dropout_only
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Dropout(.1), torch.nn.Linear(3, 2))
        enable_dropout_only(model)
        self.assertFalse(model[0].training)
        self.assertTrue(model[1].training)
        self.assertFalse(model[2].training)

    def test_runtime_counts_warmups_and_synchronizes_repetitions(self):
        from spt.experiments.runtime import time_calls
        calls = []
        with patch("spt.experiments.runtime.synchronize") as sync, patch(
                "spt.experiments.runtime.time.perf_counter", side_effect=[0, 2, 2, 6, 6, 12]):
            values = time_calls(lambda: calls.append(1), repeats=3, calls_per_repeat=2, warmup=3)
        self.assertEqual(len(calls), 9)
        self.assertEqual(sync.call_count, 7)
        self.assertEqual(values, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
