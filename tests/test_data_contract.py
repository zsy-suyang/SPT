"""Synthetic input-contract tests; these are not scientific benchmark data."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from spt.utils.data_load import Dataset
from spt.utils.data_load.data_des import parse_column_to_numpy, X_log_normalize_optimized


class VectorTests(unittest.TestCase):
    def test_json_and_numpy_forms_agree(self):
        actual = parse_column_to_numpy(["[1,2,3]", "[1 2 3]", "[1,\n2, 3]"])
        for vector in actual:
            np.testing.assert_array_equal(vector, np.array([1, 2, 3], dtype=np.float32))

    def test_legacy_frozen_vector_conversion_is_unchanged(self):
        text = "[1.23456789 2.34567891 3.45678912]"
        old = np.array([float(v) for v in text.replace("[", "").replace("]", "").split(" ")], dtype=np.float32)
        np.testing.assert_array_equal(parse_column_to_numpy([text])[0], old)

    def test_invalid_values_are_rejected(self):
        for text in ("[]", "[1,nan]", "[1,inf]", "[1,,2]", "[[1,2]]", "[1 2", "oops"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_column_to_numpy([text])

    def test_nonpositive_energy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            X_log_normalize_optimized([np.array([0.0, 1.0])], 0.0, 2.0)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.csv = Path(self.temp.name) / "synthetic.csv"
        self.frame = pd.DataFrame({
            "material_id": ["synthetic-only"],
            "center_atom_type": ["Co"],
            "XAS_Relative_energy": ["[10 20 30 40]"],
            "XAS_Intensity": ["[0 0.5 1 0.7]"],
            "OS": [2],
        })
        self.args = SimpleNamespace(
            data_file=str(self.csv), num_cont=8, min_log_data=0.0,
            max_log_data=5.0, add_noise=None, ele_dict={"Co": 0}, aux=1,
            multi_task=None, task="OS", min_class_samples=1, min_class_groups=1,
            OS={"idx": 0, "mode": "Cls", "value": [2], "label": [0]},
        )

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self):
        self.frame.to_csv(self.csv)
        return Dataset(dataset="infer", args=self.args)

    def test_none_noise_and_legacy_normalization(self):
        data = self.dataset()
        energy = (np.log(np.array([10, 20, 30, 40], dtype=np.float32)) - 0.0) / 5.0
        intensity = np.array([0, 0.5, 1, 0.7], dtype=np.float32)
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-10)
        expected = np.empty(8, dtype=np.float32)
        expected[0::2] = energy
        expected[1::2] = intensity
        np.testing.assert_array_equal(data.cont[0], expected)

    def test_unknown_element_is_rejected(self):
        self.frame.loc[0, "center_atom_type"] = "Unknown"
        with self.assertRaisesRegex(ValueError, "Unknown absorber"):
            self.dataset()

    def test_mismatched_lengths_are_rejected(self):
        self.frame.loc[0, "XAS_Intensity"] = "[0 1]"
        with self.assertRaisesRegex(ValueError, "No truncation"):
            self.dataset()

    def test_nonpositive_energy_is_rejected(self):
        self.frame.loc[0, "XAS_Relative_energy"] = "[-1 0 1 2]"
        with self.assertRaisesRegex(ValueError, "non-positive"):
            self.dataset()


if __name__ == "__main__":
    unittest.main()
