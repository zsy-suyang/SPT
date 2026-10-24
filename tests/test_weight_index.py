"""Keep the application weight table aligned with executable configurations."""

import csv
import filecmp
import json
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class ApplicationWeightIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        index = ROOT / "checkpoints/WEIGHT_INDEX.csv"
        if not index.is_file():
            raise unittest.SkipTest("Install the checkpoint index to check asset mappings")
        with index.open(encoding="utf-8-sig", newline="") as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.app_rows = [r for r in cls.rows if r["model_id"].startswith("app-")]
        cls.registry = {m["id"]: m for m in json.loads((ROOT / "configs/registry.json").read_text())["models"]}

    def test_complete_application_mapping(self):
        applications = json.loads((ROOT / "configs/applications.json").read_text())
        expected = {model for cohort in applications["cohorts"] for model in cohort["models"].values()}
        self.assertEqual({r["model_id"] for r in self.app_rows}, expected)
        self.assertEqual(len(self.app_rows), len(expected))
        for row in self.app_rows:
            with self.subTest(model=row["model_id"]):
                model = self.registry[row["model_id"]]
                self.assertEqual(row["path"], model["checkpoint"])
                self.assertEqual(row["configuration"], model["config"])
                self.assertTrue(row["application_model_group"])
                self.assertTrue((ROOT / row["training_configuration"]).is_file())
                self.assertTrue((ROOT / row["training_partition"]).is_file())

    def test_training_settings_and_selection(self):
        for row in self.app_rows:
            with self.subTest(model=row["model_id"]):
                config = yaml.safe_load((ROOT / row["training_configuration"]).read_text(encoding="utf-8"))
                self.assertEqual(config["split_file"], row["training_partition"])
                self.assertEqual(config["data_file"], "datasets/computed/metal.csv")
                self.assertEqual(config["validation_set"], "val")
                self.assertEqual(config["seed"], config["split_seed"])
                self.assertIn(config["selection_metric"], ("f1", "r2"))
                if config["selection_metric"] == "f1":
                    self.assertEqual(config["f1_average"], "weighted")

    def test_group_metadata_agree(self):
        groups = {}
        fields = ("training_configuration", "training_partition")
        for row in self.rows:
            group = row["application_model_group"]
            if group:
                values = tuple(row[field] for field in fields)
                self.assertEqual(groups.setdefault(group, values), values, group)

    def test_application_training_configs_preserve_label_types(self):
        for task in ("OS", "OCN"):
            config = yaml.safe_load((ROOT / f"configs/models/application-metal-{task}.yaml").read_text(encoding="utf-8"))
            for descriptor in ("OS", "CN", "OCN", "CN2", "PSGO"):
                self.assertTrue(all(isinstance(label, int) for label in config[descriptor]["cnts"]))
            self.assertIsNone(config["ckpt"])
            self.assertEqual(config["architecture"], "spt")

    def test_shared_files_are_identical_when_installed(self):
        mapped = [r for r in self.rows if r["application_model_group"]]
        if not all((ROOT / r["path"]).is_file() for r in mapped):
            self.skipTest("Install application and Expert checkpoints for content checks")
        representatives = {}
        for row in mapped:
            path = ROOT / row["path"]
            representative = representatives.setdefault(row["application_model_group"], path)
            self.assertTrue(filecmp.cmp(path, representative, shallow=False), row["model_id"])


if __name__ == "__main__":
    unittest.main()
