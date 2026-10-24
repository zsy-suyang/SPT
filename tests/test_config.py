"""Portable config loading, with no model weights or datasets required."""

from pathlib import Path
import tempfile
import unittest

from spt.config.cfg import load_config, parse_config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "pyproject.toml").write_text("[project]\nname = 'config-test'\n", encoding="utf-8")
        (self.root / "configs" / "models").mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_inheritance_and_project_paths(self):
        self.write("configs/base.yaml", "dim: 96\nbatch_size: 128\neval_batch_size: 128\n")
        path = self.write(
            "configs/models/test.yaml",
            "__base__: [../base.yaml]\nbatch_size: 64\n"
            "data_file: datasets/metal.csv\nsplit_file: data/splits/metal.json\n"
            "save_folder: runs/expert\nckpt: None\nadd_noise: null\n",
        )
        args = load_config(path)
        self.assertEqual(args.dim, 96)
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.eval_batch_size, 128)
        self.assertEqual(args.project_root, str(self.root))
        self.assertEqual(args.data_file, str(self.root / "datasets" / "metal.csv"))
        self.assertEqual(args.save_folder, str(self.root / "runs" / "expert"))
        self.assertIsNone(args.ckpt)
        self.assertIsNone(args.add_noise)
        self.assertFalse(hasattr(args, "__base__"))

    def test_project_relative_base_and_explicit_root(self):
        self.write("config/base.yaml", "dim: 96\n")
        path = self.write("configs/models/test.yaml", "__base__: [config/base.yaml]\ndata_file: data.csv\n")
        args = load_config(path, project_root=self.root)
        self.assertEqual(args.dim, 96)
        self.assertEqual(args.data_file, str(self.root / "data.csv"))

    def test_base_cycle_is_rejected(self):
        a = self.write("configs/a.yaml", "__base__: [b.yaml]\n")
        self.write("configs/b.yaml", "__base__: [a.yaml]\n")
        with self.assertRaisesRegex(ValueError, "Cyclic"):
            load_config(a)

    def test_missing_base_is_rejected(self):
        path = self.write("configs/a.yaml", "__base__: [missing.yaml]\n")
        with self.assertRaises(FileNotFoundError):
            load_config(path)

    def test_invalid_yaml_mapping_is_rejected(self):
        path = self.write("configs/a.yaml", "- not\n- a mapping\n")
        with self.assertRaisesRegex(ValueError, "mapping"):
            load_config(path)

    def test_overrides_and_default_evaluation_split(self):
        path = self.write(
            "configs/models/test.yaml",
            "ckpt: checkpoints/model.pth\neval_batch_size: 128\ndataset_mode: dev\n",
        )
        args = parse_config(["--config", str(path)], evaluation=True)
        self.assertEqual(args.dataset_mode, "test")
        self.assertEqual(args.test_set, "test")
        self.assertEqual(args.eval_batch_size, 128)
        self.assertEqual(args.ckpt, str(self.root / "checkpoints" / "model.pth"))
        fresh = load_config(path, overrides={"ckpt": None})
        self.assertIsNone(fresh.ckpt)

    def test_evaluation_requires_checkpoint(self):
        path = self.write("configs/models/test.yaml", "ckpt: null\n")
        with self.assertRaisesRegex(ValueError, "requires"):
            parse_config(["--config", str(path)], evaluation=True)


if __name__ == "__main__":
    unittest.main()
