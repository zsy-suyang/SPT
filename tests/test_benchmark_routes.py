"""Exercise frozen tree dispatch with real fitted estimators, without release assets."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from spt.experiments import benchmark


class InferenceBackendTests(unittest.TestCase):
    @staticmethod
    def flags():
        import torch
        return (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)

    def test_boolean_values_required_before_changes(self):
        import torch
        original = self.flags()
        for invalid in (1, 0, "true", "false", 0.0, None):
            for key in ("cuda_matmul_allow_tf32", "cudnn_allow_tf32"):
                with self.subTest(key=key, invalid=invalid), self.assertRaisesRegex(ValueError, "booleans"):
                    with benchmark.inference_backend(
                            SimpleNamespace(inference_backend={key: invalid}), torch.device("cpu")):
                        self.fail("Invalid configuration entered the inference context")
                self.assertEqual(self.flags(), original)
        for invalid in (True, [], {"unknown": True}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                with benchmark.inference_backend(
                        SimpleNamespace(inference_backend=invalid), torch.device("cpu")):
                    self.fail("Invalid mapping entered the inference context")
            self.assertEqual(self.flags(), original)

    def test_configured_flags_restored_and_cpu_metadata_qualified(self):
        import torch
        original = self.flags()
        requested = {"cuda_matmul_allow_tf32": not original[0],
                     "cudnn_allow_tf32": not original[1]}
        with patch.dict("os.environ", {"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
                                       "NVIDIA_TF32_OVERRIDE": "0",
                                       "PRIVATE_TEST_VALUE": "not recorded"}):
            with benchmark.inference_backend(SimpleNamespace(inference_backend=requested),
                                              torch.device("cpu")) as metadata:
                self.assertEqual(self.flags(), (not original[0], not original[1]))
                self.assertEqual(metadata["configured"], requested)
                self.assertFalse(metadata["cuda_flags_applicable"])
                self.assertEqual(metadata["torch_version"], str(torch.__version__))
                self.assertEqual(metadata["torch_cuda_version"], torch.version.cuda)
                self.assertEqual(metadata["tf32_environment"], {
                    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1", "NVIDIA_TF32_OVERRIDE": "0"})
        self.assertEqual(self.flags(), original)

    def test_restoration_after_exception(self):
        import torch
        original = self.flags()
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            with benchmark.inference_backend(SimpleNamespace(inference_backend={
                    "cuda_matmul_allow_tf32": not original[0],
                    "cudnn_allow_tf32": not original[1]}), torch.device("cpu")):
                raise RuntimeError("synthetic failure")
        self.assertEqual(self.flags(), original)

    def test_unconfigured_context_does_not_change_defaults(self):
        import torch
        original = self.flags()
        with benchmark.inference_backend(SimpleNamespace(), torch.device("cpu")) as metadata:
            self.assertEqual(self.flags(), original)
            self.assertEqual(metadata["configured"], {})
        self.assertEqual(self.flags(), original)

    def test_matmul_precision_restored_with_flags(self):
        import torch
        original = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("medium")
            with benchmark.inference_backend(SimpleNamespace(inference_backend={
                    "cuda_matmul_allow_tf32": False}), torch.device("cpu")):
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertEqual(torch.get_float32_matmul_precision(), "medium")
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        finally:
            torch.set_float32_matmul_precision(original)

    def test_context_covers_model_construction_and_prediction(self):
        import torch
        original = self.flags()
        requested = {"cuda_matmul_allow_tf32": not original[0],
                     "cudnn_allow_tf32": not original[1]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, config, reference = (root / name for name in ("model.pth", "model.yaml", "reference.csv"))
            checkpoint.write_bytes(b"synthetic checkpoint")
            config.write_text("{}", encoding="utf-8")
            frame = pd.DataFrame({"row_index": [0, 1], "label": [0, 1], "prediction": [0, 1]})
            frame.to_csv(reference, index=False)
            job = {"id": "synthetic-neural", "site": "metal", "config": config,
                   "checkpoint": checkpoint, "checkpoint_bytes": checkpoint.stat().st_size,
                   "reference_predictions": reference}
            args = SimpleNamespace(architecture="spt", task="target", target={"mode": "Cls"},
                                   data_file="synthetic", split_file="synthetic", inference_backend=requested)

            def construct(*unused):
                self.assertEqual(self.flags(), (not original[0], not original[1]))
                return object()

            def prediction(*unused):
                self.assertEqual(self.flags(), (not original[0], not original[1]))
                return {"Acc": 1.0, "F1": 1.0}, frame

            with patch.object(benchmark, "registry_job", return_value=job), \
                 patch.object(benchmark, "load_config", return_value=args), \
                 patch.object(benchmark, "load_dataset", return_value=object()), \
                 patch.object(benchmark, "load_model", side_effect=construct), \
                 patch.object(benchmark, "predict", side_effect=prediction):
                self.assertEqual(benchmark.main(["--model-id", job["id"], "--device", "cpu", "--output-dir", str(root / "output")]), 0)
            self.assertEqual(self.flags(), original)
            metadata = json.loads((root / "output/synthetic-neural/metadata.json").read_text())
            self.assertEqual(metadata["inference_backend"]["configured"], requested)
            self.assertFalse(metadata["inference_backend"]["cuda_flags_applicable"])


class JointMetricTests(unittest.TestCase):
    def test_original_float32_classification_metric(self):
        import torch
        from torchmetrics import F1Score
        logits = torch.tensor([[2., 1.], [3., 1.], [0., 2.]], dtype=torch.float32)
        labels = np.array([0, 1, 1])
        result = benchmark.joint_metric_values("Cls", logits, labels, torch.device("cpu"))
        expected = F1Score(task="multiclass", num_classes=2, average="weighted")(logits, torch.tensor(labels)).item()
        self.assertEqual(result["F1"], expected)

    def test_joint_regression_keeps_separate_float64_audit(self):
        import torch
        from torchmetrics import R2Score

        class Dataset(torch.utils.data.Dataset):
            args = SimpleNamespace(multi_task=["geometry"], geometry={"mode": "Reg"})
            indices = np.array([1, 2, 3, 4])
            task_valid_masks = {"geometry": np.array([True, True, True, False])}
            target = torch.tensor([2.70, 2.71, 2.72, float("nan")], dtype=torch.float32)
            prediction = torch.tensor([2.702, 2.708, 2.723, 2.70], dtype=torch.float32)

            def __len__(self):
                return 4

            def __getitem__(self, index):
                return torch.tensor([0]), self.prediction[index:index+1], {"geometry": self.target[index]}

        def model(cate, cont, truth):
            return {"geometry": cont}, None

        scores, frame = benchmark.joint_predictions(model, Dataset(), torch.device("cpu"), 2)["geometry"]
        expected = R2Score()(Dataset.prediction[:3], Dataset.target[:3]).item()
        audit = benchmark.metric_values("Reg", frame.label.to_numpy(dtype=np.float64),
                                       frame.prediction.to_numpy(dtype=np.float64))
        self.assertEqual(scores["R2"], expected)
        self.assertEqual(scores["audit_float64_R2"], audit["R2"])
        self.assertEqual(frame.row_index.tolist(), [1, 2, 3])


class FrozenTreeRouteTests(unittest.TestCase):
    def test_registered_tree_routes_classification_and_regression(self):
        features = np.array([[0, 0.1], [0, 0.2], [1, 0.8], [1, 0.9]], dtype=np.float32)
        for architecture, estimator_class, mode in (
            ("decision_tree", DecisionTreeClassifier, "Cls"),
            ("decision_tree", DecisionTreeRegressor, "Reg"),
            ("extra_trees", ExtraTreesClassifier, "Cls"),
            ("extra_trees", ExtraTreesRegressor, "Reg"),
        ):
            with self.subTest(architecture=architecture, mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                labels = np.array([0, 0, 1, 1]) if mode == "Cls" else np.array([0.1, 0.2, 0.8, 0.9])
                estimator = estimator_class(random_state=42)
                if architecture == "extra_trees":
                    estimator.set_params(n_estimators=3, n_jobs=1)
                estimator.fit(features, labels)
                checkpoint = root / "frozen.joblib"
                joblib.dump(estimator, checkpoint)
                config = root / "config.json"
                config.write_text("{}", encoding="utf-8")
                frame = pd.DataFrame({"row_index": [9, 2, 7, 4], "label": labels,
                                      "prediction": estimator.predict(features)})
                reference = root / "reference.csv"
                frame.to_csv(reference, index=False)
                args = SimpleNamespace(architecture=architecture, task="target", target={"mode": mode},
                                       data_file="synthetic", split_file="synthetic")
                dataset = SimpleNamespace(cate=features[:, :1], cont=features[:, 1:],
                                          indices=frame.row_index.to_numpy(), label=labels)
                job = {"id": "synthetic-tree", "site": "metal", "config": config,
                       "checkpoint": checkpoint, "checkpoint_bytes": checkpoint.stat().st_size,
                       "reference_predictions": reference}
                with patch.object(benchmark, "registry_job", return_value=job), \
                     patch.object(benchmark, "load_config", return_value=args), \
                     patch.object(benchmark, "load_dataset", return_value=dataset), \
                     patch.object(benchmark, "load_model", side_effect=AssertionError("Tree entered neural route")):
                    self.assertEqual(benchmark.main(["--model-id", job["id"], "--device", "cpu", "--output-dir", str(root / "output")]), 0)
                result = pd.read_csv(root / "output/benchmark_results.csv")
                self.assertTrue(result.reference_verified.all())
                actual = pd.read_csv(root / "output/synthetic-tree/target_predictions.csv")
                np.testing.assert_allclose(actual.prediction, frame.prediction, rtol=1e-14, atol=1e-14)
                metadata = json.loads((root / "output/synthetic-tree/metadata.json").read_text(encoding="utf-8"))
                self.assertTrue(metadata["checks"]["target"]["passed"])


if __name__ == "__main__":
    unittest.main()
