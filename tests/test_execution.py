"""Publication replay selects its recorded device before model evaluation."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from spt.utils.execution import inference_backend


def test_primary_cuda_reference_rejects_cpu_before_loading():
    args = SimpleNamespace(reference_device="cuda", inference_backend={
        "cuda_matmul_allow_tf32": True, "cudnn_allow_tf32": True})
    before = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    with pytest.raises(ValueError, match="publication-reference replay requires CUDA"):
        with inference_backend(args, "cpu"):
            pytest.fail("Mismatched reference device entered inference")
    assert before == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)


def test_primary_cuda_reference_reports_missing_cuda():
    with patch("torch.cuda.is_available", return_value=False), pytest.raises(RuntimeError, match="CUDA-enabled"):
        with inference_backend(SimpleNamespace(reference_device="cuda"), "cuda:0"):
            pytest.fail("Unavailable device entered inference")


def test_no_reference_device_keeps_generic_cpu_inference():
    with inference_backend(SimpleNamespace(), "cpu") as metadata:
        assert metadata["reference_device"] is None
        assert not metadata["cuda_flags_applicable"]
