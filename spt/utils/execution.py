"""Apply the numerical execution settings recorded for a model or cohort."""
from contextlib import contextmanager
import os


# Absolute replay tolerance in angstroms, not an experimental uncertainty.
NI_GEOMETRY_ATOL = 1e-5


@contextmanager
def inference_backend(args, device):
    """Use explicit backend settings, restoring caller settings on exit."""
    import torch
    device = torch.device(device)
    required = getattr(args, "reference_device", None)
    if required not in (None, "cpu", "cuda"):
        raise ValueError("reference_device must be cpu or cuda")
    if required is not None and device.type != required:
        raise ValueError(
            f"This publication-reference replay requires {required.upper()} inference. "
            f"Use --device {'cuda:0' if required == 'cuda' else 'cpu'} with the recorded environment. "
            "For general inference on another device, use spt predict or spt evaluate."
        )
    if required == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Publication-reference replay requires a CUDA-enabled PyTorch build and GPU.")
    configured = getattr(args, "inference_backend", None)
    targets = {"cuda_matmul_allow_tf32": torch.backends.cuda.matmul,
               "cudnn_allow_tf32": torch.backends.cudnn}
    if configured is not None:
        if not isinstance(configured, dict) or set(configured) - set(targets):
            raise ValueError("inference_backend must map supported CUDA TF32 flags to booleans")
        if any(type(value) is not bool for value in configured.values()):
            raise ValueError("inference_backend flag values must be booleans")
    configured = configured or {}
    previous = {key: targets[key].allow_tf32 for key in configured}
    precision = torch.get_float32_matmul_precision() if "cuda_matmul_allow_tf32" in configured else None
    try:
        for key, value in configured.items():
            targets[key].allow_tf32 = value
        yield {
            "configured": dict(configured),
            "reference_device": required,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_flags_applicable": device.type == "cuda",
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "tf32_environment": {key: os.environ.get(key) for key in (
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE")},
        }
    finally:
        for key, value in previous.items():
            targets[key].allow_tf32 = value
        if precision is not None:
            torch.set_float32_matmul_precision(precision)
