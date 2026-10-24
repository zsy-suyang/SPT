"""Runtime selection shared by the training and evaluation entry points."""

import os

import torch


def resolve_device(args):
    requested = str(getattr(args, "device", "auto") or "auto").lower()
    if requested == "auto":
        gpu_id = getattr(args, "gpu_id", None)
        if gpu_id is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are auto, cpu, cuda, and cuda:<index>")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    return device
