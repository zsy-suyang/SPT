"""CUDA model-compute latency and update timing using frozen registered checkpoints."""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import io
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from spt.models import build_model
from spt.training.trainer_delta import ModelEMA
from .common import fresh_output, load_config, load_dataset, registry_job, save_json, set_seed

def quiet_config(path):
    with contextlib.redirect_stdout(io.StringIO()):
        return load_config(path)


def synchronize():
    torch.cuda.synchronize()


def summarize(samples, batch):
    return {
        "raw_seconds_per_call": samples,
        "median_seconds_per_call": statistics.median(samples),
        "mean_seconds_per_call": statistics.mean(samples),
        "sd_seconds_per_call": statistics.stdev(samples),
        "minimum_seconds_per_call": min(samples),
        "maximum_seconds_per_call": max(samples),
        "median_spectra_per_second": batch / statistics.median(samples),
        "batch_size": batch,
    }


def time_calls(call, repeats, calls_per_repeat, warmup):
    for _ in range(warmup):
        call()
    synchronize()
    samples = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        for _ in range(calls_per_repeat):
            call()
        synchronize()
        samples.append((time.perf_counter() - start) / calls_per_repeat)
    return samples


def tensor_labels(dataset, args, selected, tasks):
    result = {}
    for task in tasks:
        dtype = torch.long if getattr(args, task)["mode"] == "Cls" else torch.float32
        result[task] = torch.as_tensor(dataset.label[task][selected], dtype=dtype, device="cuda")
    return result if len(tasks) > 1 else result[tasks[0]]


def optimizer(model, args):
    no_decay = ("bias", "LayerNorm.bias", "LayerNorm.weight", "norm.bias", "norm.weight")
    decay, other = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim > 1 and not any(x in name for x in no_decay) else other).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(args.weight_decay)},
         {"params": other, "weight_decay": 0.0}],
        lr=float(args.lr), betas=(0.9, 0.999), eps=1e-8,
    )


def benchmark_entry(entry, datasets, cli):
    path = Path(entry["config"])
    args = quiet_config(path)
    dataset = datasets[entry["site"]]
    tasks = args.multi_task if isinstance(args.multi_task, list) else [args.task]
    reference_args = dataset.args
    for key in ("num_cont", "min_log_data", "max_log_data", "min_class_samples", "min_class_groups"):
        assert getattr(args, key, None) == getattr(reference_args, key, None), key
    for task in tasks:
        keys = ("value", "label", "mode", "idx") if getattr(args, task)["mode"] == "Cls" else ("mode", "idx")
        for key in keys:
            assert getattr(args, task)[key] == getattr(reference_args, task)[key], (task, key)

    if entry["kind"] == "Expert":
        valid = dataset.task_valid_masks[tasks[0]]
    else:
        valid = np.ones(len(dataset), dtype=bool)
    eligible = np.flatnonzero(valid)
    assert len(eligible) >= 128
    train_batch = int(args.batch_size)
    selected = eligible[:max(train_batch, 128)]
    cate = torch.as_tensor(dataset.cate[selected], dtype=torch.long, device="cuda")
    cont = torch.as_tensor(dataset.cont[selected], dtype=torch.float32, device="cuda")
    checkpoint = Path(entry["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu")
    model = build_model(args, torch.device("cuda")).to("cuda")
    model.load_state_dict(payload["net_state_dict"], strict=True)
    del payload
    model.eval()
    info = {
        **entry,
        "task_names": ["SGN" if task == "PSGO" else task for task in tasks],
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "eligible_training_rows": int(len(eligible)),
        "training_batch_size": train_batch,
        "source_training_rows": dataset.indices[selected].tolist(),
        "input_shape": [128, int(cont.shape[1])], "inference": {},
    }
    assert not bool(args.use_amp), "Audit currently requires source full-precision configurations."
    assert not float(getattr(args, "mixup_alpha", 0.0)), "Mixup needs explicit timing support."

    with torch.inference_mode():
        for batch in (1, 128):
            torch.cuda.reset_peak_memory_stats()
            samples = time_calls(
                lambda: model(cate[:batch], cont[:batch], None),
                cli.repeats, cli.inference_calls, cli.warmup,
            )
            info["inference"][str(batch)] = summarize(samples, batch)
            info["inference"][str(batch)]["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / (1024**2)

    labels = tensor_labels(dataset, args, eligible[:train_batch], tasks)
    optim = optimizer(model, args)
    ema_decay = float(getattr(args, "ema_decay", 0.0) or 0.0)
    ema = ModelEMA(model, ema_decay) if ema_decay else None
    model.train()
    assert len(cate) >= train_batch
    step_number = 2000

    def train_step():
        nonlocal step_number
        model.zero_grad()
        optim.zero_grad()
        _, loss = model(cate[:train_batch], cont[:train_batch], labels, step=step_number)
        loss.backward()
        optim.step()
        if ema is not None:
            ema.update(model)
        step_number += 1

    torch.cuda.reset_peak_memory_stats()
    samples = time_calls(train_step, cli.repeats, cli.training_calls, cli.warmup)
    info["training_step"] = summarize(samples, train_batch)
    info["training_step"]["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / (1024**2)
    info["training_step"]["ema_decay"] = ema_decay
    drop_last = bool(getattr(args, "drop_last", True))
    n_steps = len(eligible) // train_batch if drop_last else math.ceil(len(eligible) / train_batch)
    info["epoch_estimate"] = {
        "steps_per_epoch": n_steps, "drop_last": drop_last,
        "estimated_training_compute_seconds_per_epoch": n_steps * statistics.median(samples),
        "excludes": ["data reading and preprocessing", "data loader iteration and host-to-device transfers", "validation", "metric aggregation", "checkpoint writing", "learning-rate scheduling"],
        "not_original_training_wall_time": True,
    }
    del labels, optim, ema, model, cate, cont
    gc.collect()
    torch.cuda.empty_cache()
    return info


def benchmark_bundle(site, entries, dataset, cli):
    """Measure all descriptors from sequential Experts versus one Joint call."""
    tasks = dataset.args.multi_task
    valid = np.ones(len(dataset), dtype=bool)
    for task in tasks:
        valid &= dataset.task_valid_masks[task]
    chosen = np.flatnonzero(valid)[:128]
    assert len(chosen) == 128
    cate = torch.as_tensor(dataset.cate[chosen], dtype=torch.long, device="cuda")
    cont = torch.as_tensor(dataset.cont[chosen], dtype=torch.float32, device="cuda")
    info = {
        "site": site, "task_names": ["SGN" if task == "PSGO" else task for task in tasks],
        "source_training_rows": dataset.indices[chosen].tolist(),
        "same_inputs_for_experts_and_joint": True, "variants": {},
    }
    for kind in ("Expert", "Joint"):
        selected = [entry for entry in entries if entry["site"] == site and entry["kind"] == kind]
        models = []
        for entry in selected:
            args = quiet_config(Path(entry["config"]))
            model = build_model(args, torch.device("cuda")).to("cuda")
            payload = torch.load(entry["checkpoint"], map_location="cpu")
            model.load_state_dict(payload["net_state_dict"], strict=True)
            del payload
            model.eval()
            models.append(model)

        def all_heads(batch):
            return [model(cate[:batch], cont[:batch], None)[0] for model in models]

        variant = {"number_of_forward_calls_per_batch": len(models), "inference": {}}
        with torch.inference_mode():
            for batch in (1, 128):
                samples = time_calls(lambda: all_heads(batch), cli.repeats, cli.inference_calls, cli.warmup)
                variant["inference"][str(batch)] = summarize(samples, batch)
                print(f"BUNDLE {site} {kind} batch {batch}: {statistics.median(samples)*1000:.3f} ms", flush=True)
        info["variants"][kind] = variant
        del model, models, variant
        gc.collect()
        torch.cuda.empty_cache()
    info["joint_speedup"] = {
        str(batch): info["variants"]["Expert"]["inference"][str(batch)]["median_seconds_per_call"] /
        info["variants"]["Joint"]["inference"][str(batch)]["median_seconds_per_call"]
        for batch in (1, 128)
    }
    return info



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Checkout containing configs/registry.json")
    parser.add_argument("--site", nargs="+", choices=("metal", "oxygen"), default=["metal", "oxygen"])
    parser.add_argument("--model-id", nargs="+", help="Optional subset for individual timings")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--inference-calls", type=int, default=3)
    parser.add_argument("--training-calls", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measure-bundles", action="store_true")
    cli = parser.parse_args(argv)
    if cli.repeats < 3 or min(cli.inference_calls, cli.training_calls, cli.warmup) < 1:
        parser.error("At least three repeats and positive call/warm-up counts are required")
    if not torch.cuda.is_available():
        raise RuntimeError("This protocol requires a CUDA GPU; CPU timings are not paper-comparable")
    from spt.registry import load_registry
    root, registry = load_registry(cli.root)
    experiment_registry = json.loads((root / "configs/experiments.json").read_text(encoding="utf-8"))
    runtime_ids = set(experiment_registry["experiments"]["runtime"]["model_ids"])
    candidates = [e for e in registry["models"] if e["site"] in cli.site and e["id"] in runtime_ids]
    if cli.model_id:
        missing = set(cli.model_id) - {e["id"] for e in candidates}
        if missing:
            raise ValueError(f"Non-Expert/Joint or unknown timing IDs: {sorted(missing)}")
        selected = [e for e in candidates if e["id"] in cli.model_id]
    else:
        selected = candidates
    if not selected:
        raise ValueError("No timing models selected")
    set_seed()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    entries, datasets = [], {}
    for item in selected:
        job = registry_job(item["id"], root)
        from spt.registry import verify_asset
        verify_asset(job["checkpoint"], job.get("checkpoint_bytes"))
        entries.append({"id": job["id"], "kind": "Joint" if job["kind"] == "joint" else "Expert",
                        "site": job["site"], "task": job["task"],
                        "config": job["config"], "checkpoint": job["checkpoint"]})
    for site in sorted({entry["site"] for entry in entries}):
        matches = [entry for entry in candidates if entry["site"] == site and entry["kind"] == "joint"]
        if len(matches) != 1:
            raise ValueError(f"One Joint configuration is required to define shared training masks for {site}")
        joint = registry_job(matches[0]["id"], root)
        args = load_config(joint["config"])
        datasets[site] = load_dataset(args, "train")
        if cli.measure_bundles:
            actual = {e["task"] for e in entries if e["site"] == site and e["kind"] == "Expert"}
            if actual != set(args.multi_task) or sum(e["kind"] == "Joint" and e["site"] == site for e in entries) != 1:
                raise ValueError("Bundle timing requires all Expert heads and the Joint model for each site")
    output = fresh_output(cli.output_dir)
    protocol = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0), "python": sys.version,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "platform": platform.platform(),
        "dtype": "float32", "amp": False, "seed": 42,
        "host_threads": torch.get_num_threads(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "repeats": cli.repeats, "warmup_calls": cli.warmup,
        "inference_calls_per_repeat": cli.inference_calls,
        "training_calls_per_repeat": cli.training_calls,
        "inputs": "first eligible frozen training rows, resident on GPU",
        "inference_scope": "encoder and prediction heads; no preparation, transfer or serialization",
        "training_scope": "source-model loss, backward, AdamW and EMA; no scheduler or validation",
        "synchronization": "CUDA synchronization before and after every timed repeat",
        "training_checkpoint_saved": False, "measured_full_epoch": False,
    }
    result = {"protocol": protocol, "models": []}
    for entry in entries:
        result["models"].append(benchmark_entry(entry, datasets, cli))
        save_json(output / "runtime_results.json", result)
    if cli.measure_bundles:
        result["bundles"] = [benchmark_bundle(site, entries, dataset, cli) for site, dataset in datasets.items()]
    save_json(output / "runtime_results.json", result)
    rows = []
    for item in result["models"]:
        for batch, measured in item["inference"].items():
            rows.append({"id": item["id"], "site": item["site"], "task": item["task"],
                         "batch_size": int(batch), "median_ms": 1000*measured["median_seconds_per_call"],
                         "spectra_per_second": measured["median_spectra_per_second"],
                         "training_step_ms": 1000*item["training_step"]["median_seconds_per_call"],
                         "estimated_epoch_compute_s": item["epoch_estimate"]["estimated_training_compute_seconds_per_epoch"]})
    import pandas as pd
    pd.DataFrame(rows).to_csv(output / "runtime_summary.csv", index=False)

