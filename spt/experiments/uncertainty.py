"""Estimate predictive uncertainty with Monte Carlo dropout on a fixed checkpoint."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from .common import (load_config, load_dataset, load_model, fresh_output,
                     add_model_arguments, resolve_model_arguments)



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser, reference=False)
    parser.add_argument("--passes", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return resolve_model_arguments(parser.parse_args(argv), parser, reference=False)


def enable_dropout_only(model):
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def summarize_task(logits, labels, row_indices, mode):
    rows = []
    if mode == "Reg":
        values = logits.squeeze(-1)
        means = values.mean(dim=0)
        stds = values.std(dim=0, unbiased=False)
        for row_index, label, mean, std in zip(
            row_indices, labels, means, stds
        ):
            rows.append(
                {
                    "row_index": int(row_index),
                    "label": float(label),
                    "prediction": float(mean),
                    "uncertainty": float(std),
                    "absolute_error": float(abs(mean - label)),
                }
            )
        return rows

    probabilities = torch.softmax(logits, dim=-1)
    mean_prob = probabilities.mean(dim=0)
    pred = mean_prob.argmax(dim=-1)
    entropy = -(mean_prob * mean_prob.clamp_min(1e-12).log()).sum(dim=-1)
    mutual_information = entropy - (
        -(probabilities * probabilities.clamp_min(1e-12).log())
        .sum(dim=-1)
        .mean(dim=0)
    )
    confidence = mean_prob.max(dim=-1).values
    for row_index, label, prediction, conf, ent, mi, prob in zip(
        row_indices, labels, pred, confidence, entropy, mutual_information, mean_prob
    ):
        rows.append(
            {
                "row_index": int(row_index),
                "label": int(label),
                "prediction": int(prediction),
                "confidence": float(conf),
                "probabilities": json.dumps(prob.tolist()),
                "uncertainty": float(ent),
                "mutual_information": float(mi),
                "error": int(prediction != label),
            }
        )
    return rows


def summarize_detection(frame):
    error_column = "error" if "error" in frame else "absolute_error"
    target = frame[error_column].to_numpy()
    if error_column == "absolute_error":
        target = (target >= np.quantile(target, 0.75)).astype(int)
    if len(np.unique(target)) < 2:
        auroc = None
    else:
        auroc = float(roc_auc_score(target, frame["uncertainty"]))
    return {
        "n_samples": int(len(frame)),
        "mean_uncertainty": float(frame["uncertainty"].mean()),
        "error_detection_auroc": auroc,
    }


def main(argv=None):
    from spt.utils.execution import inference_backend
    cli = parse_args(argv)
    if cli.passes < 1:
        raise ValueError("--passes must be positive")
    if cli.batch_size is not None and cli.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    args = load_config(cli.config)
    device = torch.device(cli.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    with inference_backend(args, device) as backend:
        return _run(cli, args, device, backend)


def _run(cli, args, device, backend):
    output_dir = fresh_output(cli.output_dir)
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)
    dataset = load_dataset(args, cli.split)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cli.batch_size or args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = load_model(args, cli.checkpoint, device)
    enable_dropout_only(model)

    is_multi = (
        hasattr(args, "multi_task")
        and isinstance(args.multi_task, list)
        and len(args.multi_task) > 0
    )
    task_names = args.multi_task if is_multi else [args.task]
    records = {task: [] for task in task_names}
    offset = 0

    with torch.no_grad():
        for cate, cont, label in loader:
            cate, cont = cate.to(device), cont.to(device)
            batch_size = len(cate)
            batch_start = offset
            batch_stop = offset + batch_size
            sample_indices = dataset.indices[batch_start:batch_stop]
            offset = batch_stop
            draws = {task: [] for task in task_names}
            for _ in range(cli.passes):
                prediction, _ = model(cate, cont, None)
                if is_multi:
                    for task in task_names:
                        draws[task].append(prediction[task].cpu())
                else:
                    draws[task_names[0]].append(prediction.cpu())

            for task in task_names:
                task_label = label[task] if is_multi else label
                task_config = getattr(args, task)
                task_draws = torch.stack(draws[task])
                if is_multi:
                    valid = torch.as_tensor(
                        dataset.task_valid_masks[task][batch_start:batch_stop],
                        dtype=torch.bool,
                    )
                    if not valid.any():
                        continue
                    task_draws = task_draws[:, valid]
                    task_label = task_label[valid]
                    task_indices = sample_indices[valid.numpy()]
                else:
                    task_indices = sample_indices
                records[task].extend(
                    summarize_task(
                        task_draws,
                        task_label.cpu(),
                        task_indices,
                        task_config["mode"],
                    )
                )

    summary = {}
    for task, task_records in records.items():
        frame = pd.DataFrame(task_records)
        frame.to_csv(output_dir / f"{task}_uncertainty.csv", index=False)
        summary[task] = summarize_detection(frame)
    with open(output_dir / "uncertainty_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    metadata = {
        "config": str(Path(cli.config).resolve()),
        "checkpoint": str(Path(cli.checkpoint).resolve()),
        "split": cli.split, "passes": cli.passes, "seed": cli.seed,
        "batch_size": cli.batch_size or args.batch_size,
        "n_samples": len(dataset), "method": "MC dropout; only nn.Dropout enabled",
        "device": str(device), "inference_backend": backend,
    }
    (output_dir / "uncertainty_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote uncertainty outputs to {output_dir}")


if __name__ == "__main__":
    main()


