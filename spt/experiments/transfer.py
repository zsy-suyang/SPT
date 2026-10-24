"""Train or evaluate an explicitly configured disordered-oxide SPT model.

Input preparation, physical-label metrics and target-domain training use the
frozen data partitions and descriptor-specific configurations distributed here.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .common import fresh_output, save_json, set_seed


def prepare_inputs(frame, args, site):
    """Apply the original normalizers without label-eligibility filtering."""
    import torch
    from spt.utils.data_load.data_des import (
        parse_column_to_numpy, X_log_normalize_optimized,
        Y_log_normalize_optimized,
    )

    if "XANES_spec_15" in frame:
        energy, intensity = [], []
        for value in frame["XANES_spec_15"]:
            clean = str(value).replace("[", "").replace("]", "").replace(",", " ")
            array = np.loadtxt(io.StringIO(clean), ndmin=2)
            if array.shape[1] != 2 or len(array) < 2:
                raise ValueError("Spectrum requires energy/intensity pairs")
            e, y = array[:, 0], array[:, 1]
            if np.any(np.diff(e) <= 0):
                raise ValueError("Energies must be strictly increasing")
            grid = np.linspace(e.min(), e.max(), args.num_cont // 2)
            energy.append(grid)
            intensity.append(np.interp(grid, e, y))
        elements = frame["absorption_atom_type"].astype(str)
    else:
        energy = parse_column_to_numpy(frame["XAS_Relative_energy"])
        intensity = parse_column_to_numpy(frame["XAS_Intensity"])
        elements = (
            frame["center_atom_type"].astype(str)
            if "center_atom_type" in frame
            else pd.Series(["O"] * len(frame), index=frame.index)
        )
        if "center_atom_type" not in frame and site != "oxygen":
            raise ValueError("Missing non-oxygen absorber identities")
    if any(
        len(e) != args.num_cont // 2 or len(y) != len(e)
        or not np.isfinite(e).all() or not np.isfinite(y).all()
        or np.any(np.asarray(e) <= 0)
        for e, y in zip(energy, intensity)
    ):
        raise ValueError("Spectral length, finite values or positive energies failed")
    unknown = sorted(set(elements) - set(args.ele_dict))
    if unknown:
        raise ValueError(f"Unsupported absorber elements: {unknown}")
    x = X_log_normalize_optimized(energy, args.min_log_data, args.max_log_data)
    y = Y_log_normalize_optimized(intensity)
    cont = np.empty((len(frame), args.num_cont), dtype=np.float32)
    cont[:, 0::2], cont[:, 1::2] = np.asarray(x), np.asarray(y)
    cate = np.asarray([args.ele_dict[e] for e in elements], dtype=np.int64)[:, None]
    if not np.isfinite(cont).all():
        raise ValueError("Non-finite normalized spectra")
    return torch.from_numpy(cate), torch.from_numpy(cont)


def descriptor_metrics(truth, prediction, class_values=None):
    """Score every finite physical label, including classes outside the head."""
    from sklearn.metrics import (
        accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
        precision_score, r2_score, recall_score,
    )
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    if truth.shape != prediction.shape or not len(truth):
        raise ValueError("Prediction and truth must have the same nonempty shape")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Non-finite rows cannot be silently removed from evaluation")
    result = {"n": int(len(truth))}
    if class_values is not None:
        labels = sorted(set(truth) | set(prediction))
        supported = np.isin(truth, class_values)
        result.update(
            accuracy=float(accuracy_score(truth, prediction)),
            precision_weighted=float(precision_score(truth, prediction, average="weighted", zero_division=0)),
            recall_weighted=float(recall_score(truth, prediction, average="weighted", zero_division=0)),
            f1_weighted=float(f1_score(truth, prediction, average="weighted", zero_division=0)),
            f1_macro=float(f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)),
            observed_true_classes=int(len(set(truth))),
            macro_label_policy="union of observed true and predicted physical labels",
            unsupported_truth_rows=int((~supported).sum()),
            head_label_coverage=float(supported.mean()),
            unsupported_truth_values=sorted(set(truth[~supported])),
            all_rows_included_in_metrics=True,
            mae=float(np.abs(prediction - truth).mean()),
            within_one=float((np.abs(prediction - truth) <= 1).mean()),
        )
    else:
        mse = float(mean_squared_error(truth, prediction))
        result.update(
            mse=mse, rmse=float(np.sqrt(mse)),
            mae=float(mean_absolute_error(truth, prediction)),
            r2=float(r2_score(truth, prediction)) if len(truth) > 1 and np.var(truth) > 0 else None,
        )
    return result


def score_predictions(raw, frame, task, spec, group_column):
    import torch
    raw = np.asarray(raw)
    if raw.ndim != 2 or len(raw) != len(frame) or not np.isfinite(raw).all():
        raise ValueError("Model output must be finite and cover the complete cohort")
    class_values, probabilities = None, None
    if spec["mode"] == "Cls":
        mapping = dict(zip(spec["label"], spec["value"]))
        if set(mapping) != set(range(raw.shape[1])):
            raise ValueError("Output columns differ from the saved physical-class mapping")
        class_values = [mapping[index] for index in range(raw.shape[1])]
        prediction = np.asarray(class_values)[raw.argmax(1)]
        probabilities = torch.softmax(torch.as_tensor(raw), dim=1).numpy()
    elif spec["mode"] == "Reg" and raw.shape[1] == 1:
        prediction = raw.reshape(-1)
    else:
        raise ValueError("Unsupported descriptor mode or regression output shape")
    metrics = descriptor_metrics(frame[task], prediction, class_values)
    if class_values is not None:
        from sklearn.metrics import roc_auc_score
        truth = frame[task].to_numpy()
        aucs, weights = [], []
        for index, value in enumerate(class_values):
            positive = truth == value
            if positive.any() and (~positive).any():
                aucs.append(float(roc_auc_score(positive, probabilities[:, index])))
                weights.append(int(positive.sum()))
        metrics["auroc_weighted"] = (
            float(np.average(aucs, weights=weights))
            if aucs and np.isin(truth, class_values).all() else None
        )
    columns = [name for name in ("source_row", "structure", group_column, "site") if name in frame]
    table = frame[columns].copy()
    table["label"], table["prediction"] = frame[task].to_numpy(), prediction
    if class_values is not None:
        table["class_values"] = json.dumps(class_values)
        table["probabilities"] = [json.dumps(value.tolist()) for value in probabilities]
        table["label_supported_by_head"] = np.isin(frame[task], class_values)
    group_metrics = {}
    for group in sorted(frame[group_column].astype(str).unique()):
        keep = frame[group_column].astype(str).eq(group).to_numpy()
        group_metrics[group] = descriptor_metrics(frame.loc[keep, task], prediction[keep], class_values)
    return metrics, table, group_metrics


def verify_reference(actual, reference, *, atol=5e-6, rtol=1e-5,
                     strict_probabilities=False, classification=None):
    """Keep discrete results exact; report finite score drift unless strict."""
    if not math.isfinite(atol) or not math.isfinite(rtol) or min(atol, rtol) < 0:
        raise ValueError("Reference probability tolerances must be finite and nonnegative")
    required = {"source_row", "label", "prediction"}
    if any(not required.issubset(frame) or frame.empty for frame in (actual, reference)):
        raise ValueError("Reference comparison requires nonempty row identities, labels and predictions")
    if any(frame.source_row.isna().any() or frame.source_row.duplicated().any()
           for frame in (actual, reference)) or set(actual.source_row) != set(reference.source_row):
        raise ValueError("Reference and evaluation row identities differ")
    reference = reference.set_index("source_row").loc[actual.source_row]
    for name in ("label", "prediction"):
        observed, expected = actual[name].to_numpy(), reference[name].to_numpy()
        if not np.isfinite(observed.astype(float)).all() or not np.isfinite(expected.astype(float)).all():
            raise ValueError(f"Non-finite reference or evaluation {name}")
        if not np.array_equal(expected, observed):
            raise ValueError(f"Frozen reference {name} differs")
    comparison = {
        "row_identities_equal": True, "labels_equal": True,
        "hard_predictions_equal": True, "strict_probabilities": bool(strict_probabilities),
        "probability_atol": atol, "probability_rtol": rtol,
        "max_absolute_probability_difference": None,
        "probabilities_within_tolerance": None,
        "probability_reference_available": "probabilities" in reference,
    }
    has_scores = any("probabilities" in frame or "class_values" in frame
                     for frame in (actual, reference))
    is_classification = has_scores if classification is None else classification
    if is_classification:
        from sklearn.metrics import f1_score
        reference_f1 = float(f1_score(reference.label, reference.prediction,
                                      average="weighted", zero_division=0))
        actual_f1 = float(f1_score(actual.label, actual.prediction,
                                 average="weighted", zero_division=0))
        if actual_f1 != reference_f1:
            raise ValueError("Frozen reference weighted F1 differs")
        comparison.update(f1_weighted_equal=True, reference_f1_weighted=reference_f1)
    if not has_scores:
        if strict_probabilities:
            raise ValueError("Strict probability comparison requires probability tables")
        return comparison
    def score_arrays(frame):
        if not {"probabilities", "class_values"}.issubset(frame):
            raise ValueError("Reference and evaluation require probabilities and class_values")
        try:
            probabilities = np.stack(frame.probabilities.map(json.loads)).astype(float)
            classes = np.stack(frame.class_values.map(json.loads)).astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid reference probability or class-value arrays") from error
        if (probabilities.ndim != 2 or probabilities.shape[1] == 0
                or classes.shape != probabilities.shape):
            raise ValueError("Reference probability and class-value shapes differ")
        if not np.isfinite(probabilities).all() or not np.isfinite(classes).all():
            raise ValueError("Non-finite reference or evaluation probabilities/class values")
        if ((probabilities < 0).any() or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-5, rtol=0)):
            raise ValueError("Reference probabilities must be normalized values in [0, 1]")
        if any(len(np.unique(row)) != len(row) for row in classes):
            raise ValueError("Reference probability class values must be unique")
        decoded = classes[np.arange(len(frame)), probabilities.argmax(axis=1)]
        if not np.array_equal(decoded, frame.prediction.to_numpy()):
            raise ValueError("Reference probabilities disagree with hard predictions")
        return probabilities, classes

    observed, actual_classes = score_arrays(actual)
    if not {"probabilities", "class_values"}.intersection(reference.columns):
        if strict_probabilities:
            raise ValueError("Strict probability comparison requires probability tables")
        return comparison
    expected, reference_classes = score_arrays(reference)
    if expected.shape != observed.shape:
        raise ValueError("Frozen reference probability shapes differ")
    if not np.array_equal(reference_classes, actual_classes):
        raise ValueError("Frozen reference probability class order differs")
    difference = float(np.abs(expected - observed).max())
    close = bool(np.allclose(expected, observed, rtol=rtol, atol=atol))
    comparison.update(max_absolute_probability_difference=difference,
                      probabilities_within_tolerance=close)
    if not close and strict_probabilities:
        raise ValueError(f"Frozen reference probabilities differ: maximum absolute "
                         f"difference {difference:.8g}, atol={atol:.8g}, rtol={rtol:.8g}")
    return comparison


def adaptation_trainer(args, device):
    """Retain the original SPT optimizer groups and adaptation schedule.

    Target-domain training uses the fixed training budget and final-epoch save
    rule. The base trainer supplies the optimizer, losses and learning-rate
    schedule specified by the configuration.
    """
    from spt.training.trainer_delta import DefaultTrainer

    class AdaptationTrainer(DefaultTrainer):
        def __init__(self, config, runtime_device):
            super().__init__(config, runtime_device)
            self.adaptation_strategy = config.adaptation_strategy
            if (self.adaptation_strategy != "scratch"
                    and getattr(config, "adaptation_head_scope", "projection") != "projection"):
                raise ValueError("Frozen SPT transfer cases use the original projection head")
            self._head_prefixes = ("projection.",)
            self._encoder_frozen, self._unfreeze_step = False, None
            multiplier = float(getattr(config, "backbone_lr_multiplier", 1.0))
            if not math.isfinite(multiplier) or multiplier <= 0:
                raise ValueError("backbone_lr_multiplier must be finite and positive")
            if multiplier != 1.0 and self.adaptation_strategy != "scratch":
                head_ids = {id(value) for name, value in self.model.named_parameters()
                            if name.startswith("projection.")}
                groups = []
                for original in self.optim.param_groups:
                    for is_head, scale in ((False, multiplier), (True, 1.0)):
                        parameters = [value for value in original["params"]
                                      if (id(value) in head_ids) == is_head]
                        if parameters:
                            group = dict(original, params=parameters, lr_multiplier=scale,
                                         adaptation_group="head" if is_head else "backbone")
                            group["lr"] = original["lr"] * scale
                            groups.append(group)
                self.optim.param_groups[:] = groups
            self._frozen_eval_hook = self.model.register_forward_pre_hook(self._keep_frozen_eval)
            if self.adaptation_strategy in {"head_only", "gradual_unfreeze"}:
                self._set_encoder_trainable(False)
            elif self.adaptation_strategy not in {"scratch", "full_finetune"}:
                raise ValueError(f"Unknown adaptation strategy: {self.adaptation_strategy}")
            self.initial_trainable_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        def _set_encoder_trainable(self, trainable):
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(bool(trainable) or name.startswith("projection."))
            self._encoder_frozen = not bool(trainable)

        def _keep_frozen_eval(self, module, _inputs):
            import torch
            if not self._encoder_frozen or not module.training or not torch.is_grad_enabled():
                return
            for child in module.children():
                child.eval()
            module.projection.train()

        def _adjust_learning_rate_iter(self, step):
            super()._adjust_learning_rate_iter(step)
            for group in self.optim.param_groups:
                group["lr"] *= group.get("lr_multiplier", 1.0)

        def train(self, train_dataloader, valid_dataloader=None):
            if valid_dataloader is not None:
                raise ValueError("Frozen target recipes do not perform in-training validation")
            if self.adaptation_strategy == "gradual_unfreeze":
                fraction = float(self.args.unfreeze_fraction)
                if not 0 < fraction < 1:
                    raise ValueError("unfreeze_fraction must lie strictly between zero and one")
                self._unfreeze_step = max(1, int(len(train_dataloader) * self.epoch * fraction))
            return super().train(train_dataloader, None)

        def train_iter(self, step, dataloader):
            if self._encoder_frozen and self._unfreeze_step is not None and step >= self._unfreeze_step:
                self._set_encoder_trainable(True)
            return super().train_iter(step, dataloader)

    return AdaptationTrainer(args, device)


def attach_source_anchor(model, source_state, weight):
    """Original training-only backbone anchoring, without extra parameters."""
    import torch
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("source_anchor_weight must be finite and nonnegative")
    if not weight:
        return None
    anchors = []
    for name, parameter in model.named_parameters():
        if name.startswith("projection."):
            continue
        reference = source_state[name]
        if reference.shape != parameter.shape or reference.dtype != parameter.dtype:
            raise ValueError(f"Source anchor differs: {name}")
        anchors.append((parameter, reference.detach().to(parameter.device).clone()))

    def add_anchor(module, inputs, output):
        if not module.training or not torch.is_grad_enabled():
            return None
        if not isinstance(output, tuple) or len(output) != 2 or output[1] is None:
            return None
        terms = [(parameter.float() - reference.float()).square().sum()
                 for parameter, reference in anchors if parameter.requires_grad]
        if not terms:
            return None
        return output[0], output[1] + torch.stack(terms).sum() * (weight / 2.0)

    return model.register_forward_hook(add_anchor)


def train(args):
    """Fit an explicit frozen target recipe on its original ordered row budget."""
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    import torch
    import yaml
    from spt.config.cfg import load_config
    from spt.utils.runtime import resolve_device

    config = Path(args.config).expanduser().resolve()
    data_file = Path(args.data_file).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.site != manifest.get("site"):
        raise ValueError("Requested site family differs from the frozen transfer manifest")
    if manifest.get("training_budget_unit") != "spectra":
        raise ValueError("The paper's target comparison requires the frozen spectral-row budget")
    budget = manifest["train_budgets"][str(args.budget)]
    selected = list(budget["train_source_rows"])
    if len(selected) != len(set(selected)) or len(selected) != budget["train_n"]:
        raise ValueError("Frozen training-row budget is inconsistent")
    frame = pd.read_csv(data_file, index_col=0)
    group_column = manifest["group_column"]
    if "source_row" not in frame or group_column not in frame or frame.source_row.duplicated().any():
        raise ValueError("Fitting CSV requires unique source_row and saved parent-group identities")
    if set(frame.source_row) != set(selected):
        raise ValueError("Fitting CSV must contain exactly the selected frozen TRAIN rows")
    frame = frame.set_index("source_row", drop=False).loc[selected].reset_index(drop=True)
    if frame[group_column].isna().any():
        raise ValueError("Fitting rows have missing parent identities")
    groups = frame[group_column].astype(str)
    assignments = budget["assignments"]
    if any(assignments.get(group) != "train" for group in groups):
        raise ValueError("Fitting CSV contains a non-TRAIN parent")
    if set(groups) != set(budget["train_groups"]):
        raise ValueError("Fitting parents differ from the frozen training budget")

    model_args = load_config(config, args.project_root, {"device": args.device})
    if int(model_args.seed) != int(manifest["seed"]):
        raise ValueError("Training seed differs from the frozen row-budget manifest")
    if str(getattr(model_args, "architecture", "spt")).lower() != "spt":
        raise ValueError("Native transfer training requires architecture=spt")
    if getattr(model_args, "multi_task", None):
        raise ValueError("Frozen transfer cases are single-task")
    if (bool(getattr(model_args, "evaluate_during_training", True))
            or bool(getattr(model_args, "evaluate_test_after_training", True))
            or getattr(model_args, "checkpoint_policy", None) != "last"
            or float(getattr(model_args, "ema_decay", 0.0) or 0.0) != 0.0):
        raise ValueError("Target recipes require final raw weights and no in-training evaluation")
    if (model_args.adaptation_strategy != "scratch"
            and getattr(model_args, "adaptation_head_scope", "projection") != "projection"):
        raise ValueError("The reported SPT target recipes adapt the original projection head")
    if float(getattr(model_args, "head_category_dropout", 0.0) or 0.0):
        raise ValueError("Reported target cases do not use category dropout")
    task = model_args.task
    if task not in frame or not np.isfinite(frame[task].to_numpy(float)).all():
        raise ValueError("All frozen fitting labels must be finite")
    spec = getattr(model_args, task)
    if spec["mode"] != "Cls":
        raise ValueError("The reported transfer entry point covers OS, CN and OCN classification")
    mapping = {float(value): int(label) for value, label in zip(spec["value"], spec["label"])}
    if sorted(mapping.values()) != list(range(spec["num"])):
        raise ValueError("Invalid physical-class map")
    if set(frame[task].astype(float)) - set(mapping):
        raise ValueError("Frozen fitting labels lie outside the saved class map")
    is_scratch = model_args.adaptation_strategy == "scratch"
    source_state = None
    if is_scratch:
        if args.source_checkpoint or args.source_config:
            raise ValueError("Scratch training does not load source weights")
        model_args.ckpt = None
    else:
        if not args.source_checkpoint or not args.source_config:
            raise ValueError("Transfer requires the matching frozen source config and checkpoint")
        source_args = load_config(args.source_config, args.project_root)
        source_spec = getattr(source_args, source_args.task)
        if source_args.task != task or any(source_spec[key] != spec[key] for key in ("mode", "value", "label")):
            raise ValueError("The reported native transfer preserves the complete source class map")
        for field in ("ele_dict", "min_log_data", "max_log_data", "num_cont"):
            if getattr(source_args, field) != getattr(model_args, field):
                raise ValueError(f"Source and target input specification differs: {field}")
        model_args.ckpt = str(Path(args.source_checkpoint).expanduser().resolve())
        source_state = torch.load(model_args.ckpt, map_location="cpu", weights_only=False)["net_state_dict"]
    model_args.ckpt_strict, model_args.load_optim = True, False
    model_args.num_workers, model_args.exp_name = 0, "fixed_final"
    output_dir = fresh_output(args.output_dir)
    model_args.save_folder = str(output_dir / "training")
    model_args.save_log = str(output_dir / "tensorboard")
    model_args.data_file = str(data_file)
    device = resolve_device(model_args)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    set_seed(int(model_args.seed))
    cate, cont = prepare_inputs(frame, model_args, args.site)
    labels = torch.tensor([mapping[float(value)] for value in frame[task]], dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(cate, cont, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=model_args.batch_size,
                                       shuffle=True, num_workers=0, drop_last=False)
    trainer = adaptation_trainer(model_args, device)
    anchor_handle = None
    initialization = {"mode": "scratch" if is_scratch else "tl", "exact_source_state": False}
    try:
        if trainer.ema is not None:
            raise ValueError("Target fitting requires raw, non-EMA checkpoint weights")
        if source_state is not None:
            state = trainer.model.state_dict()
            if state.keys() != source_state.keys() or any(
                value.shape != source_state[name].shape or value.dtype != source_state[name].dtype
                or not torch.equal(value.detach().cpu(), source_state[name])
                for name, value in state.items()
            ):
                raise ValueError("Model initialization differs from the exact source checkpoint")
            initialization["exact_source_state"] = True
            anchor_handle = attach_source_anchor(trainer.model, source_state,
                                                 float(getattr(model_args, "source_anchor_weight", 0.0) or 0.0))
        elif getattr(model_args, "spt_scratch_init", "legacy") == "variance_balanced":
            scale = float(getattr(model_args, "category_embedding_initial_scale", 0.1))
            if scale != 0.1:
                raise ValueError("Frozen variance-balanced Scratch scale is 0.1")
            with torch.no_grad():
                trainer.model.categorical_embeds.weight.mul_(scale)
            initialization.update(spt_scratch_init="variance_balanced", category_embedding_initial_scale=scale)
        frozen_config = output_dir / "config.yaml"
        frozen_config.write_text(yaml.safe_dump(vars(model_args), sort_keys=False), encoding="utf-8")
        checkpoint = trainer.train(loader, None)
        if trainer.best_selection is not None or trainer.best_checkpoint_path is not None:
            raise ValueError("Target training unexpectedly selected a validation checkpoint")
        record = {
            "task": task, "training_fraction": args.budget, "train_n": len(frame),
            "train_source_rows": selected, "train_groups": sorted(set(groups)),
            "checkpoint": str(checkpoint), "checkpoint_policy": "fixed_final_epoch",
            "checkpoint_weights": "raw", "final_checkpoint": trainer.last_selection,
            "initialization": initialization,
            "initial_trainable_parameters": trainer.initial_trainable_parameters,
            "total_parameters": sum(value.numel() for value in trainer.model.parameters()),
            "inputs": {name: {"path": str(path)} for name, path in
                       (("config", config), ("manifest", manifest_path), ("fitting_data", data_file))},
        }
        if source_state is not None:
            source_path = Path(model_args.ckpt)
            record["inputs"]["source_checkpoint"] = {"path": str(source_path)}
            source_config = Path(args.source_config).resolve()
            record["inputs"]["source_config"] = {"path": str(source_config)}
        save_json(output_dir / "training_complete.json", record)
        return record
    finally:
        if anchor_handle is not None:
            anchor_handle.remove()
        trainer._frozen_eval_hook.remove()
        trainer.writer.close()


def evaluate(args):
    """Strictly load one supplied frozen checkpoint and score a fixed CSV."""
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    import torch
    from spt.config.cfg import load_config
    from spt.models import build_model
    from spt.utils.runtime import resolve_device

    config, checkpoint, data_file = (Path(value).expanduser().resolve() for value in
                                     (args.config, args.checkpoint, args.data_file))
    for path in (config, checkpoint, data_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    model_args = load_config(config, args.project_root, {"device": args.device})
    if getattr(model_args, "multi_task", None):
        raise ValueError("The frozen transfer comparison uses single-task models")
    if str(getattr(model_args, "architecture", "spt")).lower() != "spt":
        raise ValueError("This entry point currently evaluates native SPT checkpoints")
    frame = pd.read_csv(data_file, index_col=0)
    group_column = args.group_column
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        saved_group = manifest["group_column"]
        if args.site != manifest["site"] or saved_group not in frame:
            raise ValueError("Evaluation table differs from the frozen manifest site or parent column")
        if frame[saved_group].isna().any():
            raise ValueError("Evaluation table has missing parent identities")
        expected_ids = set(manifest["task_source_rows"][args.site][model_args.task])
        if "source_row" not in frame or set(frame.source_row) != expected_ids:
            raise ValueError("Complete target row identities differ from the frozen manifest")
        assigned = frame[saved_group].astype(str).map(manifest["single_split"]["assignments"])
        if assigned.isna().any():
            raise ValueError("Evaluation parent is absent from the frozen manifest")
        if args.cohort == "test":
            frame = frame.loc[assigned.eq("test")].copy()
        frame = frame.sort_values("source_row").reset_index(drop=True)
        if group_column not in frame:
            frame.insert(frame.columns.get_loc(saved_group) + 1, group_column, frame[saved_group].astype(str))
        elif not frame[group_column].astype(str).equals(frame[saved_group].astype(str)):
            raise ValueError("Evaluation parent aliases disagree")
    required = {"source_row", group_column, model_args.task}
    if not required.issubset(frame):
        raise ValueError(f"Evaluation table lacks columns: {sorted(required - set(frame))}")
    if frame.source_row.isna().any() or frame[group_column].isna().any():
        raise ValueError("Frozen row identities and parent groups must be present")
    if frame.source_row.duplicated().any():
        raise ValueError("Evaluation source_row identifiers must be unique")
    if len(frame) != args.expected_rows:
        raise ValueError(f"Frozen cohort requires {args.expected_rows} rows, received {len(frame)}")
    set_seed(int(model_args.seed))
    device = resolve_device(model_args)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_model(model_args, device).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["net_state_dict"], strict=True)
    model.eval()
    cate, cont = prepare_inputs(frame, model_args, args.site)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(frame), args.batch_size):
            output, _ = model(cate[start:start + args.batch_size].to(device),
                              cont[start:start + args.batch_size].to(device), None)
            chunks.append(output.detach().float().cpu())
    metrics, table, groups = score_predictions(
        torch.cat(chunks).numpy(), frame, model_args.task,
        getattr(model_args, model_args.task), group_column,
    )
    reference_comparison = None
    if args.reference_predictions:
        reference_comparison = verify_reference(
            table, pd.read_csv(args.reference_predictions),
            atol=float(getattr(args, "reference_atol", 5e-6)),
            rtol=float(getattr(args, "reference_rtol", 1e-5)),
            strict_probabilities=bool(getattr(args, "strict_probabilities", False)),
            classification=getattr(model_args, model_args.task)["mode"] == "Cls",
        )
        if ("f1_weighted" in metrics
                and metrics["f1_weighted"] != reference_comparison["reference_f1_weighted"]):
            raise ValueError("Evaluation weighted F1 differs from the frozen reference")
    output_dir = fresh_output(args.output_dir)
    table.to_csv(output_dir / "predictions.csv", index=False)
    import platform
    result = {
        "task": model_args.task, "site": args.site, "metrics": metrics,
        "group_metrics": groups,
        "inputs": {name: {"path": str(path)} for name, path in
                   (("config", config), ("checkpoint", checkpoint), ("data", data_file))},
        "evaluation_policy": "complete supplied frozen cohort; no label eligibility filtering",
        "execution": {
            "device": str(device), "dtype": "float32", "seed": int(model_args.seed),
            "batch_size": args.batch_size, "torch_num_threads": torch.get_num_threads(),
            "python_version": platform.python_version(), "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda, "numpy_version": np.__version__,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }
    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        result["inputs"]["manifest"] = {"path": str(manifest_path)}
        result["cohort"] = args.cohort
    if reference_comparison is not None:
        reference_path = Path(args.reference_predictions).resolve()
        result["inputs"]["reference_predictions"] = {
            "path": str(reference_path)}
        result["reference_comparison"] = reference_comparison
    save_json(output_dir / "metrics.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["evaluate", "train"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="Frozen checkpoint for evaluation")
    parser.add_argument("--data-file", required=True, help="Frozen evaluation cohort or exact fitting CSV")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--manifest", help="Frozen spectral-row training-budget manifest")
    parser.add_argument("--cohort", choices=["test", "all"], default="test",
                        help="With --manifest, use its fixed test parents or the complete target table")
    parser.add_argument("--budget", choices=[10, 25, 50], type=int)
    parser.add_argument("--source-config", help="Matching source config for transfer training")
    parser.add_argument("--source-checkpoint", help="Matching source checkpoint for transfer training")
    parser.add_argument("--group-column", default="transfer_group")
    parser.add_argument("--site", choices=["metal", "oxygen"], default="metal")
    parser.add_argument("--project-root")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--reference-predictions")
    parser.add_argument("--reference-atol", type=float, default=5e-6,
                        help="Absolute tolerance for reference probabilities only")
    parser.add_argument("--reference-rtol", type=float, default=1e-5,
                        help="Relative tolerance for reference probabilities only")
    parser.add_argument("--strict-probabilities", action="store_true",
                        help="Fail on probability differences beyond the reference tolerances; "
                             "identities, labels, hard predictions and weighted F1 remain exact by default")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    if (not math.isfinite(args.reference_atol) or not math.isfinite(args.reference_rtol)
            or min(args.reference_atol, args.reference_rtol) < 0):
        parser.error("Reference probability tolerances must be finite and nonnegative")
    if args.strict_probabilities and (args.operation != "evaluate" or not args.reference_predictions):
        parser.error("--strict-probabilities requires evaluation with --reference-predictions")
    if args.operation == "evaluate":
        if not args.checkpoint or args.expected_rows is None or args.expected_rows < 1:
            parser.error("Evaluation requires --checkpoint and a positive --expected-rows")
        return evaluate(args)
    if not args.manifest or args.budget is None:
        parser.error("Training requires --manifest and --budget")
    if args.checkpoint:
        parser.error("Use --source-checkpoint for transfer training, not --checkpoint")
    return train(args)


if __name__ == "__main__":
    main()
