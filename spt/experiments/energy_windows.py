#!/usr/bin/env python3
"""Frozen-model, label-independent XANES energy-window interventions.

This script never retrains a model or changes a saved split.  Windows are
fixed in relative eV before predictions are inspected.  The test labels are
used only to evaluate the already-defined interventions.

For every clean spectrum, E0 is the sampled energy at the maximum positive
dI/dE.  Equal-width windows are then moved over E-E0.  Two complementary
interventions are provided:

* endpoint_linear: replace the samples in a window by a straight line between
  the nearest clean samples outside that window;
* element_train_median: replace the window by the median normalized spectrum
  of the same absorber element, estimated only from the frozen train split and
  aligned by its own E0.

The exact training preprocessing and prediction functions are imported from
the shared public experiment module.  Outputs include condition-level predictions,
overall and element-stratified F1/R2 changes, material_id cluster-bootstrap
95% intervals, and perturbation magnitudes normalized by each raw spectrum's
intensity span.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    load_config,
    predict,
    project_path,
    raw_to_cont,
    validate_spectrum,
    verify_reference,
    add_model_arguments,
    resolve_model_arguments,
)


REGION_BOUNDS = (
    ("pre_edge", -math.inf, -2.0),
    ("edge_white_line", -2.0, 15.0),
    ("post_edge", 15.0, math.inf),
)


@dataclass(frozen=True)
class Window:
    index: int
    lo: float
    hi: float

    @property
    def center(self) -> float:
        return 0.5 * (self.lo + self.hi)

    @property
    def tag(self) -> str:
        return f"w{_number_tag(self.lo)}_{_number_tag(self.hi)}"

    @property
    def region(self) -> str:
        if self.center < -2.0:
            return "pre_edge"
        if self.center <= 15.0:
            return "edge_white_line"
        return "post_edge"

    def overlap_fractions(self) -> dict[str, float]:
        width = self.hi - self.lo
        result = {}
        for name, region_lo, region_hi in REGION_BOUNDS:
            overlap = max(0.0, min(self.hi, region_hi) - max(self.lo, region_lo))
            result[f"overlap_{name}"] = overlap / width
        return result


@dataclass
class BootstrapPlan:
    group_names: np.ndarray
    row_to_group: np.ndarray
    weights: np.ndarray


def window_metadata(window: Window) -> dict:
    """Return the common window fields required by every result table."""
    return {
        "window_index": window.index,
        "window_lo_ev": window.lo,
        "window_hi_ev": window.hi,
        "window_center_ev": window.center,
        "operational_region": window.region,
        **window.overlap_fractions(),
    }


def _number_tag(value: float) -> str:
    return (f"{value:.6g}".replace("-", "m").replace(".", "p"))


def _stable_seed(seed: int, text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    sequence = np.random.SeedSequence(
        [int(seed), int.from_bytes(digest[:4], "little")]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def make_windows(start: float, stop: float, width: float, stride: float) -> list[Window]:
    values = (start, stop, width, stride)
    if not all(np.isfinite(values)) or width <= 0 or stride <= 0 or stop <= start:
        raise ValueError("window range, width, and stride must be finite and positive")
    if width > stop - start + 1e-10:
        raise ValueError("window width exceeds the requested range")
    count = int(math.floor((stop - start - width) / stride + 1e-10)) + 1
    windows = [Window(i, start + i * stride, start + i * stride + width)
               for i in range(count)]
    if not windows or windows[-1].hi > stop + 1e-8:
        raise RuntimeError("invalid deterministic window grid")
    return windows


def make_region_aligned_windows(
    start: float, stop: float, width: float, stride: float
) -> list[Window]:
    """Build equal-width windows without crossing the operational boundaries.

    Each region is tiled independently.  If the regular stride does not end at
    a region boundary, one final equal-width window is anchored to that
    boundary.  This avoids calling a mixed pre-edge/edge window a regional
    intervention while retaining the same perturbation width everywhere.
    """
    regions = [
        (start, min(stop, -2.0)),
        (max(start, -2.0), min(stop, 15.0)),
        (max(start, 15.0), stop),
    ]
    result = []
    for lower, upper in regions:
        if upper - lower < width - 1e-9:
            continue
        local = make_windows(lower, upper, width, stride)
        if local[-1].hi < upper - 1e-8:
            anchored_lower = upper - width
            if not np.isclose(local[-1].lo, anchored_lower):
                local.append(Window(len(local), anchored_lower, upper))
        result.extend(local)
    return [Window(index, window.lo, window.hi)
            for index, window in enumerate(result)]


def edge_energy(energy, intensity) -> tuple[float, float, int]:
    """Return E0, max dI/dE, and its sampled index from the clean spectrum."""
    energy, intensity = validate_spectrum(energy, intensity)
    derivative = np.gradient(intensity, energy)
    if not np.all(np.isfinite(derivative)):
        raise ValueError("non-finite dI/dE")
    index = int(np.argmax(derivative))
    maximum = float(derivative[index])
    if maximum <= 0:
        raise ValueError("spectrum has no positive dI/dE for operational E0")
    return float(energy[index]), maximum, index


def normalize_intensity(intensity) -> tuple[np.ndarray, float, float]:
    intensity = np.asarray(intensity, dtype=float)
    minimum = float(np.min(intensity))
    span = float(np.ptp(intensity))
    normalized = (intensity - minimum) / (span + 1e-10)
    return normalized, minimum, span


def endpoint_linear_inpaint(
    relative_energy: np.ndarray, intensity: np.ndarray, window: Window
) -> tuple[np.ndarray, dict]:
    relative_energy = np.asarray(relative_energy, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    mask = (relative_energy >= window.lo) & (relative_energy < window.hi)
    output = intensity.copy()
    if not mask.any():
        return output, {
            "window_points": 0, "changed_points": 0,
            "fallback": "no_samples_in_window",
        }
    left = np.flatnonzero(relative_energy < window.lo)
    right = np.flatnonzero(relative_energy >= window.hi)
    if len(left) and len(right):
        left_index, right_index = int(left[-1]), int(right[0])
        output[mask] = np.interp(
            relative_energy[mask],
            [relative_energy[left_index], relative_energy[right_index]],
            [intensity[left_index], intensity[right_index]],
        )
        fallback = "none"
    else:
        # Linear inpainting is undefined without both bracketing samples.  A
        # no-op avoids inventing a constant extrapolation at a measured edge.
        fallback = "insufficient_bracketing_coverage_noop"
    return output, {
        "window_points": int(mask.sum()),
        "changed_points": int(mask.sum()) if fallback == "none" else 0,
        "fallback": fallback,
    }


def reference_grid(start: float, stop: float, step: float) -> np.ndarray:
    if not np.isfinite(step) or step <= 0:
        raise ValueError("reference grid step must be positive")
    count = int(math.ceil((stop - start) / step)) + 1
    grid = start + np.arange(count, dtype=float) * step
    grid[-1] = stop
    return grid


def build_element_train_medians(
    energies,
    intensities,
    elements,
    material_ids,
    grid: np.ndarray,
    min_support: int,
) -> tuple[dict[str, np.ndarray], dict]:
    """Build structure-balanced E0-aligned train-only median curves."""
    if min_support <= 0:
        raise ValueError("min reference support must be positive")
    elements = np.asarray(elements, dtype=str)
    material_ids = np.asarray(material_ids, dtype=str)
    if not (len(energies) == len(intensities) == len(elements) == len(material_ids)):
        raise ValueError("train reference arrays are not row-aligned")
    references, diagnostics = {}, {}
    for element in sorted(np.unique(elements)):
        rows = []
        row_groups = []
        e0_values = []
        for energy, intensity, material_id in zip(
            np.asarray(energies, dtype=object)[elements == element],
            np.asarray(intensities, dtype=object)[elements == element],
            material_ids[elements == element],
        ):
            e0, _, _ = edge_energy(energy, intensity)
            normalized, _, _ = normalize_intensity(intensity)
            relative = np.asarray(energy, dtype=float) - e0
            rows.append(np.interp(grid, relative, normalized, left=np.nan, right=np.nan))
            row_groups.append(material_id)
            e0_values.append(e0)
        if not rows:
            continue
        spectrum_matrix = np.stack(rows)
        group_curves = []
        for material_id in sorted(set(row_groups)):
            selected = spectrum_matrix[np.asarray(row_groups) == material_id]
            curve = np.full(len(grid), np.nan, dtype=float)
            supported = np.isfinite(selected).any(axis=0)
            curve[supported] = np.nanmedian(selected[:, supported], axis=0)
            group_curves.append(curve)
        # Each parent structure contributes one curve regardless of how many
        # absorbing sites it supplies, preventing repeated sites from
        # dominating the counterfactual reference.
        matrix = np.stack(group_curves)
        support = np.isfinite(matrix).sum(axis=0)
        median = np.full(len(grid), np.nan, dtype=float)
        any_support = support > 0
        # Restrict nanmedian to columns with at least one finite observation so
        # NumPy does not emit an all-NaN warning for intentionally unsupported
        # grid tails.
        median[any_support] = np.nanmedian(matrix[:, any_support], axis=0)
        valid = (support >= min_support) & np.isfinite(median)
        if valid.sum() < 2:
            raise ValueError(
                f"Element {element!r} has fewer than two supported median-grid points"
            )
        # Keep unsupported points as NaN.  A target row will use endpoint
        # inpainting for only those target samples, and that fallback is logged.
        median[~valid] = np.nan
        references[element] = median
        diagnostics[element] = {
            "n_train_spectra": int(len(rows)),
            "n_train_materials": int(len(group_curves)),
            "supported_grid_points": int(valid.sum()),
            "total_grid_points": int(len(grid)),
            "min_support_parent_structures": int(min_support),
            "minimum_parent_structure_support_on_valid_grid": int(
                support[valid].min()
            ),
            "median_e0_ev": float(np.median(e0_values)),
        }
    return references, diagnostics


def element_train_median_replace(
    relative_energy: np.ndarray,
    intensity: np.ndarray,
    element: str,
    window: Window,
    grid: np.ndarray,
    references: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict]:
    if element not in references:
        raise ValueError(f"No train-median reference for test element {element!r}")
    relative_energy = np.asarray(relative_energy, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    mask = (relative_energy >= window.lo) & (relative_energy < window.hi)
    output = intensity.copy()
    if not mask.any():
        return output, {
            "window_points": 0, "changed_points": 0,
            "fallback": "no_samples_in_window",
        }
    normalized, minimum, span = normalize_intensity(intensity)
    target_indices = mask.nonzero()[0]
    replacement = np.interp(
        relative_energy[mask], grid, references[element], left=np.nan, right=np.nan
    )
    supported = np.isfinite(replacement)
    fallback_count = int((~supported).sum())
    selected = np.array([], dtype=int)
    if supported.any():
        # A raised-cosine (sin^2) gate is zero at both window boundaries and
        # one at its centre, preventing a hard train/test reference splice.
        phase = np.clip(
            (relative_energy[target_indices[supported]] - window.lo)
            / (window.hi - window.lo),
            0.0,
            1.0,
        )
        blend = np.sin(np.pi * phase) ** 2
        selected = target_indices[supported]
        normalized[selected] = (
            (1.0 - blend) * normalized[selected] + blend * replacement[supported]
        )
    # Unsupported reference coordinates remain exactly clean rather than being
    # extrapolated.  Their count is explicit in the condition diagnostics.
    # The formal experiment operates on the already preprocessed model
    # intensity channel. Write back only supported samples in the declared
    # window so the energy channel and every out-of-window model input remain
    # bitwise identical to clean inference.
    if span > 0 and len(selected):
        output[selected] = minimum + span * normalized[selected]
    return output, {
        "window_points": int(mask.sum()),
        "changed_points": int(supported.sum()),
        "fallback": "none" if fallback_count == 0 else "clean_noop_partial",
        "fallback_points": fallback_count,
    }


def perturbation_magnitude(clean, perturbed, relative_energy, window: Window) -> dict:
    clean = np.asarray(clean, dtype=float)
    perturbed = np.asarray(perturbed, dtype=float)
    scale = max(float(np.ptp(clean)), 1e-10)
    difference = (perturbed - clean) / scale
    mask = ((np.asarray(relative_energy) >= window.lo)
            & (np.asarray(relative_energy) < window.hi))
    complete_two_flank = bool(
        mask.any()
        and np.any(np.asarray(relative_energy) < window.lo)
        and np.any(np.asarray(relative_energy) >= window.hi)
    )
    changed = difference[mask]
    # Inputs are cast to the exact float32 representation consumed by the
    # model before this function is called, so an applied row means at least
    # one value truly differs at inference precision.
    numeric_change = changed != 0.0
    return {
        "perturbation_l1_full": float(np.mean(np.abs(difference))),
        "perturbation_l2_full": float(np.sqrt(np.mean(difference ** 2))),
        "perturbation_l1_window": (
            float(np.mean(np.abs(changed))) if len(changed) else 0.0
        ),
        "perturbation_l2_window": (
            float(np.sqrt(np.mean(changed ** 2))) if len(changed) else 0.0
        ),
        "window_sample_fraction": float(mask.mean()),
        "complete_two_flank_coverage": complete_two_flank,
        "numerically_changed_points": int(numeric_change.sum()),
    }


def apply_intervention(
    method: str,
    window: Window,
    energies,
    intensities,
    e0_values,
    elements,
    grid,
    references,
) -> tuple[list[np.ndarray], pd.DataFrame, dict]:
    transformed, rows = [], []
    fallback_counts: dict[str, int] = {}
    for energy, intensity, e0, element in zip(
        energies, intensities, e0_values, elements
    ):
        relative = np.asarray(energy, dtype=float) - float(e0)
        if method == "endpoint_linear":
            result, diagnostic = endpoint_linear_inpaint(relative, intensity, window)
        elif method == "element_train_median":
            result, diagnostic = element_train_median_replace(
                relative, intensity, str(element), window, grid, references
            )
        else:
            raise ValueError(f"Unknown intervention method: {method}")
        clean_float32 = np.asarray(intensity, dtype=np.float32)
        result_float32 = np.asarray(result, dtype=np.float32)
        transformed.append(result_float32)
        magnitude = perturbation_magnitude(
            clean_float32, result_float32, relative, window
        )
        rows.append({
            **magnitude,
            "window_points": diagnostic["window_points"],
            "reference_supported_points": diagnostic["changed_points"],
            "has_window_samples": diagnostic["window_points"] > 0,
            "fully_reference_supported": bool(
                diagnostic["window_points"] > 0
                and diagnostic["changed_points"] == diagnostic["window_points"]
            ),
            "intervention_applied": magnitude["numerically_changed_points"] > 0,
        })
        fallback = diagnostic.get("fallback", "none")
        fallback_counts[fallback] = fallback_counts.get(fallback, 0) + 1
    return transformed, pd.DataFrame(rows), {"fallback_spectrum_counts": fallback_counts}


def coverage_summary(magnitudes: pd.DataFrame) -> dict:
    any_samples = magnitudes["has_window_samples"].to_numpy(dtype=bool)
    complete = magnitudes["complete_two_flank_coverage"].to_numpy(dtype=bool)
    fully_supported = magnitudes["fully_reference_supported"].to_numpy(dtype=bool)
    applied = magnitudes["intervention_applied"].to_numpy(dtype=bool)
    formal = any_samples & complete & fully_supported
    return {
        "n_spectra_any_sample_coverage": int(any_samples.sum()),
        "fraction_spectra_any_sample_coverage": float(any_samples.mean()),
        "n_spectra_complete_two_flank_coverage": int(complete.sum()),
        "fraction_spectra_complete_two_flank_coverage": float(complete.mean()),
        "n_spectra_fully_reference_supported": int(fully_supported.sum()),
        "fraction_spectra_fully_reference_supported": float(fully_supported.mean()),
        "n_spectra_intervention_applied": int(applied.sum()),
        "fraction_spectra_intervention_applied": float(applied.mean()),
        "n_spectra_method_eligible": int(formal.sum()),
        "fraction_spectra_method_eligible": float(formal.mean()),
        "mean_window_point_count": float(magnitudes["window_points"].mean()),
        "mean_reference_supported_point_count": float(
            magnitudes["reference_supported_points"].mean()
        ),
        "mean_modified_point_count": float(
            magnitudes["numerically_changed_points"].mean()
        ),
        "coverage_interpretation_eligible_ge_0p8": bool(formal.mean() >= 0.8),
    }


def common_formal_mask(magnitudes_by_method: dict[str, pd.DataFrame]) -> np.ndarray:
    """Return one label-independent evaluation population shared by all methods."""
    if not magnitudes_by_method:
        raise ValueError("at least one intervention is required")
    lengths = {len(frame) for frame in magnitudes_by_method.values()}
    if len(lengths) != 1:
        raise ValueError("intervention coverage tables are not row-aligned")
    eligible = np.ones(next(iter(lengths)), dtype=bool)
    for frame in magnitudes_by_method.values():
        eligible &= frame["has_window_samples"].to_numpy(dtype=bool)
        eligible &= frame["complete_two_flank_coverage"].to_numpy(dtype=bool)
        eligible &= frame["fully_reference_supported"].to_numpy(dtype=bool)
    return eligible


def make_bootstrap_plan(groups, n_bootstrap: int, seed: int, label: str) -> BootstrapPlan:
    groups = np.asarray(groups, dtype=str)
    names, inverse = np.unique(groups, return_inverse=True)
    if len(names) < 2:
        return BootstrapPlan(names, inverse, np.empty((0, len(names)), dtype=np.int16))
    rng = np.random.default_rng(_stable_seed(seed, label))
    # Drawing G groups with replacement is exactly multinomial cluster bootstrap.
    weights = rng.multinomial(
        len(names), np.full(len(names), 1.0 / len(names)), size=n_bootstrap
    )
    dtype = np.int16 if len(names) < np.iinfo(np.int16).max else np.int32
    return BootstrapPlan(names, inverse, weights.astype(dtype, copy=False))


def _classification_cluster_matrices(labels, predictions, plan, num_classes):
    matrices = np.zeros(
        (len(plan.group_names), num_classes, num_classes), dtype=np.int64
    )
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    np.add.at(matrices, (plan.row_to_group, labels, predictions), 1)
    return matrices


def _weighted_f1_from_confusions(confusions: np.ndarray) -> np.ndarray:
    confusions = np.asarray(confusions, dtype=float)
    true_support = confusions.sum(axis=2)
    predicted_support = confusions.sum(axis=1)
    true_positive = np.diagonal(confusions, axis1=1, axis2=2)
    denominator = true_support + predicted_support
    class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    total = true_support.sum(axis=1)
    return np.divide(
        (class_f1 * true_support).sum(axis=1),
        total,
        out=np.full(len(total), np.nan),
        where=total > 0,
    )


def _regression_sufficient_statistics(labels, predictions, plan):
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    size = len(plan.group_names)
    count = np.zeros(size, dtype=float)
    sum_y = np.zeros(size, dtype=float)
    sum_y2 = np.zeros(size, dtype=float)
    squared_error = np.zeros(size, dtype=float)
    np.add.at(count, plan.row_to_group, 1.0)
    np.add.at(sum_y, plan.row_to_group, labels)
    np.add.at(sum_y2, plan.row_to_group, labels ** 2)
    np.add.at(squared_error, plan.row_to_group, (labels - predictions) ** 2)
    return count, sum_y, sum_y2, squared_error


def _r2_from_statistics(weights, statistics):
    count, sum_y, sum_y2, squared_error = statistics
    n = weights @ count
    total = weights @ sum_y2 - (weights @ sum_y) ** 2 / n
    residual = weights @ squared_error
    return np.divide(
        1.0 - residual / total,
        np.ones_like(total),
        out=np.full_like(total, np.nan),
        where=(n >= 2) & (total > 0),
    )


def metric_score(frame: pd.DataFrame, mode: str, num_classes: int | None) -> float:
    if mode == "Cls":
        labels = frame["label"].to_numpy(dtype=int)
        predictions = frame["prediction"].to_numpy(dtype=int)
        confusion = np.zeros((1, int(num_classes), int(num_classes)), dtype=np.int64)
        np.add.at(confusion, (0, labels, predictions), 1)
        return float(_weighted_f1_from_confusions(confusion)[0])
    labels = frame["label"].to_numpy(dtype=float)
    predictions = frame["prediction"].to_numpy(dtype=float)
    if len(frame) < 2:
        return float("nan")
    total = float(np.sum((labels - labels.mean()) ** 2))
    if total <= 0:
        return float("nan")
    return float(1.0 - np.sum((labels - predictions) ** 2) / total)


def paired_cluster_bootstrap(
    clean: pd.DataFrame,
    perturbed: pd.DataFrame,
    plan: BootstrapPlan,
    mode: str,
    num_classes: int | None,
) -> tuple[dict, np.ndarray]:
    if len(clean) != len(perturbed) or not np.array_equal(
        clean["row_index"].to_numpy(), perturbed["row_index"].to_numpy()
    ):
        raise ValueError("clean and perturbed frames are not row-aligned")
    if not np.array_equal(clean["label"].to_numpy(), perturbed["label"].to_numpy()):
        raise ValueError("clean and perturbed labels differ")
    clean_score = metric_score(clean, mode, num_classes)
    perturbed_score = metric_score(perturbed, mode, num_classes)
    point_drop = clean_score - perturbed_score
    if len(plan.weights) == 0:
        draws = np.array([], dtype=float)
    elif mode == "Cls":
        labels = clean["label"].to_numpy(dtype=int)
        clean_cm = _classification_cluster_matrices(
            labels, clean["prediction"], plan, int(num_classes)
        )
        perturbed_cm = _classification_cluster_matrices(
            labels, perturbed["prediction"], plan, int(num_classes)
        )
        clean_boot = _weighted_f1_from_confusions(
            np.einsum("bg,gij->bij", plan.weights, clean_cm, optimize=True)
        )
        perturbed_boot = _weighted_f1_from_confusions(
            np.einsum("bg,gij->bij", plan.weights, perturbed_cm, optimize=True)
        )
        draws = clean_boot - perturbed_boot
    else:
        clean_statistics = _regression_sufficient_statistics(
            clean["label"], clean["prediction"], plan
        )
        perturbed_statistics = _regression_sufficient_statistics(
            clean["label"], perturbed["prediction"], plan
        )
        draws = (
            _r2_from_statistics(plan.weights, clean_statistics)
            - _r2_from_statistics(plan.weights, perturbed_statistics)
        )
    finite = draws[np.isfinite(draws)]
    result = {
        "clean_score": clean_score,
        "perturbed_score": perturbed_score,
        "score_drop_clean_minus_perturbed": point_drop,
        "bootstrap_ci95_low": (
            float(np.quantile(finite, 0.025)) if len(finite) else None
        ),
        "bootstrap_ci95_high": (
            float(np.quantile(finite, 0.975)) if len(finite) else None
        ),
        "n_bootstrap_requested": int(len(plan.weights)),
        "n_bootstrap_finite": int(len(finite)),
        "n_groups": int(len(plan.group_names)),
        "n_rows": int(len(clean)),
    }
    return result, draws


def prefixed(mapping: dict, prefix: str) -> dict:
    return {f"{prefix}{key}": value for key, value in mapping.items()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-start-ev", type=float, default=-15.0)
    parser.add_argument("--window-stop-ev", type=float, default=45.0)
    parser.add_argument("--window-width-ev", type=float, default=5.0)
    parser.add_argument("--window-stride-ev", type=float, default=2.5)
    parser.add_argument(
        "--region-aligned-windows", action="store_true",
        help=("Tile pre-edge, edge/white-line and post-edge independently so no "
              "window crosses the operational -2 or 15 eV boundaries."),
    )
    parser.add_argument("--reference-grid-step-ev", type=float, default=0.1)
    parser.add_argument("--reference-min-support", type=int, default=3)
    parser.add_argument(
        "--interventions", nargs="+",
        choices=["endpoint_linear", "element_train_median"],
        default=["endpoint_linear", "element_train_median"],
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--max-windows", type=int, default=None,
        help="Deterministic first N windows; marks output as a smoke/partial analysis.",
    )
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="Smoke test only; truncate the test rows and mark output partial.",
    )
    return resolve_model_arguments(parser.parse_args(argv), parser)


def main(argv=None):
    import torch
    from spt.utils.execution import inference_backend

    cli = parse_args(argv)
    args = load_config(project_path(cli.config))
    device = torch.device(cli.device)
    with inference_backend(args, device) as backend:
        return _run(cli, args, device, backend)


def _run(cli, args, device, backend):
    import torch
    from spt.evaluate import _load_model
    from spt.utils import data_load
    from spt.utils.data_load.data_des import parse_column_to_numpy

    config_path = project_path(cli.config)
    checkpoint = project_path(cli.checkpoint)
    reference_predictions = project_path(cli.reference_predictions)
    output = project_path(cli.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    if not config_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("config or checkpoint does not exist")
    if not reference_predictions.is_file():
        raise FileNotFoundError(reference_predictions)
    if cli.batch_size <= 0 or cli.bootstrap_replicates <= 0:
        raise ValueError("batch size and bootstrap replicates must be positive")
    if cli.max_windows is not None and cli.max_windows <= 0:
        raise ValueError("max-windows must be positive")
    if cli.max_batches is not None and cli.max_batches <= 0:
        raise ValueError("max-batches must be positive")

    window_builder = (
        make_region_aligned_windows if cli.region_aligned_windows else make_windows)
    windows = window_builder(
        cli.window_start_ev, cli.window_stop_ev,
        cli.window_width_ev, cli.window_stride_ev)
    full_window_count = len(windows)
    if cli.max_windows is not None:
        windows = windows[:cli.max_windows]

    if getattr(args, "multi_task", None) not in (None, "None", []):
        raise ValueError("Only single-task frozen checkpoints are supported")
    if str(getattr(args, "add_noise", None)).lower() != "none":
        raise ValueError("Frozen clean config must have add_noise=None")
    split_path = project_path(args.split_file)
    data_path = project_path(args.data_file)
    if not split_path.is_file() or not data_path.is_file():
        raise FileNotFoundError("saved split or data CSV does not exist")

    args.ckpt, args.dataset_mode = str(checkpoint), "test"
    args.batch_size, args.num_workers = cli.batch_size, 0
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    test_dataset = getattr(data_load, args.data_name.lower())(dataset="test", args=args)
    train_dataset = getattr(data_load, args.data_name.lower())(dataset="train", args=args)
    full_test_n = len(test_dataset)
    if cli.max_batches is not None:
        limit = min(full_test_n, cli.max_batches * cli.batch_size)
        test_dataset.indices = test_dataset.indices[:limit]
        test_dataset.cate = test_dataset.cate[:limit]
        test_dataset.cont = test_dataset.cont[:limit]
        test_dataset.label = test_dataset.label[:limit]
    if not len(test_dataset) or not len(train_dataset):
        raise ValueError("empty frozen train or test partition")
    partial_rows = len(test_dataset) != full_test_n
    partial_windows = len(windows) != full_window_count

    columns = [
        "material_id", "center_atom_type", "XAS_Relative_energy", "XAS_Intensity"
    ]
    source = pd.read_csv(data_path, comment="#", usecols=columns)
    if source["material_id"].isna().any() or source["center_atom_type"].isna().any():
        raise ValueError("material_id and center_atom_type must be non-missing")
    test_raw = source.iloc[np.asarray(test_dataset.indices, dtype=int)].copy()
    train_raw = source.iloc[np.asarray(train_dataset.indices, dtype=int)].copy()
    train_groups = set(train_raw["material_id"].astype(str))
    test_groups = set(test_raw["material_id"].astype(str))
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(f"Frozen train/test material_id leakage: {len(overlap)} groups")

    test_energies = parse_column_to_numpy(test_raw["XAS_Relative_energy"])
    test_intensities = parse_column_to_numpy(test_raw["XAS_Intensity"])
    train_energies = parse_column_to_numpy(train_raw["XAS_Relative_energy"])
    train_intensities = parse_column_to_numpy(train_raw["XAS_Intensity"])
    for energy, intensity in zip(
        list(test_energies) + list(train_energies),
        list(test_intensities) + list(train_intensities),
    ):
        validate_spectrum(energy, intensity)

    original_cont = np.asarray(test_dataset.cont, dtype=np.float32).copy()
    if original_cont.ndim != 2 or original_cont.shape[1] % 2:
        raise ValueError(
            "Expected a two-dimensional interleaved energy/intensity model input"
        )
    rebuilt = raw_to_cont(test_energies, test_intensities, args)
    if not np.array_equal(rebuilt, original_cont):
        raise RuntimeError("Raw clean preprocessing differs from the training data loader")
    test_model_intensities = original_cont[:, 1::2].copy()
    if any(len(values) != test_model_intensities.shape[1] for values in test_energies):
        raise ValueError("Raw energy vectors do not align with model intensity samples")

    test_e0_details = [edge_energy(e, y) for e, y in zip(test_energies, test_intensities)]
    test_e0 = np.asarray([row[0] for row in test_e0_details])
    test_elements = test_raw["center_atom_type"].astype(str).to_numpy()
    test_material_ids = test_raw["material_id"].astype(str).to_numpy()

    margin = cli.window_width_ev
    median_grid = reference_grid(
        cli.window_start_ev - margin,
        cli.window_stop_ev + margin,
        cli.reference_grid_step_ev,
    )
    references, reference_diagnostics = build_element_train_medians(
        train_energies,
        train_intensities,
        train_raw["center_atom_type"].astype(str).to_numpy(),
        train_raw["material_id"].astype(str).to_numpy(),
        median_grid,
        cli.reference_min_support,
    )
    missing_elements = sorted(set(test_elements) - set(references))
    if missing_elements:
        raise ValueError(f"Test elements missing train references: {missing_elements}")

    model, _ = _load_model(args, device)
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    analysis_site = str(args.data_name)
    clean_dir = output / "clean"
    clean_dir.mkdir(exist_ok=True)

    test_dataset.cont = original_cont
    clean_metrics, clean_frame = predict(model, test_dataset, device, cli.batch_size)
    reference_check = verify_reference(
        clean_frame, reference_predictions, partial_rows
    )
    print("Clean reference match: PASS", flush=True)
    clean_frame.insert(1, "material_id", test_material_ids)
    clean_frame.insert(2, "center_atom_type", test_elements)
    clean_frame.insert(3, "e0_ev", test_e0)
    clean_frame.to_csv(clean_dir / f"{args.task}_predictions.csv", index=False)
    _write_json(clean_dir / "metrics.json", clean_metrics)

    task_conf = getattr(args, args.task)
    mode = task_conf["mode"]
    primary = "F1" if mode == "Cls" else "R2"
    num_classes = int(task_conf["num"]) if mode == "Cls" else None
    manual_clean = metric_score(clean_frame, mode, num_classes)
    if not np.isclose(manual_clean, clean_metrics[primary], rtol=2e-5, atol=2e-5):
        raise RuntimeError(
            f"Manual primary metric {manual_clean} differs from project metric "
            f"{clean_metrics[primary]}"
        )

    plans = {
        "__overall__": make_bootstrap_plan(
            test_material_ids, cli.bootstrap_replicates, cli.seed, "overall"
        )
    }
    for element in sorted(np.unique(test_elements)):
        mask = test_elements == element
        plans[element] = make_bootstrap_plan(
            test_material_ids[mask], cli.bootstrap_replicates, cli.seed,
            f"element:{element}",
        )

    summary_rows, element_rows = [], []

    try:
        for window in windows:
            prepared = {}
            for method in cli.interventions:
                prepared[method] = apply_intervention(
                    method,
                    window,
                    test_energies,
                    test_model_intensities,
                    test_e0,
                    test_elements,
                    median_grid,
                    references,
                )
            common_eligible = common_formal_mask({
                method: item[1] for method, item in prepared.items()
            })
            common_fraction = float(common_eligible.mean())
            formal_plan = make_bootstrap_plan(
                test_material_ids[common_eligible],
                cli.bootstrap_replicates,
                cli.seed,
                f"formal:{analysis_site}:{args.task}:{window.tag}",
            )
            element_formal_plans = {}
            for element in sorted(np.unique(test_elements)):
                element_common = common_eligible & (test_elements == element)
                element_formal_plans[element] = make_bootstrap_plan(
                    test_material_ids[element_common],
                    cli.bootstrap_replicates,
                    cli.seed,
                    f"formal:{analysis_site}:{args.task}:{element}:{window.tag}",
                )

            for method in cli.interventions:
                condition = f"{method}__{window.tag}"
                print(f"Running {condition}", flush=True)
                transformed, magnitudes, intervention_diagnostic = prepared[method]
                transformed_matrix = np.stack(transformed).astype(
                    np.float32, copy=False
                )
                for row_index, (energy, e0) in enumerate(zip(test_energies, test_e0)):
                    relative = np.asarray(energy, dtype=float) - float(e0)
                    inside = (relative >= window.lo) & (relative < window.hi)
                    if not np.array_equal(
                        transformed_matrix[row_index, ~inside],
                        test_model_intensities[row_index, ~inside],
                    ):
                        raise RuntimeError(
                            "Intensity intervention changed an out-of-window model input"
                        )
                intervened_cont = original_cont.copy()
                intervened_cont[:, 1::2] = transformed_matrix
                if not np.array_equal(
                    intervened_cont[:, 0::2], original_cont[:, 0::2]
                ):
                    raise RuntimeError(
                        "Energy channel changed during intensity intervention"
                    )
                test_dataset.cont = intervened_cont
                metrics, frame = predict(model, test_dataset, device, cli.batch_size)
                frame.insert(1, "material_id", test_material_ids)
                frame.insert(2, "center_atom_type", test_elements)
                frame.insert(3, "e0_ev", test_e0)
                for column in magnitudes:
                    frame[column] = magnitudes[column].to_numpy()

                overall, _ = paired_cluster_bootstrap(
                    clean_frame, frame, plans["__overall__"], mode, num_classes
                )
                formal_clean = clean_frame.loc[common_eligible].reset_index(drop=True)
                formal_frame = frame.loc[common_eligible].reset_index(drop=True)
                formal, _ = paired_cluster_bootstrap(
                    formal_clean, formal_frame, formal_plan, mode, num_classes
                )
                coverage = coverage_summary(magnitudes)
                formal_applied_fraction = float(
                    magnitudes.loc[
                        common_eligible, "intervention_applied"
                    ].mean()
                    if common_eligible.any() else 0.0
                )
                magnitude_summary = {
                    f"mean_{column}": float(magnitudes[column].mean())
                    for column in magnitudes
                    if column.startswith("perturbation_") or column == "window_sample_fraction"
                }
                row = {
                    "condition": condition,
                    "method": method,
                    **window_metadata(window),
                    "primary_metric": primary,
                    **overall,
                    **prefixed(formal, "formal_"),
                    **coverage,
                    "fraction_spectra_common_formal_eligible": common_fraction,
                    "fraction_formal_rows_intervention_applied": (
                        formal_applied_fraction
                    ),
                    **magnitude_summary,
                    **intervention_diagnostic,
                }
                summary_rows.append(row)
                condition_dir = output / method / window.tag
                condition_dir.mkdir(parents=True, exist_ok=True)
                frame.to_csv(
                    condition_dir / f"{args.task}_predictions.csv", index=False
                )
                _write_json(
                    condition_dir / "metrics.json",
                    {"project_metrics": metrics, "intervention_summary": row},
                )

                for element in sorted(np.unique(test_elements)):
                    mask = test_elements == element
                    clean_element = clean_frame.loc[mask].reset_index(drop=True)
                    frame_element = frame.loc[mask].reset_index(drop=True)
                    result, _ = paired_cluster_bootstrap(
                        clean_element, frame_element, plans[element], mode, num_classes
                    )
                    element_magnitudes = magnitudes.loc[mask].reset_index(drop=True)
                    element_coverage = coverage_summary(element_magnitudes)
                    element_common = common_eligible[mask]
                    formal_clean_element = clean_element.loc[
                        element_common
                    ].reset_index(drop=True)
                    formal_frame_element = frame_element.loc[
                        element_common
                    ].reset_index(drop=True)
                    formal_element, _ = paired_cluster_bootstrap(
                        formal_clean_element,
                        formal_frame_element,
                        element_formal_plans[element],
                        mode,
                        num_classes,
                    )
                    element_common_fraction = float(element_common.mean())
                    element_applied_fraction = float(
                        element_magnitudes.loc[
                            element_common, "intervention_applied"
                        ].mean()
                        if element_common.any() else 0.0
                    )
                    element_row = {
                        "condition": condition,
                        "method": method,
                        **window_metadata(window),
                        "center_atom_type": element,
                        "primary_metric": primary,
                        **result,
                        **prefixed(formal_element, "formal_"),
                        **element_coverage,
                        "fraction_spectra_common_formal_eligible": (
                            element_common_fraction
                        ),
                        "fraction_formal_rows_intervention_applied": (
                            element_applied_fraction
                        ),
                    }
                    element_rows.append(element_row)
                print(
                    f"{condition}: {primary}={overall['perturbed_score']:.6f}, "
                    f"full_drop={overall['score_drop_clean_minus_perturbed']:+.6f}, "
                    f"formal_drop={formal['score_drop_clean_minus_perturbed']:+.6f}",
                    flush=True,
                )
    finally:
        test_dataset.cont = original_cont

    summary_frame = pd.DataFrame(summary_rows)
    element_frame = pd.DataFrame(element_rows)
    summary_frame.to_csv(output / "window_summary.csv", index=False)
    element_frame.to_csv(output / "element_window_summary.csv", index=False)

    region_rows = []
    for level, frame, grouping in (
        ("overall", summary_frame, ["method", "operational_region"]),
        (
            "element", element_frame,
            ["method", "center_atom_type", "operational_region"],
        ),
    ):
        for keys, subset in frame.groupby(grouping, sort=True, dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            key_values = dict(zip(grouping, keys))
            coverage_eligible = (
                (subset["fraction_spectra_common_formal_eligible"] >= 0.8)
                & (subset["fraction_formal_rows_intervention_applied"] >= 0.8)
            )
            eligible_drops = subset.loc[
                coverage_eligible,
                "formal_score_drop_clean_minus_perturbed",
            ].dropna()
            all_drops = subset[
                "formal_score_drop_clean_minus_perturbed"
            ].dropna()
            region_rows.append({
                "level": level,
                "center_atom_type": key_values.get("center_atom_type"),
                "method": key_values["method"],
                "operational_region": key_values["operational_region"],
                "n_predeclared_windows": int(len(subset)),
                "n_coverage_eligible_windows": int(coverage_eligible.sum()),
                "mean_formal_score_drop_all_windows": (
                    float(all_drops.mean()) if len(all_drops) else None
                ),
                "mean_formal_score_drop_coverage_eligible_windows": (
                    float(eligible_drops.mean()) if len(eligible_drops) else None
                ),
                "region_ci_available": False,
            })
    pd.DataFrame(region_rows).to_csv(output / "region_summary.csv", index=False)

    metadata = {
        "description": (
            "Frozen-model equal-width relative-eV intervention analysis; "
            "test labels are used only for evaluation after windows are fixed"
        ),
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "reference_predictions": str(reference_predictions),
        "split_file": str(split_path),
        "data_file": str(data_path),
        "split": "test",
        "train_reference_split": "train",
        "train_test_material_id_overlap": 0,
        "n_train_rows_for_reference": len(train_dataset),
        "n_test_rows": len(test_dataset),
        "full_test_rows": full_test_n,
        "partial_rows": partial_rows,
        "partial_windows": partial_windows,
        "smoke_or_partial": partial_rows or partial_windows,
        "seed": cli.seed,
        "batch_size": cli.batch_size,
        "device": str(device),
        "inference_backend": backend,
        "torch_version": torch.__version__,
        "inference_precision": "float32, no autocast",
        "clean_input_exact_match": True,
        "clean_reference_check": reference_check,
        "e0_definition": "sampled energy at argmax dI/dE of each clean raw spectrum",
        "window_definition": {
            "coordinate": "E - E0 in eV",
            "start_ev": cli.window_start_ev,
            "stop_ev": cli.window_stop_ev,
            "width_ev": cli.window_width_ev,
            "stride_ev": cli.window_stride_ev,
            "full_window_count": full_window_count,
            "evaluated_window_count": len(windows),
            "region_aligned": bool(cli.region_aligned_windows),
            "selection": "fixed from CLI before inference; never selected using test labels",
        },
        "operational_region_labels": {
            "pre_edge": "window center < -2 eV",
            "edge_white_line": "-2 <= window center <= 15 eV",
            "post_edge": "window center > 15 eV",
            "warning": (
                "operational labels are not element-specific electronic-state assignments"
            ),
        },
        "interventions": {
            "selected": list(cli.interventions),
            "endpoint_linear": (
                "straight line between the nearest clean model-intensity samples "
                "outside each window"
            ),
            "element_train_median": (
                "same-element, E0-aligned, parent-structure-balanced median normalized "
                "line shape from train rows only; mixed into the clean model-intensity "
                "channel using a sin^2 raised-cosine gate that is zero at both window "
                "boundaries and one at the centre"
            ),
        },
        "reference_grid": {
            "start_ev": float(median_grid[0]),
            "stop_ev": float(median_grid[-1]),
            "step_ev": cli.reference_grid_step_ev,
            "min_support_parent_structures": cli.reference_min_support,
            "element_diagnostics": reference_diagnostics,
        },
        "bootstrap": {
            "unit": "material_id",
            "paired_clean_and_perturbed": True,
            "replicates": cli.bootstrap_replicates,
            "interval": "percentile 95%",
            "score_drop_sign": "clean minus perturbed; positive means degraded performance",
        },
        "coverage_policy": {
            "any_sample": "at least one measured sample lies inside the window",
            "complete_two_flank": (
                "the window contains at least one sample, at least one clean measured "
                "sample lies below its lower boundary, and at least one lies at or "
                "above its upper boundary"
            ),
            "shared_formal_population": (
                "same rows for both methods: window samples and both flanks are present, "
                "and every window sample is supported by both interventions"
            ),
            "formal_interpretation_threshold": (
                "interpret only windows whose shared formal population covers >=0.8 of "
                "the scope and for which each method changes >=0.8 of those formal rows; "
                "all windows remain exported"
            ),
        },
        "perturbation_magnitude": (
            "L1=mean absolute and L2=RMS change in the training-preprocessed intensity "
            "channel after division by that channel's clean span; reported over the "
            "full spectrum and the window"
        ),
        "preprocessing": (
            "clean input is reconstructed exactly with the unchanged training loader; "
            "interventions then modify only odd (intensity) columns of that fixed model "
            "input. Energy columns and every out-of-window intensity value remain "
            "bitwise identical, so per-spectrum renormalization cannot spread a local "
            "intervention outside its window"
        ),
        "test_label_policy": (
            "labels are not used for E0, window locations, reference construction, or "
            "perturbation generation; they are used only for F1/R2 evaluation"
        ),
        "primary_metric": primary,
        "clean_project_metrics": clean_metrics,
        "outputs": {
            "window_summary": str(output / "window_summary.csv"),
            "element_window_summary": str(output / "element_window_summary.csv"),
            "region_summary": str(output / "region_summary.csv"),
            "condition_predictions": str(output / "<method>/<window>"),
        },
    }
    _write_json(output / "metadata.json", metadata)
    _write_json(
        output / "summary.json",
        {"metadata": metadata, "window_results": summary_rows,
         "element_window_results": element_rows, "region_results": region_rows},
    )
    print(
        f"Saved {len(summary_rows)} window conditions to {output}", flush=True
    )


if __name__ == "__main__":
    main()


