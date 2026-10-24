"""Five non-DC input-frequency interventions on frozen models."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .common import (fresh_output, load_config, load_dataset, load_model,
                     metric_values, predict, save_json, set_seed, verify_reference,
                     add_model_arguments, resolve_model_arguments)

SEED = 42
INTERVENTIONS = ("mask", "perturb", "phase_randomize", "amplitude_attenuate")

def frequency_band_slices(signal_length, num_bands=5):
    """Return near-equal contiguous slices covering all real-FFT bins."""
    if signal_length < 2:
        raise ValueError("signal_length must be at least 2")
    if num_bands < 1:
        raise ValueError("num_bands must be positive")
    num_frequency_bins = signal_length // 2 + 1
    if num_bands > num_frequency_bins:
        raise ValueError("num_bands exceeds the number of rFFT bins")
    edges = np.linspace(0, num_frequency_bins, num_bands + 1, dtype=int)
    return [slice(int(edges[i]), int(edges[i + 1])) for i in range(num_bands)]


def band_metadata(signal_length, num_bands=5, preserve_dc=False):
    """Describe bin and normalized-frequency boundaries for each band."""
    slices = frequency_band_slices(signal_length, num_bands)
    nyquist_bin = signal_length / 2.0
    rows = []
    for index, band_slice in enumerate(slices):
        if preserve_dc and band_slice.start == 0:
            band_slice = slice(1, band_slice.stop)
        rows.append(
            {
                "band": index + 1,
                "band_label": f"band_{index + 1}",
                "bin_start": band_slice.start,
                "bin_stop_exclusive": band_slice.stop,
                "normalized_frequency_start": band_slice.start / nyquist_bin,
                "normalized_frequency_stop": (band_slice.stop - 1) / nyquist_bin,
            }
        )
    return rows


def apply_rfft_intervention(
    signal,
    band,
    intervention,
    *,
    num_bands=5,
    perturb_std=0.25,
    attenuation=0.25,
    seed=SEED,
    preserve_dc=False,
):
    """Apply one intervention along the last axis and preserve signal shape.

    ``perturb_std`` scales complex Gaussian noise by the selected band's RMS
    coefficient magnitude. ``attenuation`` is the multiplicative amplitude
    retained for ``amplitude_attenuate``.
    """
    if intervention not in INTERVENTIONS:
        raise ValueError(f"unknown intervention: {intervention}")
    if not 1 <= band <= num_bands:
        raise ValueError(f"band must be in [1, {num_bands}]")
    if perturb_std < 0:
        raise ValueError("perturb_std must be non-negative")
    if not 0 <= attenuation <= 1:
        raise ValueError("attenuation must be in [0, 1]")

    tensor = torch.as_tensor(signal)
    if not tensor.is_floating_point():
        tensor = tensor.float()
    coefficients = torch.fft.rfft(tensor, dim=-1)
    band_slice = frequency_band_slices(tensor.shape[-1], num_bands)[band - 1]
    if preserve_dc and band_slice.start == 0:
        band_slice = slice(1, band_slice.stop)
    selected = coefficients[..., band_slice]
    if selected.numel() == 0:
        return tensor.clone()

    if intervention == "mask":
        selected = torch.zeros_like(selected)
    elif intervention == "amplitude_attenuate":
        selected = selected * attenuation
    elif intervention == "phase_randomize":
        generator = torch.Generator(device=tensor.device)
        generator.manual_seed(seed)
        phase = 2 * torch.pi * torch.rand(
            selected.shape,
            generator=generator,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        selected = selected.abs() * torch.polar(torch.ones_like(phase), phase)
        # The self-conjugate bins have no continuously variable phase.
        # Projecting a random complex phase to real would change their
        # magnitude, confounding a phase-only experiment with attenuation.
        if band_slice.start == 0:
            selected[..., 0] = coefficients[..., 0]
        if tensor.shape[-1] % 2 == 0 and band_slice.stop == coefficients.shape[-1]:
            selected[..., -1] = coefficients[..., -1]
    else:
        generator = torch.Generator(device=tensor.device)
        generator.manual_seed(seed)
        rms = selected.abs().square().mean(dim=-1, keepdim=True).sqrt()
        real_noise = torch.randn(
            selected.shape,
            generator=generator,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        imag_noise = torch.randn(
            selected.shape,
            generator=generator,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        selected = selected + (real_noise + 1j * imag_noise) * (
            perturb_std * rms / np.sqrt(2.0)
        )

    coefficients = coefficients.clone()
    coefficients[..., band_slice] = selected
    # DC and even-length Nyquist coefficients must be real for a real signal.
    dc_real = coefficients[..., 0].real.clone()
    coefficients[..., 0] = torch.complex(dc_real, torch.zeros_like(dc_real))
    if tensor.shape[-1] % 2 == 0:
        nyquist_real = coefficients[..., -1].real.clone()
        coefficients[..., -1] = torch.complex(
            nyquist_real, torch.zeros_like(nyquist_real)
        )
    return torch.fft.irfft(coefficients, n=tensor.shape[-1], dim=-1)


def intervene_interleaved_input(cont, band, intervention, **kwargs):
    """Intervene on intensity (odd columns), preserving energy columns exactly."""
    if cont.ndim != 2 or cont.shape[1] % 2:
        raise ValueError("cont must be a 2D interleaved energy/intensity tensor")
    output = cont.clone()
    output[:, 1::2] = apply_rfft_intervention(
        cont[:, 1::2], band, intervention, **kwargs
    )
    return output


def move_to_device(value, device):
    if isinstance(value, dict):
        return {key: item.to(device) for key, item in value.items()}
    return value.to(device)


def collect_predictions(
    model,
    loader,
    device,
    task_names,
    intervention=None,
    max_batches=None,
    **kwargs,
):
    predictions = {}
    labels = {}
    with torch.no_grad():
        for batch_index, (cate, cont, label) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            cate = cate.to(device)
            cont = cont.to(device)
            label = move_to_device(label, device)
            if intervention is not None:
                intervention_kwargs = dict(kwargs)
                if "seed" in intervention_kwargs:
                    intervention_kwargs["seed"] += batch_index
                cont = intervene_interleaved_input(
                    cont,
                    intervention=intervention,
                    **intervention_kwargs,
                )
            output, _ = model(cate, cont, None)
            output = output if isinstance(output, dict) else {task_names[0]: output}
            label = label if isinstance(label, dict) else {task_names[0]: label}
            for descriptor in output:
                predictions.setdefault(descriptor, []).append(output[descriptor].cpu())
                labels.setdefault(descriptor, []).append(label[descriptor].cpu())
    if not predictions:
        raise ValueError("No test batches were evaluated")
    return (
        {key: torch.cat(value).numpy() for key, value in predictions.items()},
        {key: torch.cat(value).numpy() for key, value in labels.items()},
    )




def main(argv=None):
    from spt.utils.execution import inference_backend
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    cli = resolve_model_arguments(parser.parse_args(argv), parser)
    if cli.batch_size < 1:
        parser.error("batch size must be positive")
    set_seed(cli.seed)
    args = load_config(cli.config)
    args.batch_size = cli.batch_size
    device = torch.device(cli.device)
    with inference_backend(args, device) as backend:
        return _run(cli, args, device, backend)


def _run(cli, args, device, backend):
    data = load_dataset(args)
    if isinstance(getattr(args, "multi_task", None), list) and args.multi_task:
        raise ValueError("Frequency comparisons require a single-task checkpoint")
    model = load_model(args, cli.checkpoint, device)
    clean_metrics, clean_frame = predict(model, data, device, cli.batch_size)
    verification = verify_reference(clean_frame, cli.reference_predictions)
    output = fresh_output(cli.output_dir)
    clean_frame.to_csv(output / "clean_predictions.csv", index=False)
    loader = torch.utils.data.DataLoader(data, batch_size=cli.batch_size, shuffle=False, num_workers=0)
    mode = getattr(args, args.task)["mode"]
    primary = "F1" if mode == "Cls" else "R2"
    records = []
    for j, operation in enumerate(INTERVENTIONS):
        for band in range(1, 6):
            values, labels = collect_predictions(model, loader, device, [args.task],
                intervention=operation, band=band, preserve_dc=True,
                seed=cli.seed + 100*j + band, perturb_std=.25, attenuation=.25)
            logits, truth = values[args.task], labels[args.task].reshape(-1)
            prediction = logits.argmax(axis=1) if mode == "Cls" else logits.reshape(-1)
            scores = metric_values(mode, truth, prediction)
            frame = pd.DataFrame({"row_index": data.indices, "label": truth, "prediction": prediction})
            frame.to_csv(output / f"{operation}_B{band}_predictions.csv", index=False)
            records.append({"task": args.task, "intervention": operation, "band": band,
                "n_spectra": len(frame), "primary_metric": primary,
                "clean_score": clean_metrics[primary], "score": scores[primary],
                "score_loss": clean_metrics[primary] - scores[primary]})
    pd.DataFrame(records).to_csv(output / "band_results.csv", index=False)
    save_json(output / "metadata.json", {"seed": cli.seed, "batch_size": cli.batch_size,
        "device": str(device), "checkpoint": str(cli.checkpoint),
        "config": str(cli.config), "clean_verification": verification,
        "inference_backend": backend,
        "bands": band_metadata(256, preserve_dc=True),
        "generator_seed": "seed + 100 * operation_index + band + batch_index",
        "row_order": np.asarray(data.indices).astype(int).tolist(),
        "scope": "input-frequency interventions, not internal AFLD coefficient ablation"})

