# Reproducing the numerical experiments

These commands run model inference, spectral interventions and statistical analyses, exporting per-spectrum predictions and numerical summaries.

## Assets and environment

Install the package and its test dependencies from the checkout:

```bash
python -m pip install -e '.[dev]'
spt list-models
spt verify-assets --model expert-metal-OS
spt verify-assets --all-assets
python -m pytest tests/test_experiments.py
```

The registry, `configs/registry.json`, maps each model to its configuration, checkpoint, reference predictions and computed dataset/split pair. Select experiment-specific models through `configs/experiments.json`; application data and models are mapped in `configs/applications.json`. See the [README](../README.md) for data and weight access.

The reported computed-benchmark and noise evaluations use Python 3.10, PyTorch 2.0.0 with CUDA 11.8, float32 and a GeForce RTX 4090 D. Install the matching dependencies in a Python 3.10 environment:

```bash
python -m pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -c constraints-cuda.txt -e '.[dev]'
```

The PyTorch wheel is listed in the [official previous-version instructions](https://pytorch.org/get-started/previous-versions/#v200). The experiment configurations specify the CUDA arithmetic settings, and reference checks compare the generated predictions with the supplied results. General prediction also supports CPU; prepared applications use their cohort-specific execution settings. Load checkpoints and serialized estimators only from trusted sources.

For the tested Python 3.12 CPU dependencies, install with `python -m pip install -c constraints-cpu.txt -e '.[dev]'`. These constraints specify the local replay environment, separately from the original CUDA timing environment.

Run commands from the repository root. Registry-based experiments also accept `--root PATH_TO_CHECKOUT`. Use a new or empty output directory for each run.

```bash
spt experiment --help
spt experiment gaussian --help
spt experiment energy-windows --help
```

The commands can also be invoked as `python -m spt.experiments COMMAND ...`.

## Computed benchmark and architectural controls

There are seven non-oxygen and five oxygen descriptor tasks. SGN uses the key `PSGO` in configuration files. Each model configuration specifies its preprocessing, class map, task eligibility, structure-level split and architecture.

The benchmark entry point supports frozen neural models and tree estimators, using each model's own configuration and saved test partition:

```bash
spt experiment benchmark --root . \
  --model-id expert-metal-OS baseline-random_forest-metal-OS \
    baseline-direct_nn-metal-OS baseline-amformer-metal-OS \
  --device cuda:0 --batch-size 128 --seed 42 \
  --output-dir outputs/benchmark/metal-OS
```

The same interface accepts fixed-transform, Bare, Graph, Joint and all four tree-baseline registry IDs. XGBoost requires its optional dependency (`pip install '.[trees]'`). Outputs include predictions, metrics and comparisons with the reference predictions. Joint models use task-specific valid-row masks. For a custom model without reference predictions, use `--allow-missing-reference`.

For Joint reference replay, use the CUDA command below. The two Joint configurations retain the original TF32 settings for matrix multiplication and cuDNN. Reference replay uses Python 3.10, PyTorch 2.0.0+cu118 and a GeForce RTX 4090 D, with batches of 128. Joint metrics use the original float32 TorchMetrics aggregation; `audit_float64_*` columns provide an additional float64 calculation from the same predictions. The archived evaluation metrics are retained under `joint-metal` and `joint-oxygen` in `reference_results/benchmark_metrics.json`.

```bash
spt experiment benchmark --root . --model-id joint-metal joint-oxygen \
  --device cuda:0 --batch-size 128 --seed 42 \
  --output-dir outputs/benchmark/joint
```

The lower-level neural interface is also available:

```bash
spt evaluate --config configs/models/expert-metal-OS.yaml \
  --checkpoint checkpoints/benchmark/spt_expert/non_oxygen/SPT_OS.pth \
  --split test --device cuda:0 --num-workers 0 \
  --output-dir outputs/benchmark/expert-metal-OS
```

For supported neural-model training configurations, use `spt train --config CONFIG --checkpoint none --output-dir NEW_RUN` with the saved partition and checkpoint-selection rule. This training command does not accept tree-model configurations; evaluate the supplied tree estimators with the benchmark command above. Use the supplied checkpoints to evaluate the reported models. Target-domain transfer commands are described in [transfer.md](transfer.md).

## Matched clipped-Gaussian comparison

The supported methods are SPT, RF, NN–XANES (`direct_nn`) and AMFormer. Each task is evaluated on clean input and six amplitudes, `A = 0.05, 0.10, 0.15, 0.20, 0.25, 0.30`. Noise is added to raw intensity before the original training normalization. At each amplitude, Python's seeded Gaussian generator produces a shared 256-point float32 vector with standard deviation `A/3`, clipped to `[-A, A]`. Energy coordinates are unchanged. These amplitudes are in raw input-intensity units, not percentages.

One-task, four-model example:

```bash
spt experiment gaussian --root . \
  --model-id gaussian-expert-metal-OS gaussian-baseline-random_forest-metal-OS \
    gaussian-baseline-direct_nn-metal-OS gaussian-baseline-amformer-metal-OS \
  --device cuda:0 --batch-size 128 --seed 42 \
  --output-dir outputs/gaussian-metal-OS
```

To run all 12 tasks and four methods in a single job, select the dedicated Gaussian experiment IDs:

```python
from spt.experiments import main
from spt.registry import load_registry
import json

root, registry = load_registry()
experiments = json.loads((root / "configs/experiments.json").read_text())
ids = experiments["experiments"]["gaussian"]["model_ids"]
assert len(ids) == 48, "Expected 12 tasks for each of four methods"
main(["gaussian", "--root", str(root), "--model-id", *ids,
      "--device", "cuda:0", "--batch-size", "128", "--seed", "42",
      "--output-dir", "outputs/gaussian-all-tasks"])
```

The runner compares clean predictions with the reference and checks that all methods use the same row order, raw spectra, labels and normalized inputs for each task. It writes predictions for every condition, `gaussian_results.csv` and execution metadata. Classification results include weighted F1 and accuracy; regression results include R², MAE, MSE and RMSE. F1 is stored on a 0–1 scale; multiply a score difference by 100 to express percentage points.

The reported comparison uses one noise realization at each amplitude with seed 42. Use `--seed` to evaluate another realization.

## Paired parent-structure bootstrap

This stage recomputes uncertainty from saved predictions, without running a model or retraining. Use the candidate and comparator prediction tables for the same task, and the computed dataset to recover each row's parent structure.

```bash
spt experiment bootstrap \
  --predictions reference_predictions/expert-metal-OS.csv \
  --reference-predictions reference_predictions/baseline-direct_nn-metal-OS.csv \
  --data-file datasets/computed/metal.csv --group-column material_id \
  --metric f1 --iterations 2000 --seed 42 \
  --output outputs/bootstrap/metal-OS-vs-direct-nn.json
```

Select the comparator and iteration count for the analysis being reproduced. Supported metrics are `f1` (weighted), `accuracy`, `mae` and `r2`.

The procedure resamples parent structures with replacement, retaining all spectra of each sampled structure. Paired files must have identical row identities and labels. Outputs are pointwise percentile intervals for fixed predictions and a fixed comparator. For MAE, a negative candidate-minus-reference difference favours the candidate; for F1 and R², a positive difference does.

## Input-frequency interventions

```bash
spt experiment bands --model-id bands-expert-metal-OS --root . \
  --device cuda:0 --batch-size 128 --seed 42 \
  --output-dir outputs/bands/metal-OS
```

This runs masking, RMS-scaled complex perturbation, phase randomization and amplitude attenuation in five input rFFT bands, retaining the DC coefficient and energy channel. For a 256-point spectrum, the non-DC bin ranges are `[1,25)`, `[25,51)`, `[51,77)`, `[77,103)` and `[103,129)`. The generator seed is `42 + 100*j + b + q` for operation index `j`, one-based band `b` and zero-based batch index `q`. Use batch size 128 to reproduce the reported stochastic realizations.

Outputs include per-spectrum predictions for each condition and unscaled task-score losses. Apply within-task scaling when plotting the frequency-sensitivity heatmap. The interventions act on input frequency content.

## Energy-window interventions

```bash
spt experiment energy-windows --model-id energy_windows-expert-metal-OS --root . \
  --region-aligned-windows --device cuda:0 --batch-size 128 --seed 42 \
  --window-start-ev -15 --window-stop-ev 45 \
  --window-width-ev 5 --window-stride-ev 2.5 \
  --bootstrap-replicates 2000 \
  --output-dir outputs/energy-windows/metal/OS
```

Use `--region-aligned-windows` for the reported 22-window analysis. Windows are tiled within the operational pre-edge, edge/white-line and post-edge regions, without crossing the −2 or 15 eV boundaries. Without this flag, the command uses a uniform 23-window grid.

The two interventions are endpoint-linear inpainting and element-specific training-median replacement. E₀ is estimated independently from each clean spectrum. The median reference uses the fixed training partition, with parent train/test overlap checked. Both methods are compared on the same eligible rows with adequate energy support. Outputs include predictions, coverage, intervention magnitudes and paired parent-bootstrap results.

For a quick setup check, `--max-batches` and `--max-windows` limit the run to a partial analysis.

## Unchanged-coefficient AFLD round trip

```bash
spt experiment roundtrip --model-id expert-metal-OS --root . \
  --element Co --samples 32 --seed 42 \
  --output-dir outputs/afld-roundtrip
```

The runner selects the first 32 distinct-material Co spectra in the saved test-prediction order. It captures the inputs to each AFLD spectral branch, applies analysis followed by synthesis with unchanged coefficients, and exports per-sample absolute and relative reconstruction errors. Both float32 and float64 are tested; the float64 test promotes the learned float32 parameters and captured features. This measures reconstruction by the AFLD analysis/synthesis pair.

## Monte Carlo dropout and calibration

```bash
spt experiment uncertainty --model-id uncertainty-expert-metal-OS --root . \
  --passes 30 --batch-size 128 --seed 42 --split test --device cuda:0 \
  --output-dir outputs/uncertainty/metal-OS
```

The runner enables `nn.Dropout` while keeping other modules in evaluation mode. It supports single-task and Joint outputs, with the saved per-task masks for Joint models. Outputs include mean predictions, classification entropy and mutual information, or regression prediction standard deviation. Error-detection AUROC uses classification error or the upper quartile of absolute regression error. Use the recorded batch size and seed to reproduce the dropout realization.

The data package includes the frozen validation and test MC outputs for all 12 tasks. Recompute the classification and regression calibration tables without model inference:

```bash
spt experiment calibration --root . --output-dir outputs/calibration
```

This applies the SI's validation order statistic at nominal 95% coverage, retains empty classification sets, and checks the summaries against `reference_results/uncertainty_*.csv`. For a new MC run, repeat the uncertainty command with `--split val` and a separate output directory, then summarize the pair:

```bash
spt experiment calibration \
  --validation outputs/uncertainty/metal-OS-val/OS_uncertainty.csv \
  --test outputs/uncertainty/metal-OS/OS_uncertainty.csv \
  --output-dir outputs/calibration/metal-OS
```

## Runtime

```bash
spt experiment runtime --root . --site metal oxygen \
  --repeats 5 --warmup 3 --inference-calls 3 --training-calls 2 \
  --measure-bundles --output-dir outputs/runtime
```

This CUDA-only protocol uses the runtime model IDs in `configs/experiments.json` and prepared GPU-resident inputs at batch sizes 1 and 128. It compares sequential Experts and Joint inference on identical eligible input rows. Latency is measured per forward call; throughput is batch size divided by latency.

Training timing includes loss evaluation, backward propagation, AdamW and EMA updates on an in-memory model, leaving the supplied checkpoint unchanged. Estimated per-epoch computation is the timed update cost multiplied by the configured number of batches. Metadata records the device, library versions, synchronization and timing scope.

## Measured applications and transfer

Prepared measured and computed application spectra use the cohort-specific mappings in [applications.md](applications.md). Dataset contents and preparation protocols are described in [application-data.md](application-data.md). Application inference starts from the supplied 256-point inputs.

The transfer-specific models, comparison baselines, parent-excluded source data, fixed target partition, training budgets, initialization and final-epoch evaluation are described in [transfer.md](transfer.md).

## Measurement and input controls

The following suites run the additional controls reported in the SI, using the corresponding fixed inputs and models:

```bash
spt experiment measurement-controls --suite computed --output outputs/controls/computed
spt experiment measurement-controls --suite co --output outputs/controls/co
spt experiment measurement-controls --suite fe --output outputs/controls/fe
spt experiment measurement-controls --suite lco --output outputs/controls/lco
spt experiment measurement-controls --suite ni --output outputs/controls/ni
```

The `computed` suite applies fixed-grid energy shifts and Gaussian broadening to the computed test spectra, retaining the tables' float32 TorchMetrics aggregation. The `co` suite covers fixed-grid signal shifts, broadening and intensity drift; `fe` covers energy-coordinate offsets, broadening and E₀ neutralization; `lco` covers fixed alignment-anchor shifts and pre-edge extensions. The `ni` suite uses the four reported combinations of energy grid and foil referencing for the Dry and 4 h NNRS/MOOD readouts. These are the existing SI controls, separate from the four-model Gaussian-noise comparison above.

The computed, Co, Fe and Li–Co–O suites use CUDA; the Ni suite uses CPU. Retain these registered defaults for comparison with the supplied results. Use the CUDA environment described above and a new output directory. To run from outside the checkout, add `--project-root PATH_TO_CHECKOUT`.

Ni NNRS/MOOD replay uses an absolute tolerance of `1e-5` Å (`rtol=0`), shared with the Ni–Fe application regression test. `verification.json` records the maximum numerical difference, tolerance and execution settings; row identities must match and predictions must be finite. This tolerance concerns float32 numerical reproducibility, not experimental uncertainty. Other control suites retain their existing tolerances.

## Tests

Run `python -m pytest -q` for software and numerical checks covering noise generation, paired sampling, row identity, output handling, intervention locality, energy-window coverage, FFT/DC handling, dropout mode and timing calls. With the Ni–Fe assets installed, the suite also replays the five reported descriptors on its eight prepared spectra.
