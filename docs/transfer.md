# Disordered-oxide transfer experiments

The manuscript comparison uses one fixed parent-structure partition of 399 target spectra: a pool of 285 training spectra and an evaluation cohort of 114 spectra. Its nested training budgets contain 29, 72 and 143 spectra. The relevant tasks are OS, CN and OCN.

Three task-specific SPT source models were trained after removing the known target parent structures from the computed source table. Each transfer experiment has a corresponding source configuration, target configuration and checkpoint.

## Data and models

The case catalogue is [`configs/transfer/cases.json`](../configs/transfer/cases.json). Input and checkpoint paths are recorded there and in [`configs/transfer/assets.json`](../configs/transfer/assets.json).

| Data or model | Location |
| --- | --- |
| Source training configurations | `configs/transfer/transfer-source-<task>.yaml` |
| Source zero-shot evaluation configurations | `configs/transfer/transfer-source-<task>-evaluate.yaml` |
| Parent-excluded computed source table | `datasets/transfer/source/metal_parent_excluded.csv` |
| Complete target tables | `datasets/transfer/target/OS.csv`, `CN.csv`, `OCN.csv` |
| Case-specific fitting tables | `datasets/transfer/train/<task>/<scratch\|tl>_<budget>.csv` |
| Frozen spectral budgets and parent assignments | `data/splits/transfer/manifest.json` |
| Source weights | `checkpoints/transfer/<task>/source.pth` |
| Final target weights | `checkpoints/transfer/<task>/<scratch\|tl>_<budget>.pth` |
| Experiment-specific source and initialization weights | `checkpoints/transfer/<task>/tl_<budget>_source.pth`, `tl_<budget>_initializer.pth` |
| Frozen predictions | `reference_predictions/transfer/<task>/` |

Here `<task>` is OS, CN or OCN and `<budget>` is 10, 25 or 50. Each experiment has its own data and model files. The source checkpoint initializes transfer learning, the initializer records the starting target-model state, and the final checkpoint is used for target evaluation. Their paths are listed in `cases.json`.

## Frozen evaluation

The evaluation entry point uses every spectrum in the selected fixed cohort and decodes predictions using the checkpoint's physical-class map. Its metrics include all ground-truth classes present in that cohort.

```bash
python -m spt.experiments.transfer evaluate \
  --config configs/transfer/transfer-source-CN-evaluate.yaml \
  --checkpoint checkpoints/transfer/CN/source.pth \
  --data-file datasets/transfer/target/CN.csv \
  --manifest data/splits/transfer/manifest.json \
  --cohort test \
  --expected-rows 114 \
  --reference-predictions reference_predictions/transfer/CN/source_test.csv \
  --output-dir outputs/transfer/CN/source-test-evaluation
```

This command evaluates the source model without target fitting. The evaluation configuration inherits the source training settings and specifies the real-inverse-FFT endpoint handling used for zero-shot evaluation. The runner uses float32 with TF32 disabled. For all-target zero-shot evaluation, set `--cohort all --expected-rows 399` and use `source_all.csv` as the reference. For a fitted target model, use its experiment-specific configuration, final checkpoint and reference, for example:

```bash
python -m spt.experiments.transfer evaluate \
  --config configs/transfer/transfer-CN-tl-10.yaml \
  --checkpoint checkpoints/transfer/CN/tl_10.pth \
  --data-file datasets/transfer/target/CN.csv \
  --manifest data/splits/transfer/manifest.json \
  --cohort test --expected-rows 114 \
  --reference-predictions reference_predictions/transfer/CN/tl_10.csv \
  --reference-atol 5e-5 \
  --output-dir outputs/transfer/CN/tl-10-evaluation
```

The input table retains its original index, unique `source_row`, parent-group identifier, absorber, spectrum and physical descriptor labels. The manifest supplies the parent-group column and fixed cohort. Optional `--reference-predictions` checks identical row identities, truth labels and hard predictions against a frozen prediction table. The output contains predictions, weighted and macro F1, accuracy, evaluable weighted AUROC and parent-group metrics, together with input paths.

Reference replay requires identical row identities, labels, predicted classes and weighted F1. Probability scores are checked separately: the output records their maximum absolute difference and whether they meet `--reference-atol 5e-6 --reference-rtol 1e-5` (the fitted-target example uses `atol=5e-5`). Add `--strict-probabilities` to make an out-of-tolerance score difference an error. Invalid or non-finite probability arrays remain errors in either mode.

Float32 results need not be bitwise identical across devices, numerical backends or PyTorch versions, as described in the [PyTorch numerical-accuracy documentation](https://docs.pytorch.org/docs/2.7/notes/numerical_accuracy.html). Class-score agreement is therefore reported separately from reproduction of class predictions and weighted F1; execution settings are saved with the results.

## Comparison baselines

NN–XANES, AMFormer and RF use the same target partition and budgets. Their frozen weights, effective configurations, source-model links and reference predictions are mapped in [`configs/transfer/baselines.json`](../configs/transfer/baselines.json) and the central weight index.

```bash
spt experiment transfer-baselines summarize --project-root . \
  --output-dir outputs/transfer/baseline-tables
spt experiment transfer-baselines evaluate --project-root . --device cpu \
  --output-dir outputs/transfer/baseline-replay
```

`summarize` checks all 63 published baseline scores from the supplied predictions. `evaluate` replays all 54 source/fitted models on the 114-spectrum test cohort, checking row identities, physical classes and weighted F1. Use `list` to display case IDs or `--model-ids ID ...` to select cases. Source models also support `--cohort all` for the 399-spectrum cohort. Saved inference blocks are retained where specified; class-score deviations are recorded separately, with `--strict-probabilities` available for score-level checks.

## SPT training configurations

The comparison evaluates complete, task- and budget-specific transfer and Scratch workflows on the same training spectra and held-out parent-structure test cohort. Transfer initializes the SPT encoder and projection head from the corresponding source model, retaining its source class space. Scratch trains all parameters without source weights. Each workflow uses the optimization, trainable-layer, loss and initialization settings listed below.

All target runs use seed 42, AdamW, 100 epochs, weight decay 0.001 and a cosine schedule with linear warm-up over the first 5% of training steps. Training uses no validation loader and saves the final raw checkpoint for evaluation. Transfer uses batches of 16 throughout. The listed learning rates are base values before warm-up and cosine decay. For gradual unfreezing, the encoder learning rate applies after the initial head-only stage.

| Task | Training spectra | Transfer schedule | Transfer head / encoder learning rate | Transfer loss | Scratch learning rate / batch size | Scratch loss |
| --- | ---: | --- | --- | --- | --- | --- |
| OS | 29 | Head only for 5 epochs, then all parameters | 3e-4 / 9e-5 | CE | 6e-4 / 64 | CE |
| OS | 72 | Head only for 5 epochs, then all parameters | 2e-4 / 2e-5 | CE | 6e-4 / 64 | CE |
| OS | 143 | Head only for 10 epochs, then all parameters | 4e-4 / 4e-4 | CE | 6e-4 / 64 | CE |
| CN | 29 | Projection head only for all 100 epochs | 9e-5 / 0 (frozen) | CE + adjacent CE (1.0) | 3e-4 / 16 | Smoothed CE + distance (0.2) |
| CN | 72 | Head only for 10 epochs, then all parameters | 3e-4 / 3e-5 | CE | 1e-4 / 64 | CE |
| CN | 143 | Head only for 5 epochs, then all parameters | 3e-4 / 9e-5 | CE | 3e-4 / 16 | Smoothed CE + distance (0.2) |
| OCN | 29 | Head only for 1 epoch, then all parameters | 3e-4 / 3e-5 | CE + source anchor (0.001) | 3e-4 / 16 | Smoothed CE + distance (0.2) |
| OCN | 72 | Head only for 10 epochs, then all parameters | 3e-4 / 9e-5 | CE | 1e-4 / 64 | CE |
| OCN | 143 | Head only for 5 epochs, then all parameters | 3e-4 / 9e-5 | CE | 3e-4 / 16 | Smoothed CE + distance (0.2) |

CE denotes multiclass cross-entropy. Smoothed CE uses label smoothing of 0.01; all transfer runs and the plain-CE Scratch runs use no label smoothing. The adjacent-class term averages binary cross-entropy between the true class and each available physical count neighbour at ±1, with coefficient 1.0. The distance term is the predicted-probability-weighted absolute distance from the true physical class, divided by the full class-value range, with coefficient 0.2. Source anchoring adds `0.001 / 2` times the summed squared distance of trainable non-projection parameters from their source values. It is used only for OCN with 29 spectra. These terms affect training, not the physical-class decoding or evaluation metrics.

For all OS Scratch budgets, the randomly initialized categorical embedding is multiplied by 0.1 (`spt_scratch_init: variance_balanced`). CN and OCN Scratch use the standard initialization. Scratch trains all parameters at its listed base learning rate. The original projection is the only trainable head during transfer freezing, and frozen encoder modules remain in evaluation mode. The exact configurations are `configs/transfer/transfer-<task>-<scratch|tl>-<budget>.yaml`, where budgets 10, 25 and 50 correspond to 29, 72 and 143 spectra.

For Scratch, supply the original fitting table and the frozen row-budget manifest:

```bash
python -m spt.experiments.transfer train \
  --config configs/transfer/transfer-CN-scratch-10.yaml \
  --data-file datasets/transfer/train/CN/scratch_10.csv \
  --manifest data/splits/transfer/manifest.json \
  --budget 10 \
  --output-dir outputs/transfer/CN/scratch-10-retrain
```

For transfer, use the target configuration and its matching source configuration and checkpoint:

```bash
python -m spt.experiments.transfer train \
  --config configs/transfer/transfer-CN-tl-10.yaml \
  --data-file datasets/transfer/train/CN/tl_10.csv \
  --manifest data/splits/transfer/manifest.json \
  --budget 10 \
  --source-config configs/transfer/transfer-source-CN.yaml \
  --source-checkpoint checkpoints/transfer/CN/tl_10_source.pth \
  --output-dir outputs/transfer/CN/tl-10-retrain
```

The fitting table must contain the saved row identifiers. The entry point orders them by the manifest and verifies their parent assignments, then trains with the configured batch size, epochs, loss, learning rates and initialization. It saves the final raw checkpoint for evaluation with the commands above. Use the supplied final weights to evaluate the reported models directly.

The parent-excluded source model can be retrained through the standard training entry point:

```bash
spt train --config configs/transfer/transfer-source-CN.yaml \
  --output-dir outputs/transfer/CN/source-retrain
```

See the [README](../README.md) for data and weight access. Run `spt verify-assets --all-assets` to check that the required files are present.
