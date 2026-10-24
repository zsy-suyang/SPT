# Prepared-input applications

The application command selects a cohort's prepared spectra and descriptor models through `configs/applications.json`. `configs/registry.json` resolves their configurations and checkpoint paths. The [weight index](../checkpoints/WEIGHT_INDEX.csv) records the experiment mappings; see [Application model mapping](checkpoints.md#application-model-mapping) for inference and training configuration details.

## Run a cohort

```bash
spt application --cohort ni_fe_activation \
  --tasks OS CN OCN NNRS MOOD \
  --output outputs/ni_fe_activation.csv
```

Omit `--tasks` to predict every descriptor available for the selected cohort, or specify a subset as above. `SGN` is accepted as an alias for the stored `PSGO` task ID. Use a new output filename for each run.

Omit `--device` and `--batch-size` to use the cohort's registered execution settings. Both Li–Co–O cohorts, Co and Ti standards, and Fe charging use CUDA; the two Ni cohorts use CPU with four PyTorch threads, applied automatically by the command. All use float32 model inference with a batch-size setting of 128. The CUDA dependencies are given in [Assets and environment](reproduce.md#assets-and-environment).

The registry also retains the normalization precision and CUDA arithmetic settings for each cohort. Ti inference uses blocks of 34, with padding discarded before export, so its results contain only the nine supplied standards. These execution settings are applied automatically by the application command.

## Data and models

Each cohort uses `datasets/applications/<cohort>/prepared.csv` and its descriptor models in `checkpoints/applications/<cohort>/`. Source files and `provenance.json` accompany the prepared spectra. Download entries are listed under [Data and weights](../README.md#data-and-weights).

| Cohort ID | Input | Spectra | Registered descriptors |
| --- | --- | ---: | --- |
| `lco_computed_co_sites` | Computed Li–Co–O Co sites | 17 | OS, CN, OCN, CN2, SGN, NNRS, MOOD |
| `lco_measured_charging` | Measured Li–Co–O charging | 11 | OS |
| `co_lisa_compounds` | Measured Co standards | 8 | OS, CN, OCN, CN2, SGN, NNRS, MOOD |
| `ti_nist_standards` | Measured Ti standards | 9 | OS, CN |
| `fe_charging` | Measured Fe charging | 36 | OS, CN, OCN, CN2, SGN, NNRS, MOOD |
| `ni_fe_activation` | Six measured activation states and two references | 8 | OS, CN, OCN, NNRS, MOOD |
| `ni_first_sodiation` | Measured first-sodiation states | 6 | OS, CN, OCN, NNRS, MOOD |

For example, Ni–Fe activation OCN uses model ID `app-ni_fe_activation-OCN`, configuration `configs/experiments/app-ni_fe_activation-OCN.yaml` and weight file `checkpoints/applications/ni_fe_activation/SPT_ni_fe_activation_OCN.pth`.

The associated energy-reference, broadening and input-support controls have separate [measurement-control commands](reproduce.md#measurement-and-input-controls).

## Custom prepared spectra

Use `--input <path.csv>` to supply new prepared spectra with the selected cohort's models; omit it to use the supplied cohort. The [long CSV format](datasets.md#prediction-csv) contains `spectrum_id`, `energy_eV`, `intensity` and `absorber`, with 256 rows per spectrum. Complete the energy referencing and resampling described in [Application spectra and sources](application-data.md) before inference. The command then applies the model's saved normalization without further energy alignment or interpolation.

## Outputs

The output records physical class labels or regression values, class scores and the model ID. A companion metadata file records the input, weight and configuration paths alongside execution settings. Class scores describe the model's output across its trained categories.

For Ni–Fe activation, `reference_predictions.csv` contains the five-descriptor readouts. `reference_predictions_long.csv` includes all class scores, and `reference_predictions_long.metadata.json` identifies its inputs, models and execution settings. Check the eight-spectrum reference case with:

```bash
python -m pytest tests/test_ni_activation_reference.py -q
```
