# Spectral Pattern Translator (SPT)

Code for **Adaptive Fourier lifting for electronic and geometric descriptor inference from XANES**.

SPT combines input-dependent Fourier lifting with energy-aware graph aggregation to predict electronic and geometric descriptors of an absorbing environment from X-ray absorption near-edge structure (XANES). Expert models predict one descriptor; Joint models share an encoder and return the descriptors available for a site family.

## Installation

Use Python 3.10–3.12 and a PyTorch build appropriate for your CPU or CUDA environment. From the repository root, install SPT in your Python environment:

```bash
python -m pip install -e .
spt --help
```

For the reported benchmark and noise experiments, use the Python 3.10 / CUDA 11.8 environment in [Reproducing the numerical experiments](docs/reproduce.md#assets-and-environment), with dependencies pinned in `constraints-cuda.txt`. Prepared applications use the execution settings registered for each cohort.

To install the test dependencies and run the checks:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

The tests cover input preparation, model operations and experiment protocols. With the Ni–Fe data and weights installed, they also replay its eight-spectrum application.

## Data and weights

- Data repository: [zsysuyang/XAS_SPT](https://huggingface.co/datasets/zsysuyang/XAS_SPT). The reorganized bundle for this code version is pending upload.
- Model weights: a separate Hugging Face bundle is being prepared; its repository link will be added after upload.

Extract data into `datasets/` and `reference_predictions/`, and weights into `checkpoints/`, preserving the package paths. [DATA_INDEX.csv](datasets/DATA_INDEX.csv) lists the data; [WEIGHT_INDEX.csv](checkpoints/WEIGHT_INDEX.csv) maps each experiment to its configurations and checkpoints. Reference predictions support numerical checks and bootstrap analyses. The code includes the indices and summary tables; large assets are downloaded separately.

| Experiment | Data and model locations | Guide |
| --- | --- | --- |
| Computed benchmark | `datasets/computed/`, `checkpoints/benchmark/` | [Numerical experiments](docs/reproduce.md#computed-benchmark-and-architectural-controls) |
| Architectural controls | `datasets/computed/`, `checkpoints/ablations/` | [Numerical experiments](docs/reproduce.md#computed-benchmark-and-architectural-controls) |
| Gaussian noise comparison | `datasets/computed/`, `checkpoints/robustness/` | [Noise comparison](docs/reproduce.md#matched-clipped-gaussian-comparison) |
| Measurement and input controls | Computed and application inputs with their registered models | [SI controls](docs/reproduce.md#measurement-and-input-controls) |
| Spectral interventions | `datasets/computed/`, `checkpoints/interpretability/` | [Interventions](docs/reproduce.md#input-frequency-interventions) |
| Uncertainty estimation | `datasets/computed/`, `checkpoints/uncertainty/` | [Monte Carlo dropout](docs/reproduce.md#monte-carlo-dropout-and-calibration) |
| Runtime measurements | `datasets/computed/`, `checkpoints/runtime/` | [Runtime](docs/reproduce.md#runtime) |
| Measured and computed applications | `datasets/applications/`, `checkpoints/applications/` | [Applications](docs/applications.md) |
| Transfer learning | `datasets/transfer/`, `checkpoints/transfer/` | [Transfer experiments](docs/transfer.md) |

The [model registry](configs/registry.json) specifies each model's configuration and checkpoint. `metal` identifies non-oxygen absorbing sites; `oxygen` identifies oxygen absorbing sites. Fixed computed-data partitions are in `data/splits/final/`.

After combining the code checkout with the data and weight downloads, the main directories are:

```text
SPT/
├── spt/                    # Model, training and inference code
├── configs/                # Model and experiment configurations
├── data/                   # Fixed partitions and asset manifest
├── datasets/               # Computed and measured inputs (data download)
├── reference_predictions/  # Per-spectrum results (data download)
├── reference_results/      # Numerical summaries
├── checkpoints/            # Experiment-specific weights (weight download)
├── examples/               # Small synthetic input example
├── docs/                   # Usage and numerical-experiment guides
└── tests/                  # Software and reference-replay tests
```

See [Download layout and versions](docs/releases.md) for how the three repositories fit together.

```bash
spt list-models
spt verify-assets --all-assets
```

## Predict descriptors

Run a prepared application with its corresponding models:

```bash
spt application --cohort ni_fe_activation \
  --tasks OS CN OCN NNRS MOOD \
  --output outputs/ni_fe_activation.csv
```

The command uses the cohort's registered execution settings: CPU for the two Ni applications and CUDA for the Li–Co–O, Co, Ti and Fe applications. See [Application commands](docs/applications.md) for details.

For a custom input CSV, select a model from the registry. This example uses the synthetic spectrum supplied with the code to demonstrate the input format:

```bash
spt predict --model expert-metal-OS \
  --input examples/spectrum.csv \
  --output outputs/predictions.csv
```

The CSV contains `energy_eV`, `intensity` and `absorber` columns, with 256 ordered points per spectrum and absolute positive energies in eV. An optional `spectrum_id` column distinguishes multiple spectra. See [Datasets](docs/datasets.md) for input preparation and [Application spectra](docs/application-data.md) for the measured-data sources and energy referencing.

## Run the experiments

[Reproducing the numerical experiments](docs/reproduce.md) provides commands for benchmarking, retraining, noise tests, spectral interventions, uncertainty estimation and runtime measurements. Use each experiment's listed configuration, data partition and model. [Checkpoints](docs/checkpoints.md) explains the model mapping, and [Architecture](docs/architecture.md) describes the encoder and descriptor heads.

Run commands from the repository root. Multi-line examples use Bash syntax; in PowerShell, enter each command on one line or replace the line-continuation character with a backtick.

## Repository layout

```text
spt/                   Model, training, evaluation and analysis code
configs/               Model registry and experiment configurations
data/splits/final/     Frozen parent-structure partition manifests
datasets/              External research datasets
checkpoints/           External trained weights
examples/              Small input examples
tests/                 Software and numerical checks
docs/                  Data, checkpoint and architecture notes
reference_predictions/ Reference prediction tables
reference_results/     Benchmark summary tables
```

## Citation

Please cite the accompanying manuscript using [CITATION.cff](CITATION.cff).

## License

The repository's source-code license is [Apache 2.0](LICENSE). Third-party software and adapted components retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This license does not relicense third-party spectra or structures. Data and checkpoint reuse terms will accompany their external release records.
