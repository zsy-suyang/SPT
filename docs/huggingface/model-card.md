---
license: apache-2.0
library_name: pytorch
language:
  - en
tags:
  - xanes
  - spectroscopy
  - materials-science
  - classification
  - regression
  - custom-code
datasets:
  - zsysuyang/XAS_SPT
---

<h1 align="center">SPT: model checkpoints</h1>

<p align="center">Checkpoints accompanying <strong>Inferring structural descriptor information from X-ray absorption spectra through adaptive multiscale learning</strong>.</p>

<p align="center">Suyang Zhong, Boying Huang, Pengwei Xu, Fanjie Xu, Yuhao Zhao,<br>Jun Cheng, Fujie Tang, Weinan E, Zhong-Qun Tian</p>

<p align="center">
  <a href="https://github.com/zsy-suyang/SPT"><img src="images/badge-code.svg" alt="GitHub code"></a>
  <a href="https://huggingface.co/datasets/zsysuyang/XAS_SPT"><img src="images/badge-data.svg" alt="Hugging Face dataset"></a>
  <a href="https://huggingface.co/zsysuyang/XAS_SPT"><img src="images/badge-weights.svg" alt="Hugging Face model weights"></a>
  <a href="LICENSE"><img src="images/badge-license.svg" alt="Repo license: Apache-2.0"></a>
</p>

<p align="center"><a href="https://github.com/zsy-suyang/SPT">Code</a> · <a href="https://huggingface.co/datasets/zsysuyang/XAS_SPT">Dataset</a> · <a href="#overview">Overview</a> · <a href="#run-an-application">Download &amp; use</a> · <a href="checkpoints/WEIGHT_INDEX.csv">Weight index</a> · <a href="https://github.com/zsy-suyang/SPT/blob/main/docs/reproduce.md">Experiment guide</a> · <a href="#citation-and-contact">Citation</a></p>

<a id="overview"></a>

## 📖 Overview

This collection provides the experiment-specific checkpoints and configurations for SPT and the comparison models used in the reported analyses. SPT combines adaptive Fourier lifting with energy-aware graph aggregation to infer electronic and geometric descriptors from XANES. Expert models predict one descriptor, while Joint models share an encoder across the applicable descriptors.

![SPT framework](images/spt-framework.png)

The framework connects energy–intensity inputs to descriptor predictions through the adaptive Fourier lifting and graph branches. The diagram is reproduced from the accompanying manuscript.

### Release

| Collection | Version | Companion code |
| --- | --- | --- |
| Model checkpoints and configurations | `2026-10-25` | SPT `0.1.1` |
| Data and reference results | [`2026-10-25`](https://huggingface.co/datasets/zsysuyang/XAS_SPT/tree/2026-10-25) | SPT `0.1.1` |

<a id="descriptor-tasks"></a>

## 📊 Descriptor tasks

The computed-spectrum benchmark comprises all 12 descriptor tasks below: eight classification tasks and four regression tasks. Expert SPT has the highest test-set point estimates among the ten compared models across these 12 tasks, using weighted F1 for classification and R² for regression. The [dataset collection](https://huggingface.co/datasets/zsysuyang/XAS_SPT) provides the reference predictions and numerical summaries.

| Absorbing sites | Classification | Regression |
| --- | --- | --- |
| Non-oxygen | OS, CN, OCN, CN2, SGN | NNRS, MOOD |
| Oxygen | CN, CN2, SGN | NNRS, NNOSM |

OS denotes oxidation state, CN first-shell coordination number, OCN oxygen coordination number, CN2 two-hop neighbourhood count, and SGN space-group number. NNRS describes first-shell bond-length dispersion, MOOD the minimum nonzero oxygen–oxygen distance, and NNOSM the mean oxidation state of neighbouring atoms. Full definitions and data fields are given in the [dataset card](https://huggingface.co/datasets/zsysuyang/XAS_SPT).

<a id="select-a-checkpoint"></a>

## 🗂️ Select a checkpoint

The [weight index](checkpoints/WEIGHT_INDEX.csv) maps each experiment and descriptor to its model ID, checkpoint and configuration. Use the designated checkpoint for the intended experiment. The [registry](configs/registry.json) resolves registered model IDs; [application settings](configs/applications.json), [experiment settings](configs/experiments.json) and [transfer configurations](configs/transfer/) select the models for each application or numerical experiment.

| Directory | Use |
| --- | --- |
| `checkpoints/benchmark/` | Computed-spectrum benchmark |
| `checkpoints/ablations/` | Module and fixed-transform comparisons |
| `checkpoints/applications/` | Measured and computed application cohorts |
| `checkpoints/robustness/` | Gaussian-noise comparisons |
| `checkpoints/interpretability/` | Spectral-band and energy-window interventions |
| `checkpoints/uncertainty/` | Uncertainty analyses |
| `checkpoints/runtime/` | Runtime measurements |
| `checkpoints/transfer/` | Source and target-domain transfer models |

The collection contains 415 indexed checkpoint files: 343 PyTorch `.pth` files and 72 serialized `.joblib` estimators. [ASSET_MANIFEST.json](ASSET_MANIFEST.json) lists paths and file sizes. Configuration snapshots are included in `configs/`; they retain the preprocessing, class maps, architecture settings and experiment associations used by the corresponding checkpoints. These models run through the SPT package, not the Transformers `AutoModel` interface or a hosted inference endpoint.

<a id="inputs-and-outputs"></a>

## 🔬 Inputs and outputs

Prepared prediction inputs contain 256 energy–intensity pairs and an absorbing-element symbol. Energy is supplied in eV. The selected configuration defines energy scaling, intensity normalization and descriptor encoding.

Classification outputs are physical class labels and class scores. Regression outputs follow the descriptor units in the associated configuration. The model outputs are electronic and geometric descriptors of the absorbing environment, not a reconstructed atomic structure.

<a id="run-an-application"></a>

## 📥 Run an application

Use this collection with SPT code version 0.1.1 and the corresponding [XAS_SPT data collection](https://huggingface.co/datasets/zsysuyang/XAS_SPT). Follow the code repository's [installation guide](https://github.com/zsy-suyang/SPT#installation), then download both releases:

```bash
git clone --branch v0.1.1 https://github.com/zsy-suyang/SPT.git
cd SPT
python -m pip install -e .
python -m pip install -U huggingface_hub

hf download zsysuyang/XAS_SPT --repo-type model \
  --revision 2026-10-25 --local-dir ../spt-models
hf download zsysuyang/XAS_SPT --repo-type dataset \
  --revision 2026-10-25 --local-dir ../spt-data
```

Merge `spt-models/checkpoints/` into the code checkout's `checkpoints/`, and retain the matching `configs/`. Merge `datasets/`, `data/`, `reference_predictions/` and `reference_results/` from `spt-data` into the checkout. Keep the code README and licence at its root. The [release guide](https://github.com/zsy-suyang/SPT/blob/main/docs/releases.md) describes the assembled layout. From the assembled checkout, run:

```bash
spt verify-assets --all-assets
spt application --cohort ni_fe_activation \
  --tasks OS CN OCN NNRS MOOD \
  --output outputs/ni_fe_activation.csv
```

The command selects the five designated Ni–Fe checkpoints and applies CPU float32 inference with four PyTorch threads. The resulting CSV contains predictions and class scores; its companion metadata records the models and execution settings. Use a new output filename for each run.

The [numerical-experiment guide](https://github.com/zsy-suyang/SPT/blob/main/docs/reproduce.md) provides commands for the computed benchmark, Gaussian noise, interventions and uncertainty analyses. Those reference runs use their documented device and arithmetic settings. [Transfer commands](https://github.com/zsy-suyang/SPT/blob/main/docs/transfer.md) are documented separately.

<a id="evaluation-and-training"></a>

## ⚙️ Evaluation and training

The data collection includes per-spectrum reference predictions and numerical summaries. Use the public experiment commands with their fixed partitions to evaluate the supplied models. Supported neural-model training configurations can also be used with `spt train`; tree-model evaluation uses the benchmark command and supplied estimators. Configuration and training-partition mappings are in the weight index and the code repository's [checkpoint guide](https://github.com/zsy-suyang/SPT/blob/main/docs/checkpoints.md).

<a id="licence"></a>

## ⚖️ Licence

The repository retains its Apache 2.0 licence for author-provided material; see [LICENSE](LICENSE). Third-party components retain their respective terms, as documented in the code repository's [third-party notices](https://github.com/zsy-suyang/SPT/blob/main/THIRD_PARTY_NOTICES.md). Load serialized checkpoints and estimators only from trusted sources.

<a id="citation-and-contact"></a>

## 📝 Citation and contact

Please cite the accompanying manuscript and identify the checkpoint release used. [CITATION.cff](CITATION.cff) contains the author information and manuscript title.

```bibtex
@misc{spt_models_2026,
  author = {Zhong, Suyang and Huang, Boying and Xu, Pengwei and Xu, Fanjie and Zhao, Yuhao and Cheng, Jun and Tang, Fujie and E, Weinan and Tian, Zhong-Qun},
  title = {SPT model checkpoints and configurations},
  year = {2026},
  version = {2026-10-25},
  url = {https://huggingface.co/zsysuyang/XAS_SPT}
}
```

For usage questions, open a [GitHub issue](https://github.com/zsy-suyang/SPT/issues) or a discussion in this repository.
