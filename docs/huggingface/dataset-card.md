---
pretty_name: SPT XANES Dataset
license: cc-by-nc-sa-4.0
language:
  - en
task_categories:
  - tabular-classification
  - tabular-regression
size_categories:
  - 10K<n<100K
tags:
  - xanes
  - x-ray-absorption-spectroscopy
  - materials-science
  - chemistry
  - structure-property-prediction
  - scientific-machine-learning
configs:
  - config_name: computed_non_oxygen
    default: true
    data_files:
      - split: full
        path: datasets/computed/metal.csv
  - config_name: computed_oxygen
    data_files:
      - split: full
        path: datasets/computed/oxygen.csv
---

<h1 align="center">SPT XANES Dataset</h1>

<p align="center"><strong>Inferring structural descriptor information from X-ray absorption spectra through adaptive multiscale learning</strong></p>

<p align="center">Suyang Zhong, Boying Huang, Pengwei Xu, Fanjie Xu, Yuhao Zhao,<br>Jun Cheng, Fujie Tang, Weinan E, Zhong-Qun Tian</p>

<p align="center">
  <a href="https://github.com/zsy-suyang/SPT"><img src="images/code-badge.svg" alt="GitHub code"></a>
  <a href="https://huggingface.co/datasets/zsysuyang/XAS_SPT"><img src="images/data-badge.svg" alt="Hugging Face dataset"></a>
  <a href="https://huggingface.co/zsysuyang/XAS_SPT"><img src="images/weights-badge.svg" alt="Hugging Face model weights"></a>
  <a href="LICENSE"><img src="images/license-badge.svg" alt="Author-owned material: CC BY-NC-SA 4.0"></a>
</p>

<p align="center"><a href="https://github.com/zsy-suyang/SPT">Code &amp; experiments</a> · <a href="#dataset-overview">Overview</a> · <a href="#download-and-use">Download &amp; use</a> · <a href="datasets/DATA_INDEX.csv">Data index</a> · <a href="#citation">Citation</a></p>

## ⚠️ Usage Restriction

> Author-owned data and documentation in this release are licensed under [CC BY-NC-SA 4.0](LICENSE), for **non-commercial use only**.
> Commercial use requires separate permission from the relevant rights holders.
> Third-party materials retain their original licences; see [Licence and attribution](#licence-and-attribution).

<a id="dataset-overview"></a>

## 📖 Dataset overview

This is the dataset collection for the Spectral Pattern Translator (SPT), which infers electronic, local-geometric and crystallographic descriptors from X-ray absorption near-edge structure (XANES). It contains computed oxide spectra with structure-derived labels, prepared measured spectra with their source information, fixed data partitions and model reference predictions. The release supports the benchmark, spectral-control, application and transfer-learning experiments in the accompanying manuscript.

| Collection | Spectra | Purpose |
| --- | ---: | --- |
| Computed non-oxygen sites | 26,870 | Seven descriptor tasks; Li, Co, Mn, P, Fe, Ni and Ti absorbers |
| Computed oxygen sites | 27,874 | Five descriptor tasks; O absorbers |
| Computed Li–Co–O application | 17 | Evaluation on Co sites in ordered LiₓCoO₂ structures |
| Measured applications | 78 | Six reference, charging, activation and sodiation cohorts |
| Computed disordered target collection | 399 | OS, CN and OCN transfer-learning experiments |

The main benchmark contains **54,744 computed spectra** and **12 site-specific descriptor tasks**. Each processed spectrum contains 256 energy–intensity points. Computed and measured data are identified separately throughout the collection.

![SPT framework: XANES inputs are represented by adaptive Fourier lifting and energy-aware graph aggregation, followed by electronic, geometric and crystallographic descriptor prediction.](images/spt-framework.png)

**Framework overview.** SPT combines multiscale spectral processing with relationships between spectral energy points. The example input is a computed Co K-edge spectrum; internal features, connections and descriptor glyphs are schematic. Implementation details and experiment commands are in the [code repository](https://github.com/zsy-suyang/SPT).

<a id="release"></a>

## 📦 Data access

| Dataset branch | Companion code | Contents |
| --- | --- | --- |
| `main` | [SPT `main`](https://github.com/zsy-suyang/SPT) | Benchmark tables, application inputs and sources, Li–Co–O coordinates and site mapping, transfer data, fixed partitions and reference predictions |

Model checkpoints and configurations are distributed separately on the `main` branch of the [SPT model repository](https://huggingface.co/zsysuyang/XAS_SPT). This dataset can be downloaded and inspected independently of the weights.

<a id="repository-contents"></a>

## 🗂️ Repository contents

| Directory or file | Contents |
| --- | --- |
| `datasets/computed/` | Non-oxygen and oxygen computed-spectrum tables |
| `datasets/applications/` | Prepared application inputs, source information and descriptor predictions |
| `datasets/controls/` | Inputs for the reported spectral controls |
| `datasets/transfer/` | Source and target-domain transfer data |
| `data/splits/` | Fixed structure-level partitions |
| `reference_predictions/` | Per-spectrum model outputs |
| `reference_results/` | Benchmark, uncertainty and control summaries |
| `datasets/DATA_INDEX.csv` | File roles and locations |
| `ASSET_MANIFEST.json` | Asset paths and file sizes |
| `images/` | Framework illustration |
| `CITATION.cff`, `LICENSE` | Dataset citation metadata and repository licence |

The [data index](datasets/DATA_INDEX.csv) describes individual files. The [asset manifest](ASSET_MANIFEST.json) lists numerical assets and their sizes. Reference prediction files contain **model outputs**, distinct from structure-derived labels and experimental comparison values.

<a id="computed-benchmark"></a>

## 📊 Computed benchmark

### Tables and partitions

| Site family | Input table | Parent structures | Train | Validation | Test |
| --- | --- | ---: | ---: | ---: | ---: |
| Non-oxygen | [`metal.csv`](datasets/computed/metal.csv) | 3,201 | 18,911 | 3,989 | 3,970 |
| Oxygen | [`oxygen.csv`](datasets/computed/oxygen.csv) | 3,123 | 19,633 | 4,117 | 4,124 |

Counts are spectra before descriptor-specific class eligibility filtering. Within each site family, all absorbing sites from one parent structure share a partition. The assignments are stored in [`metal.json`](data/splits/final/metal.json) and [`oxygen.json`](data/splits/final/oxygen.json), keyed by `material_id`. Use these assignments and the selected model configuration for the reported experiments.

The Hub configurations `computed_non_oxygen` and `computed_oxygen` expose each complete table as a `full` split for browsing. This is not a training split. Classification eligibility and physical-value-to-class-index mappings are specified in the [model configurations](https://github.com/zsy-suyang/SPT/tree/main/configs/models).

### Data fields

| Field | Meaning |
| --- | --- |
| Unnamed first column (`Unnamed: 0` in pandas) | Saved zero-based row identifier |
| `material_id` | Parent-structure identifier used by the partition files |
| `center_atom_id` | Absorbing-site identifier within the structure |
| `center_atom_type` | Absorbing-element symbol |
| `XAS_Relative_energy` | Serialized list of 256 energy coordinates in eV; retain the supplied coordinate convention |
| `XAS_Intensity` | Serialized list of 256 corresponding computed intensities |
| Descriptor columns | Physical labels or values described below |

The spectrum columns are numeric lists stored as CSV strings. Parse them with `ast.literal_eval`, as shown below. Model input normalization is applied by the SPT loader.

### Descriptor tasks

| Absorbing sites | Classification | Regression |
| --- | --- | --- |
| Non-oxygen | OS, CN, OCN, CN2, SGN | NNRS, MOOD |
| Oxygen | CN, CN2, SGN | NNRS, NNOSM |

| Descriptor | Definition | Unit |
| --- | --- | --- |
| OS | Bond-valence-assigned oxidation state of the absorber | Dimensionless |
| CN | Number of selected first-shell neighbours | Count |
| OCN | Number of oxygen atoms among those first-shell neighbours | Count |
| CN2 | Distinct sites reached by two successive neighbour searches, excluding the absorber | Count |
| SGN (`PSGO` in files) | Space-group number of the parent crystal | Categorical identifier |
| NNRS | Standard deviation of absorber–neighbour distances: oxygen neighbours for non-oxygen absorbers, all first-shell neighbours for oxygen absorbers | Å |
| MOOD | Minimum nonzero O–O distance among first-shell oxygen neighbours | Å |
| NNOSM | Mean assigned oxidation state of the first-shell neighbours | Dimensionless |

The benchmark has eight classification and four regression tasks. The retained `NNAS` columns and oxygen-site `OS` column are not benchmark targets. Negative oxidation-state values are valid labels, not missing-value codes. Encoded prediction indices must be decoded with the corresponding model's class map.

<a id="application-spectra-and-sources"></a>

## 🔬 Application spectra and sources

| Cohort directory | Type | Spectra | Source |
| --- | --- | ---: | --- |
| [`lco_computed_co_sites`](datasets/applications/lco_computed_co_sites) | Computed | 17 | Ordered LiₓCoO₂ structures: [Ran et al.](https://doi.org/10.1038/s41524-021-00653-y); [structure repository](https://github.com/shuhebing/gsop) |
| [`lco_measured_charging`](datasets/applications/lco_measured_charging) | Measured | 11 | [Lahtinen et al.](https://doi.org/10.1016/j.mtener.2022.101040); processed spectra supplied by the study authors; [study record](https://etsin.fairdata.fi/dataset/0e8403db-c212-42a8-9b07-1c106247f84d) |
| [`co_lisa_compounds`](datasets/applications/co_lisa_compounds) | Measured | 8 | [LISA XAS database](https://lisa.iom.cnr.it/xasdb/); original XDI spectra and calibration information |
| [`ti_nist_standards`](datasets/applications/ti_nist_standards) | Measured | 9 | [NIST BMM standards](https://doi.org/10.18434/mds2-4032); original XDI spectra and NIST licence notice |
| [`fe_charging`](datasets/applications/fe_charging) | Measured | 36 | [DANOISE study](https://doi.org/10.1039/D5JA00155B); [Zenodo data](https://doi.org/10.5281/zenodo.14922524) |
| [`ni_fe_activation`](datasets/applications/ni_fe_activation) | Measured | 8 | [Ni–Fe activation study](https://doi.org/10.1038/s41467-024-55150-9); [source data](https://doi.org/10.6084/m9.figshare.27086794.v1) |
| [`ni_first_sodiation`](datasets/applications/ni_first_sodiation) | Measured | 6 | [Ni sodiation study](https://doi.org/10.1002/batt.202400451); [Zenodo data](https://doi.org/10.5281/zenodo.10889611) |

Each cohort includes a `prepared.csv`, source information in `provenance.json`, and its reference predictions. Original or supplied source files are stored in `raw/`, with additional preparation context in `source_context/` where available. Ni–Fe activation contains six activation-state spectra and two separate reference compounds. The Fe cohort uses the first 36 binned charging scans.

The computed Li–Co–O cohort includes [seven VASP coordinate files](datasets/applications/lco_computed_co_sites/source_context/structures) and a [Co-site mapping table](datasets/applications/lco_computed_co_sites/source_context/structure_site_mapping.csv). The table links each of the 17 spectra and its stored label row to a coordinate file and zero-based Co atom index. Atom order is preserved. CN, OCN, CN2 and SGN reproduce the stored labels; NNRS and MOOD agree within numerical precision under the neighbour and periodic-distance conventions in Supplementary Information Section 10.1.

Prepared spectra use long-format CSV:

| Column | Meaning |
| --- | --- |
| `spectrum_id` | Spectrum identifier |
| `point_index` | Point order, 0–255 within each spectrum |
| `energy_eV` | Prepared energy coordinate in eV |
| `intensity` | Prepared spectral intensity; model-specific normalization is applied during inference |
| `absorber` | Absorbing-element symbol |

Cohort-specific metadata include voltage, state of charge, scan number, sample name or activation state. The prepared inputs preserve the energy referencing and resampling used in the reported applications. [Application documentation](https://github.com/zsy-suyang/SPT/blob/main/docs/application-data.md) gives the preparation and source details. Application NNRS and MOOD predictions are in ångströms.

<a id="transfer-data-and-reference-results"></a>

## 🔄 Transfer data and reference results

`datasets/transfer/` separates the crystalline computed source data, computed disordered target data and task-specific training subsets. The OS, CN and OCN target tables provide entry points to the same 399-spectrum target collection. The fixed transfer partition contains 285 training-pool spectra and 114 evaluation spectra; the training subsets contain 29, 72 or 143 spectra. Partition and subset definitions are in `data/splits/transfer/`. These are computed spectra, not experimental measurements.

`reference_predictions/` contains per-spectrum outputs for the computed benchmark, transfer experiments and uncertainty analyses. `reference_results/` contains numerical summaries. Use the [experiment guides](https://github.com/zsy-suyang/SPT/blob/main/docs/reproduce.md) and [transfer guide](https://github.com/zsy-suyang/SPT/blob/main/docs/transfer.md) to select the corresponding inputs and models.

<a id="download-and-use"></a>

## 📥 Download and use

### Download the collection

Install the Hugging Face client. The collection is public and can be downloaded without authentication.

```bash
python -m pip install -U huggingface_hub
hf download zsysuyang/XAS_SPT --repo-type dataset \
  --revision main --local-dir spt-data
```

### Load a computed table

Install the libraries for the standalone loading examples:

```bash
python -m pip install pandas datasets
```

```python
import ast
import pandas as pd

table = pd.read_csv("spt-data/datasets/computed/metal.csv", index_col=0)
energy = ast.literal_eval(table.iloc[0]["XAS_Relative_energy"])
intensity = ast.literal_eval(table.iloc[0]["XAS_Intensity"])
print(table.shape, len(energy), len(intensity))  # (26870, 13), 256, 256
```

The two computed subsets can also be loaded with the Hugging Face `datasets` library:

```python
from datasets import load_dataset

spectra = load_dataset(
    "zsysuyang/XAS_SPT",
    "computed_non_oxygen",
    revision="main",
    split="full",
)
```

### Run SPT inference

Clone and install [SPT](https://github.com/zsy-suyang/SPT#installation). Copy `datasets/`, `data/splits/`, `reference_predictions/` and `reference_results/` from this collection into the matching code checkout. Retain the code checkout's root README and complete `data/asset_manifest.json`. Install the separate checkpoint collection in `checkpoints/` before inference.

From the assembled code checkout:

```bash
spt verify-assets --all-assets
spt application --cohort ni_fe_activation \
  --tasks OS CN OCN NNRS MOOD \
  --output outputs/ni_fe_activation.csv
```

The Ni applications use their registered CPU settings. Other cohorts and the computed reference experiments use the environments specified in the [application guide](https://github.com/zsy-suyang/SPT/blob/main/docs/applications.md) and [numerical-experiment guide](https://github.com/zsy-suyang/SPT/blob/main/docs/reproduce.md). The model registry maps each experiment to its data and checkpoint.

<a id="intended-use"></a>

## 🎯 Intended use

The collection supports descriptor-learning benchmarks, spectral sensitivity analysis and evaluation on the supplied application cohorts. Computed labels derive from the corresponding structures; measured-data comparison labels and EXAFS values are identified in the cohort source information. Reference predictions are not substitutes for experimental ground truth. Apply the supplied partitions, descriptor eligibility and preprocessing when comparing with the reported results.

<a id="licence-and-attribution"></a>

## ⚖️ Licence and attribution

Unless separately indicated, author-owned data and documentation in this release are licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](LICENSE) (CC BY-NC-SA 4.0). Attribution and ShareAlike conditions apply as specified in the licence. For separate commercial permission, contact [Jun Cheng](mailto:chengjun@xmu.edu.cn) or [Fujie Tang](mailto:tangfujie@xmu.edu.cn).

Third-party spectra, structures and source files retain their original terms and attribution requirements. In particular, retain the [NIST licence notice](datasets/applications/ti_nist_standards/source_context/NIST_LICENSE). The repository-level licence does not override these source-specific terms or revoke rights previously granted under Apache-2.0.

<a id="citation"></a>

## 📝 Citation

Please cite this dataset collection and the accompanying manuscript. Dataset metadata are provided in [CITATION.cff](CITATION.cff).

```bibtex
@misc{zhong2026sptdata,
  author = {Zhong, Suyang and Huang, Boying and Xu, Pengwei and Xu, Fanjie
            and Zhao, Yuhao and Cheng, Jun and Tang, Fujie and E, Weinan
            and Tian, Zhong-Qun},
  title = {SPT XANES Dataset},
  year = {2026},
  howpublished = {Hugging Face},
  url = {https://huggingface.co/datasets/zsysuyang/XAS_SPT}
}
```

> Suyang Zhong, Boying Huang, Pengwei Xu, Fanjie Xu, Yuhao Zhao, Jun Cheng, Fujie Tang, Weinan E and Zhong-Qun Tian. *Inferring structural descriptor information from X-ray absorption spectra through adaptive multiscale learning.*

Please also cite the original experimental sources used in your analysis. For data or usage questions, open an [SPT issue](https://github.com/zsy-suyang/SPT/issues). Scientific correspondence: [Jun Cheng](mailto:chengjun@xmu.edu.cn) and [Fujie Tang](mailto:tangfujie@xmu.edu.cn).
