# Download layout and versions

SPT is distributed as a code repository, a data collection and a checkpoint collection. Package paths are relative to the code checkout, so the same experiment commands apply after downloading the assets.

| Component | Location | Contents |
| --- | --- | --- |
| Code, version `v0.1.1` | [GitHub](https://github.com/zsy-suyang/SPT/tree/v0.1.1) | Implementation, configurations, fixed partitions, small examples, tests and experiment guides |
| Data, release `2026-10-25` | [XAS_SPT dataset](https://huggingface.co/datasets/zsysuyang/XAS_SPT/tree/2026-10-25) | Computed and measured inputs, source information, partitions and reference predictions |
| Weights, release `2026-10-25` | [XAS_SPT models](https://huggingface.co/zsysuyang/XAS_SPT/tree/2026-10-25) | Experiment-specific checkpoints, configuration files and the weight index |

Both Hugging Face collections are public. Use the release tags above with code version `0.1.1`; [configs/assets.json](../configs/assets.json) records the matching repository identifiers and tags.

## Download into the code checkout

Run these commands from the installed SPT repository root:

```bash
python -m pip install "huggingface_hub>=2.3,<3"
hf download zsysuyang/XAS_SPT --repo-type dataset --revision 2026-10-25 --local-dir . --include "datasets/**" --include "data/**" --include "reference_predictions/**" --include "reference_results/**"
hf download zsysuyang/XAS_SPT --repo-type model --revision 2026-10-25 --local-dir . --include "checkpoints/**" --include "configs/**"
```

The path filters install the numerical assets and matching configurations without replacing the code README or its complete asset manifest.

## Download separately and assemble

To keep separate copies of both collections:

```bash
hf download zsysuyang/XAS_SPT --repo-type dataset --revision 2026-10-25 --local-dir ../spt-data
hf download zsysuyang/XAS_SPT --repo-type model --revision 2026-10-25 --local-dir ../spt-weights
```

1. Clone the code repository and install it as described in the [README](../README.md#installation).
2. Copy the data bundle's `datasets/`, `data/`, `reference_predictions/` and `reference_results/` folders into the checkout, preserving their paths.
3. Copy the weight bundle's `checkpoints/` and `configs/` folders into the checkout. The configuration snapshot corresponds to code version `0.1.1` and the checkpoints in release `2026-10-25`.

Keep the code checkout's root README and its complete `data/asset_manifest.json`. Each download bundle has its own root-level `ASSET_MANIFEST.json`, listing that bundle's files and sizes.

From the assembled checkout, check the asset paths and run a prepared CPU application:

```bash
spt verify-assets --all-assets
spt application --cohort ni_fe_activation \
  --tasks OS CN OCN NNRS MOOD \
  --output outputs/ni_fe_activation.csv
```

Use a new output filename for each run. See [Applications](applications.md) for the remaining cohorts and [Numerical experiments](reproduce.md) for benchmark, uncertainty and robustness commands. Computed benchmark and noise reference runs use the CUDA environment specified there.

## Citation and terms

[CITATION.cff](../CITATION.cff) provides the authors and manuscript title. The data and model cards include collection-specific citation information and reuse terms. Record the release tag used for an analysis and cite the original measured-data sources where applicable. The source-code licence and [third-party notices](../THIRD_PARTY_NOTICES.md) do not replace the terms of third-party data or components.
