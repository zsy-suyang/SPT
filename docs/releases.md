# Download layout and versions

SPT is distributed as a code repository, a data collection and a checkpoint collection. Package paths are relative to the code checkout, so the same experiment commands apply after downloading the assets.

| Component | Location | Contents |
| --- | --- | --- |
| Code, version 0.1.0 | [GitHub](https://github.com/zsy-suyang/SPT) | Implementation, configurations, fixed partitions, small examples, tests and experiment guides |
| Data | [XAS_SPT on Hugging Face](https://huggingface.co/datasets/zsysuyang/XAS_SPT) | Computed and measured inputs, source information, partitions and reference predictions |
| Weights | Separate Hugging Face repository, link pending | Experiment-specific checkpoints, configuration files and the weight index |

The reorganized data bundle and weight bundle for this code version are pending upload. Their repository revisions will be recorded in `configs/assets.json` when available. A repository address identifies the collection; it does not by itself identify a compatible data version.

## Assemble the downloads

1. Clone the code repository and install it as described in the [README](../README.md#installation).
2. Copy the data bundle's `datasets/`, `data/`, `reference_predictions/` and `reference_results/` folders into the checkout, preserving their paths.
3. Copy the weight bundle's `checkpoints/` folder into the checkout. The accompanying `configs/` snapshot identifies the configurations associated with those weights; use it with the matching code version.

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

[CITATION.cff](../CITATION.cff) provides the authors and manuscript title. Cite the data and checkpoint repository revisions used for an analysis once their releases are available. The source-code licence and [third-party notices](../THIRD_PARTY_NOTICES.md) do not replace the data or checkpoint reuse terms in their respective repository records.
