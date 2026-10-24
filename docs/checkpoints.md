# Model checkpoints

The [weight index](../checkpoints/WEIGHT_INDEX.csv) is the central table linking experiments, descriptors, configurations and checkpoint files. Model identifiers resolve through `configs/registry.json`. Weights are organized by experiment:

```text
checkpoints/benchmark/<method>/<site>/
checkpoints/ablations/modules/<variant>/<site>/
checkpoints/ablations/transform/<variant>/<site>/
checkpoints/applications/<system>/
checkpoints/robustness/<experiment>/<method>/<site>/
checkpoints/interpretability/<experiment>/<method>/<site>/
checkpoints/uncertainty/<method>/<site>/
checkpoints/runtime/<method>/<site>/
checkpoints/transfer/<task>/
```

For example, `expert-metal-OS` uses `checkpoints/benchmark/spt_expert/non_oxygen/SPT_OS.pth`. `metal` in model identifiers corresponds to `non_oxygen` in these checkpoint directories.

Select application models through `configs/applications.json` and intervention, uncertainty or runtime models through `configs/experiments.json`. Transfer configurations identify the source model, initialization state and final target model for each case; see [Transfer experiments](transfer.md).

The [application table](applications.md) lists the descriptors for each cohort. Ti standards use OS and CN; both Ni cohorts use OS, CN, OCN, NNRS and MOOD.

See [Data and weights](../README.md#data-and-weights) for downloads and directory placement.

## Match weights to a model

Use the architecture, descriptor head, site family, absorber mapping, energy scaling and label encoding associated with the checkpoint. Expert and Joint checkpoints have different output specifications. Retain the original model configuration and frozen partition when reproducing an experiment.

## Application model mapping

Filter the weight index to model identifiers beginning with `app-` to find the application checkpoints. Each experiment has designated weight files and inference configurations, selected through `configs/applications.json`.

For application entries, `configuration` specifies inference settings; `training_configuration` and `training_partition` identify the source training settings and structure-level partition. Training configurations retain the saved source-run settings with repository-relative paths. The source models used `datasets/computed/metal.csv`, `data/splits/final/metal.json` and seed 42.

Each training configuration specifies the optimization parameters and validation checkpoint-selection rule. To retrain a source model, use its `training_configuration` with the [training command](reproduce.md#computed-benchmark-and-architectural-controls).

Check the files for an individual model with `spt verify-assets --model MODEL_ID`, or the full collection with `spt verify-assets --all-assets`.

## Outputs

Classification heads return scores for the trained classes and physical labels decoded with the saved class map. Regression outputs use the descriptor definition and units specified by the selected configuration.
