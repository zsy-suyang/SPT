# Synthetic input example

`spectrum.csv` is a deterministic, synthetic sigmoid-plus-Gaussian curve on
256 absolute-energy points. It demonstrates the CSV format and prediction interface.

Columns are `spectrum_id`, `energy_eV`, `intensity`, and `absorber`.
The identifier groups points into one spectrum; keep energy points in ascending
order. No labels are required for prediction. See [Datasets](../docs/datasets.md)
for measured-spectrum preparation and [Applications](../docs/applications.md)
for the prepared experimental cohorts.

From the repository root, with the corresponding checkpoint installed:

```bash
spt predict --model expert-metal-OS \
  --input examples/spectrum.csv \
  --output outputs/example_prediction.csv
```
