# Datasets and input preparation

SPT uses separate models for non-oxygen (`metal`) and oxygen (`oxygen`) absorbing sites.

## Prediction CSV

Provide a long-format CSV with one spectral point per row:

| Column | Meaning |
| --- | --- |
| `energy_eV` | Absolute, positive energy in eV, ordered within each spectrum |
| `intensity` | Spectral intensity paired with the energy coordinate |
| `absorber` | Absorbing-element symbol, constant within a spectrum |
| `spectrum_id` | Optional identifier for grouping multiple spectra |

Each spectrum contains 256 points. Use an absorber supported by the selected checkpoint. The prediction interface applies the model's saved input transformations; provide energy in eV rather than pre-scaled coordinates.

`examples/spectrum.csv` provides a small synthetic input example.

## Computed training and evaluation data

The processed tables are located at:

```text
datasets/computed/metal.csv
datasets/computed/oxygen.csv
```

Use the parent-structure assignments in `data/splits/final/*.json` to reproduce the reported results. Spectra from the same parent structure belong to the same partition.

The dataset, split manifest, descriptor encoding and model configuration form one input specification. Retain the original row identifiers and parent-structure identifiers when preparing a compatible table. Labels use the descriptor-specific mappings associated with the selected model; encoded class indices must be decoded before interpretation as physical descriptor values.

See [Data and weights](../README.md#data-and-weights) for the download entries. `datasets/DATA_INDEX.csv` lists the data files, and `configs/registry.json` associates the computed tables and partitions with each site family.

## Experimental spectra

Prepared measured spectra are organized by cohort under `datasets/applications/`, separately from the computed training tables. Each cohort includes its source information and measurement-specific energy referencing and resampling protocol. See [Application spectra and sources](application-data.md) and [Application commands](applications.md).

## Preprocessing and model metadata

Inference applies the trained model's site-family-specific energy scaling, per-spectrum intensity normalization, absorber mapping and class maps. Supply energy in eV and preserve energy–intensity pairing throughout interpolation and intervention analyses.

## Data terms

Third-party spectra, structures and databases retain their originating attribution and licence terms. Data and checkpoint reuse terms are separate from the source-code licence and belong with their respective download records.
