# Architecture and inputs

The Spectral Pattern Translator predicts descriptors of an absorbing environment from XANES. Two parallel operators represent variation across spectral scales and context among energy-resolved points.

## Input encoding

A spectrum is an ordered sequence of energy–intensity pairs with an absorbing-element identifier. A shared linear map embeds the spectral points, and a learned absorber embedding supplies chemical context. Fixed energy-scaling bounds are reused at inference. Intensities are normalized separately for each spectrum and standardized inside the model with a non-affine layer normalization.

The scaled energy coordinates also enter the graph operator directly. Learned positional embeddings are not used.

## Spectral–graph blocks

Each block applies two operators in parallel to separately normalized versions of the incoming tokens:

- Adaptive Fourier lifting performs an input-dependent multiscale decomposition, modifies the scale coefficients and synthesizes a sequence-shaped spectral update.
- The energy-point graph aggregates local and non-local spectral context using content features, neighbour information and supplied energy separation.

Learned channelwise scales and stochastic depth modulate the residual updates. The residual stream is normalized after combining the incoming tokens and both branches. The configuration determines the block count and model widths.

The [AFLD round-trip experiment](reproduce.md#unchanged-coefficient-afld-round-trip) tests the analysis–synthesis pair with unchanged coefficients.

## Readout and model families

Global and regional pooling summarize the final energy-ordered tokens. The pooled features are combined with the absorber embedding and passed to descriptor heads.

An Expert model is trained for one descriptor. A Joint model shares the spectral encoder and returns multiple descriptor outputs for a site family. Each configuration defines its descriptor heads and output-label mappings.

## Configuration and reproducibility

Construct models through the public registry and the configuration associated with each model identifier. Use the same spectral length, architecture, element encoding, output class map and saved scaling bounds as the checkpoint. The publication configurations define the exact dimensions and operator settings for each experiment.

See [Numerical experiments](reproduce.md) for evaluation and analysis commands.
