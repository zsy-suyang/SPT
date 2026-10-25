# Hugging Face card sources

These files maintain the presentation of the two public SPT collections:

| Source | Destination |
| --- | --- |
| `dataset-card.md` | [Dataset README](https://huggingface.co/datasets/zsysuyang/XAS_SPT) |
| `model-card.md` | [Model README](https://huggingface.co/zsysuyang/XAS_SPT) |

Publish each card as the destination repository's root `README.md`. Its relative links resolve from that root, not this source directory. Publish the four `docs/assets/badge-{code,data,weights,license}.svg` assets as `images/{code,data,weights,license}-badge.svg` in each collection. The `-badge.svg` suffix enables Hugging Face's inline badge styling. The existing framework illustration remains at `images/spt-framework.png`.

Maintain both cards on their repositories' `main` branches. Keep the model repository's `configs/assets.json` synchronized with the code repository so that the download instructions and configured branches agree.
