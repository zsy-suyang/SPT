# Hugging Face card sources

These files maintain the presentation of the two public SPT collections:

| Source | Destination |
| --- | --- |
| `dataset-card.md` | [Dataset README](https://huggingface.co/datasets/zsysuyang/XAS_SPT) |
| `model-card.md` | [Model README](https://huggingface.co/zsysuyang/XAS_SPT) |

Publish each card as the destination repository's root `README.md`. Its relative links resolve from that root, not this source directory. The four `badge-*.svg` assets in `docs/assets/` are published under `images/` in each collection. The existing framework illustration remains at `images/spt-framework.png`.
