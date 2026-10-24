# Optional AMFormer comparison implementation

`model.py` and `blocks.py` implement the AMFormer baseline used in the benchmark
and noise comparisons. The model factory selects this package for
`architecture: amformer`.

The upstream project is [aigc-apps/AMFormer](https://github.com/aigc-apps/AMFormer).
Its [GNU General Public License, version 3](https://github.com/aigc-apps/AMFormer/blob/main/LICENSE)
is reproduced in full in [LICENSE](LICENSE). This directory retains those
terms; the Apache 2.0 license at the repository root does not relicense it.

Use the AMFormer model IDs listed in `configs/registry.json` with the
[benchmark and noise commands](../../../docs/reproduce.md).
