# Third-party notices

Unless separately indicated, author-owned SPT code, documentation, data and model weights in this release are licensed under CC BY-NC-SA 4.0, provided in `LICENSE`. Third-party software, adapted code, datasets and model assets retain their respective terms and are excluded from this licence change. Rights previously granted under Apache-2.0 remain unaffected.

## External dependency: einops

Comparison-model components use `einops` for tensor rearrangement. The upstream project is <https://github.com/arogozhnikov/einops>, licensed under the MIT License. Its copyright notice names Alex Rogozhnikov, 2018. See the [upstream license](https://github.com/arogozhnikov/einops/blob/main/LICENSE). The dependency is installed separately rather than copied into this repository.

## FT-Transformer implementation

The FT-Transformer baseline contains attention, GEGLU and numerical-embedding components adapted from <https://github.com/lucidrains/tab-transformer-pytorch>. That upstream implementation is licensed under the MIT License, copyright Phil Wang, 2020. See the [upstream implementation](https://github.com/lucidrains/tab-transformer-pytorch/blob/main/tab_transformer_pytorch/ft_transformer.py) and [license](https://github.com/lucidrains/tab-transformer-pytorch/blob/main/LICENSE).

The FT-Transformer method is described in *Revisiting Deep Learning Models for Tabular Data*.

### MIT License notice for adapted FT-Transformer components

Copyright (c) 2020 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## AMFormer baseline

The AMFormer comparison implementation is in [`spt/baselines/amformer/`](spt/baselines/amformer/). Its model and attention blocks correspond to the official source at <https://github.com/aigc-apps/AMFormer>. The [upstream GNU GPL version 3](https://github.com/aigc-apps/AMFormer/blob/main/LICENSE) is retained in full in that directory's [LICENSE](spt/baselines/amformer/LICENSE); its [README](spt/baselines/amformer/README.md) describes how to select the baseline. These components retain those terms and are not relicensed or restricted to non-commercial use by the repository-level CC BY-NC-SA 4.0 licence.

## Data, weights and other dependencies

Third-party spectra and structures retain the terms of their originating resources. The source-code license does not grant rights to redistribute third-party datasets or model assets. Their external release records specify the applicable reuse terms. Other Python dependencies retain the notices and licenses in their installed distributions.
