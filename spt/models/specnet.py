import torch
import torch.nn as nn
import torch.nn.functional as F


def project_real_fft_endpoints(spectrum, length):
    """Project a one-sided FFT onto the real-signal endpoint constraints.

    DC must be real, as must Nyquist for even signal lengths. For odd lengths
    the final bin is not Nyquist and its imaginary component must be retained.
    Explicit projection avoids backend-dependent handling of invalid endpoint
    imaginary values after a learned complex response. It changes no real part
    and adds neither parameters nor persistent buffers.
    """
    length = int(length)
    if length < 1 or not spectrum.is_complex() or spectrum.shape[-1] != length // 2 + 1:
        raise ValueError("Real FFT endpoint projection requires the matching positive signal length")
    mask = spectrum.real.new_ones(spectrum.shape[-1])
    mask[0] = 0.0
    if length % 2 == 0:
        mask[-1] = 0.0
    return torch.complex(spectrum.real, spectrum.imag * mask)


def _inverse_softplus(value):
    """Raw parameter whose softplus equals ``value``."""
    return float(torch.log(torch.expm1(torch.tensor(float(value)))))


def channel_split_sizes(d_model, graph_fraction=0.5):
    """Nominal channel budget of the spectral and the graph module.

    The two modules read the full residual stream and write it back through a
    learned *partition of unity*: channel ``i`` receives ``m_i`` of the
    spectral update and ``1 - m_i`` of the graph update. Letting both write
    freely into one shared group makes their updates interchangeable degrees
    of freedom, so gradient descent assigns credit by convergence speed rather
    than by function and the branches become substitutes. Tying the two shares
    to sum to one keeps them complements without wasting capacity on whichever
    branch a given descriptor does not need.

    These sizes only set where the allocation *starts* and which channels are
    normalised together; the boundary itself is learned.
    """
    d_model = int(d_model)
    graph_width = int(round(d_model * float(graph_fraction)))
    graph_width = max(1, min(d_model - 1, graph_width))
    return d_model - graph_width, graph_width


def _triangular_frequency_masks(num_frequencies, num_bands, keep_dc=False):
    positions = torch.linspace(0.0, 1.0, num_frequencies)
    centers = torch.linspace(0.0, 1.0, num_bands)
    width = 1.0 / max(1, num_bands - 1)
    masks = torch.clamp(
        1.0 - torch.abs(positions[:, None] - centers[None, :]) / width,
        min=0.0,
    ).transpose(0, 1)
    if not keep_dc:
        masks[:, 0] = 0.0
    return masks.contiguous()


class DropPath(nn.Module):
    """Per-sample stochastic depth on a residual branch (ConvNeXt/CaiT)."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
        return x * mask / keep_prob


class MultiHeadEnergyGraph(nn.Module):
    """Graph attention over the spectrum's energy axis, local and non-local.

    The heads are split between two kinds of neighbourhood.

    *Local heads* own a **dilated** window, so together they reach from the
    immediate neighbours out to tens of grid points.

    *Non-local heads* build their own adjacency: every position scores every
    other position and keeps its ``global_k`` best partners, anywhere in the
    spectrum. This is the part of the module that a filter cannot imitate, and
    the module ablation says it is needed. Measured on the 300-epoch grid with
    local heads only, adding either branch to `bare` bought almost exactly the
    same thing -- on metal:CN2 +0.106 for the graph, +0.099 for the spectral --
    and the two together bought +0.109, the *maximum* of the two rather than the
    sum. That is the signature of redundancy, and its cause is structural: a
    diagonal gain in a transform domain is a convolution along the energy axis
    by the convolution theorem, and a dilated window of +-24 points out of 256 is
    also a convolution, merely an adaptive one. Two convolutions of similar
    reach discover the same thing.

    A learned, per-sample adjacency is a different object. It can pair the
    white-line peak with the oscillation that follows the edge even though their
    separation shifts from material to material, which no shift-invariant filter
    can express. It also gives the physical energy term something to select
    from: a band-pass on ``|E_j - E_i|`` centred at a characteristic separation
    is meaningless inside a window narrower than the separations that matter.

    Edge scores combine four terms:

    ``content``       scaled dot product between centre and neighbour
    ``position bias`` learned prior, over window offset for local heads and over
                      partner rank for non-local ones
    ``edge message``  projection of the centre-to-neighbour feature difference,
                      i.e. a genuine message-passing edge function rather than
                      pure attention
    ``energy kernel`` physical term over the energy gap ``|E_j - E_i|``: a
                      monotone decay **plus** a per-head band-pass centred at a
                      learnable characteristic separation. A purely monotone
                      penalty can only say "closer is better" and cannot express
                      couplings at a characteristic energy offset (edge to
                      white-line, multiple-scattering shells).
    """

    def __init__(
        self,
        feat_dim,
        num_neighbors,
        device=None,
        dropout=0.1,
        num_heads=4,
        dilations=None,
        max_dilation=8,
        global_heads=None,
        global_k=8,
    ):
        super().__init__()
        if feat_dim % max(1, num_heads):
            num_heads = 1
        self.num_neighbors = int(num_neighbors)
        self.num_heads = int(num_heads)
        self.head_dim = feat_dim // self.num_heads
        self.window = 2 * self.num_neighbors + 1
        # Half the heads go non-local by default. Splitting rather than adding
        # heads keeps the branch's width, and therefore its parameter count,
        # exactly what it was.
        self.global_heads = (
            self.num_heads // 2 if global_heads is None else int(global_heads)
        )
        self.global_heads = max(0, min(self.num_heads, self.global_heads))
        self.local_heads = self.num_heads - self.global_heads
        self.global_k = int(global_k)
        if dilations is None:
            # Spread geometrically over ``1 .. max_dilation`` so the local heads
            # tile the same span however many of them there are. Indexing the
            # dilation by head number instead means reassigning half the heads to
            # the non-local branch also quietly cuts the local reach -- with four
            # heads it was +-24 of 256 points, with two it would be +-6 -- so a
            # change meant to remove redundancy would have removed range as well.
            span = max(1, self.local_heads - 1)
            dilations = [
                int(round(float(max_dilation) ** (head / span)))
                for head in range(self.local_heads)
            ]
        if len(dilations) != self.local_heads:
            raise ValueError(
                "dilations must provide one entry per local graph head "
                f"({self.local_heads}), got {len(dilations)}"
            )
        self.register_buffer(
            "dilations",
            torch.tensor([int(d) for d in dilations], dtype=torch.long),
            persistent=False,
        )
        self.query = nn.Linear(feat_dim, feat_dim, bias=False)
        self.key = nn.Linear(feat_dim, feat_dim, bias=False)
        self.value_proj = nn.Linear(feat_dim, feat_dim)
        self.output_proj = nn.Linear(feat_dim, feat_dim)
        offsets = torch.arange(
            -self.num_neighbors, self.num_neighbors + 1, dtype=torch.float32
        )
        self.relative_bias = (
            nn.Parameter(
                (-0.1 * offsets.abs()).unsqueeze(0).repeat(self.local_heads, 1)
            )
            if self.local_heads
            else None
        )
        # Rank prior for the non-local heads: rank 0 is the closest match in
        # content, so a decay in rank is the analogue of a decay in distance.
        ranks = torch.arange(self.global_k, dtype=torch.float32)
        self.rank_bias = (
            nn.Parameter(
                (-0.1 * ranks).unsqueeze(0).repeat(self.global_heads, 1)
            )
            if self.global_heads
            else None
        )
        # Edge function on (neighbour - centre) features, one vector per head.
        self.edge_weight = nn.Parameter(
            torch.randn(self.num_heads, self.head_dim) * self.head_dim ** -0.5
        )
        self.distance_scale = nn.Parameter(torch.ones(self.num_heads))
        # Band-pass over the normalized energy gap. Centres are spread across
        # heads so they specialize instead of collapsing onto d = 0.
        self.band_center = nn.Parameter(
            torch.linspace(0.0, 0.5, self.num_heads)
        )
        self.band_width = nn.Parameter(
            torch.full((self.num_heads,), _inverse_softplus(0.5))
        )
        self.ffn_norm = nn.LayerNorm(feat_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, 2 * feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * feat_dim, feat_dim),
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.last_attention = None
        self.last_energy_distance = None
        self.last_band_center = None

    def _neighbor_indices(self, length, device):
        """Per-head dilated neighbour indices -> ``[H, T, W]`` plus validity."""
        offsets = torch.arange(
            -self.num_neighbors,
            self.num_neighbors + 1,
            device=device,
        )
        scaled = offsets.view(1, -1) * self.dilations.to(device).view(-1, 1)
        centers = torch.arange(length, device=device).view(1, -1, 1)
        raw = centers + scaled.unsqueeze(1)
        return raw.clamp(0, length - 1), (raw >= 0) & (raw < length)

    @staticmethod
    def _gather(per_head, indices):
        """``[B, H, T, d]`` gathered at ``[B, H, T, W]`` -> ``[B, H, T, W, d]``."""
        batch, heads, length, dim = per_head.shape
        width = indices.shape[-1]
        flat = indices.reshape(batch, heads, length * width, 1).expand(
            -1, -1, -1, dim
        )
        return torch.gather(per_head, 2, flat).view(
            batch, heads, length, width, dim
        )

    def _energy_kernel(self, gap, head_slice):
        """Physical penalty on the energy gap, normalized per head."""
        # Per-head normalization: a non-local head spans the whole spectrum
        # while a dilated one spans a few points, so one shared scale would
        # squash the near-field heads into gap ~ 0.
        scale = gap.detach().amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        gap = gap / scale
        shape = (1, -1, 1, 1)
        centers = self.band_center[head_slice].view(shape)
        return gap, (
            F.softplus(self.distance_scale[head_slice]).view(shape) * gap
            + F.softplus(self.band_width[head_slice]).view(shape)
            * (gap - centers).square()
        )

    def _aggregate(
        self, content, keys, values, centre, gap, bias, head_slice, valid=None
    ):
        """Score four terms, attend, and return ``[B, H, T, head_dim]``."""
        edge = torch.einsum(
            "bhtwd,hd->bhtw", keys - centre, self.edge_weight[head_slice]
        )
        gap, penalty = self._energy_kernel(gap, head_slice)
        scores = content + bias + edge - penalty
        if valid is not None:
            scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        return (
            torch.einsum(
                "bhtw,bhtwd->bhtd", self.attention_dropout(attention), values
            ),
            attention,
            gap,
        )

    def _local(self, queries, keys, values, energy, length):
        """Dilated-window attention over the ``local_heads`` head block."""
        head_slice = slice(0, self.local_heads)
        indices, valid = self._neighbor_indices(length, queries.device)
        batch = queries.shape[0]
        indices = indices.unsqueeze(0).expand(batch, -1, -1, -1)
        keys = self._gather(keys[:, head_slice], indices)
        values = self._gather(values[:, head_slice], indices)
        content = torch.einsum(
            "bhtd,bhtwd->bhtw", queries[:, head_slice], keys
        ) / self.head_dim ** 0.5
        gap = self._gap(energy, indices, length)
        bias = self.relative_bias.view(1, self.local_heads, 1, self.window)
        centre = keys[..., self.num_neighbors:self.num_neighbors + 1, :]
        return self._aggregate(
            content, keys, values, centre, gap, bias, head_slice, valid=valid
        )

    def _nonlocal(self, queries, keys, values, energy, length):
        """Top-k learned adjacency over the ``global_heads`` head block."""
        head_slice = slice(self.local_heads, self.num_heads)
        query_block = queries[:, head_slice]
        key_block = keys[:, head_slice]
        # Full pairwise scores, then keep each position's best partners. The
        # scores are reused as the content term, so the dense matrix is formed
        # once and never stored per edge.
        dense = torch.einsum(
            "bhtd,bhsd->bhts", query_block, key_block
        ) / self.head_dim ** 0.5
        content, indices = dense.topk(
            min(self.global_k, length), dim=-1
        )
        gathered_keys = self._gather(key_block, indices)
        gathered_values = self._gather(values[:, head_slice], indices)
        gap = self._gap(energy, indices, length)
        bias = self.rank_bias[:, : indices.shape[-1]].view(
            1, self.global_heads, 1, indices.shape[-1]
        )
        # The centre of a non-local neighbourhood is the node itself, not a
        # window midpoint, so the edge function subtracts that instead.
        return self._aggregate(
            content,
            gathered_keys,
            gathered_values,
            key_block.unsqueeze(3),
            gap,
            bias,
            head_slice,
        )

    @staticmethod
    def _gap(energy, indices, length):
        """``|E_j - E_i|`` for ``[B, H, T, W]`` neighbour indices."""
        batch, heads = indices.shape[0], indices.shape[1]
        if energy is None:
            centres = torch.arange(
                length, device=indices.device, dtype=torch.float32
            ).view(1, 1, -1, 1)
            return (indices.to(torch.float32) - centres).abs()
        if energy.ndim == 1:
            energy = energy.unsqueeze(0).expand(batch, -1)
        per_head = energy.unsqueeze(1).expand(batch, heads, length)
        neighbor = torch.gather(
            per_head, 2, indices.reshape(batch, heads, -1)
        ).view(indices.shape)
        return (neighbor - energy.view(batch, 1, length, 1)).abs()

    def forward(self, x, energy=None):
        batch, length, _ = x.shape

        def split(projected):
            return projected.view(
                batch, length, self.num_heads, self.head_dim
            ).permute(0, 2, 1, 3)

        queries = split(self.query(x))
        keys = split(self.key(x))
        values = split(self.value_proj(x))

        blocks, attentions, gaps = [], [], []
        if self.local_heads:
            block, attention, gap = self._local(
                queries, keys, values, energy, length
            )
            blocks.append(block)
            attentions.append(attention)
            gaps.append(gap)
        if self.global_heads:
            block, attention, gap = self._nonlocal(
                queries, keys, values, energy, length
            )
            blocks.append(block)
            attentions.append(attention)
            gaps.append(gap)

        aggregated = torch.cat(blocks, dim=1).permute(0, 2, 1, 3).reshape(
            batch, length, self.num_heads * self.head_dim
        )
        # The two neighbourhoods have different widths, so the traces are kept
        # per kind rather than concatenated into a ragged tensor.
        self.last_attention = tuple(a.mean(dim=1).detach() for a in attentions)
        self.last_energy_distance = tuple(g.detach() for g in gaps)
        self.last_band_center = self.band_center.detach()
        message = self.output_proj(aggregated)
        return self.output_dropout(message + self.ffn(self.ffn_norm(message)))


def _orthonormal_dct_basis(length):
    """Return an orthonormal DCT-II basis with shape [frequency, position]."""
    positions = torch.arange(length, dtype=torch.float32) + 0.5
    frequencies = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    basis = torch.cos(torch.pi * frequencies * positions / float(length))
    basis *= (2.0 / float(length)) ** 0.5
    basis[0] /= 2.0 ** 0.5
    return basis


def _hard_frequency_masks(length, num_bands, drop_dc=True):
    """Create disjoint low-to-high coefficient bands."""
    if num_bands < 1 or num_bands > length:
        raise ValueError("num_bands must be in [1, transform length]")
    edges = torch.linspace(0, length, num_bands + 1).round().long()
    masks = torch.zeros(num_bands, length)
    for index in range(num_bands):
        masks[index, edges[index]:edges[index + 1]] = 1.0
    if drop_dc:
        masks[:, 0] = 0.0
    return masks


def haar_forward(x):
    """Orthonormal Haar DWT ordered from coarse to fine coefficients."""
    if x.shape[1] < 2 or x.shape[1] & (x.shape[1] - 1):
        raise ValueError("Haar transform length must be a power of two")
    current = x
    details = []
    scale = 2.0 ** 0.5
    while current.shape[1] > 1:
        even = current[:, 0::2]
        odd = current[:, 1::2]
        details.append((even - odd) / scale)
        current = (even + odd) / scale
    return torch.cat([current] + list(reversed(details)), dim=1)


def haar_inverse(coefficients):
    """Inverse of :func:`haar_forward` for coarse-to-fine ordering."""
    length = coefficients.shape[1]
    if length < 2 or length & (length - 1):
        raise ValueError("Haar transform length must be a power of two")
    current = coefficients[:, :1]
    offset = 1
    scale = 2.0 ** 0.5
    detail_length = 1
    while detail_length < length:
        detail = coefficients[
            :, offset:offset + detail_length
        ]
        even = (current + detail) / scale
        odd = (current - detail) / scale
        current = torch.stack([even, odd], dim=2).reshape(
            coefficients.shape[0], detail_length * 2, coefficients.shape[2]
        )
        offset += detail_length
        detail_length *= 2
    return current


BAND_GAIN_MEAN = 0.1
BAND_GAIN_STD = 0.05

def _init_band_gain(gain):
    """Seed a diagonal spectral gain, in place.

    The mean keeps the branch's residual an order of magnitude below the stream
    it corrects. The proposed operator and every naive control draw from this
    same distribution, so the boundary between them is a difference of basis and
    nothing else, never of initialization -- which is how one of the controls had
    previously ended up quietly tuned.
    """
    with torch.no_grad():
        gain.normal_(BAND_GAIN_MEAN, BAND_GAIN_STD)
    return gain


def _haar_scale_masks(seq_len, num_bands, drop_dc=True):
    """Group Haar coefficients into ``num_bands`` coarse-to-fine scale bands.

    Haar coefficients arrive as one approximation value followed by detail
    blocks of length 1, 2, 4, ... so uniform coefficient bands would mix the
    coarsest approximation with several detail levels. Grouping by scale keeps
    a band physically meaningful. This is a fixed partition, not a learned or
    sample-adaptive wavelet feature. There cannot be more gain bands than Haar
    scale components: allowing that would create empty, trainable bands and
    make the apparent parameter budget larger than the effective plain filter.
    """
    seq_len = int(seq_len)
    num_bands = int(num_bands)
    if seq_len < 2 or seq_len & (seq_len - 1):
        raise ValueError("Haar transform requires a power-of-two sequence")
    component_lengths = [1]
    detail_length = 1
    while detail_length < seq_len:
        component_lengths.append(detail_length)
        detail_length *= 2
    component_count = len(component_lengths)
    if num_bands < 1 or num_bands > component_count:
        raise ValueError(
            "num_bands must be in [1, number of Haar scale components]"
        )
    masks = torch.zeros(num_bands, seq_len)
    offset = 0
    for component_index, component_length in enumerate(component_lengths):
        band = min(
            num_bands - 1,
            component_index * num_bands // component_count,
        )
        masks[band, offset:offset + component_length] = 1.0
        offset += component_length
    if drop_dc:
        masks[:, 0] = 0.0
    return masks.contiguous()


class SpectralTransform(nn.Module):
    """Parameter-free analysis / synthesis pair for one transform.

    Filtering is always ``synthesize(gain * analyze(x))``. Both the naive
    controls of the frequency-replacement table and the Expert bases route
    through this class, so a naive cell and the Expert basis on the same
    transform differ only in how the gain is parameterized -- never in the
    transform, the coefficient layout, or the DC handling. Those used to be
    three separate hand-written implementations that had quietly drifted apart:
    the STFT control had per-bin gains shared across channels while the DCT and
    Haar controls had per-channel band gains, and DCT and Haar dropped the DC
    coefficient even when the configuration told the FFT cell to keep it.

    ``analyze`` returns ``[batch, coefficient, channel]``, or
    ``[batch, coefficient, frame, channel]`` for the framed short-time
    transform, so a ``[coefficient, channel]`` gain broadcasts in both cases.
    """

    NAMES = ("fft", "dct", "haar", "stft")

    def __init__(self, seq_len, name, stft_n_fft=32):
        super().__init__()
        self.name = str(name).lower()
        if self.name not in self.NAMES:
            raise ValueError(
                f"transform must be one of {self.NAMES}, got {name}"
            )
        self.seq_len = int(seq_len)
        self.is_complex = self.name in {"fft", "stft"}
        if self.name == "fft":
            self.num_coefficients = self.seq_len // 2 + 1
        elif self.name == "dct":
            self.register_buffer(
                "basis", _orthonormal_dct_basis(self.seq_len)
            )
            self.num_coefficients = self.seq_len
        elif self.name == "haar":
            if self.seq_len < 2 or self.seq_len & (self.seq_len - 1):
                raise ValueError(
                    "Haar transform requires a power-of-two sequence"
                )
            self.num_coefficients = self.seq_len
        else:
            self.n_fft = min(int(stft_n_fft), self.seq_len)
            self.hop_length = max(1, self.n_fft // 4)
            self.register_buffer("window", torch.hann_window(self.n_fft))
            self.num_coefficients = self.n_fft // 2 + 1

    def band_masks(self, num_bands, drop_dc=True):
        """``[num_bands, num_coefficients]`` grouping that ties gains."""
        if self.name == "haar":
            return _haar_scale_masks(
                self.seq_len, num_bands, drop_dc=drop_dc
            )
        return _hard_frequency_masks(
            self.num_coefficients, num_bands, drop_dc=drop_dc
        )

    def align_gain(self, gain):
        """Make a ``[..., coefficient, channel]`` gain broadcastable."""
        return gain.unsqueeze(-2) if self.name == "stft" else gain

    def analyze(self, x):
        if self.name == "fft":
            return torch.fft.rfft(x, dim=1)
        if self.name == "dct":
            basis = self.basis.to(device=x.device, dtype=x.dtype)
            return torch.einsum("ct,btd->bcd", basis, x)
        if self.name == "haar":
            return haar_forward(x)
        batch, length, channels = x.shape
        spectrum = torch.stft(
            x.transpose(1, 2).reshape(batch * channels, length),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(device=x.device, dtype=x.dtype),
            return_complex=True,
            center=True,
        )
        return spectrum.view(
            batch, channels, *spectrum.shape[-2:]
        ).permute(0, 2, 3, 1)

    def synthesize(self, coefficients):
        if self.name == "fft":
            return torch.fft.irfft(coefficients, n=self.seq_len, dim=1)
        if self.name == "dct":
            basis = self.basis.to(
                device=coefficients.device, dtype=coefficients.dtype
            )
            return torch.einsum("ct,bcd->btd", basis, coefficients)
        if self.name == "haar":
            return haar_inverse(coefficients)
        batch, num_coefficients, frames, channels = coefficients.shape
        signal = torch.istft(
            coefficients.permute(0, 3, 1, 2).reshape(
                batch * channels, num_coefficients, frames
            ),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(
                device=coefficients.device, dtype=coefficients.real.dtype
            ),
            length=self.seq_len,
            center=True,
        )
        return signal.view(batch, channels, self.seq_len).transpose(1, 2)

    def forward(self, x, gain):
        coefficients = self.analyze(x)
        gain = self.align_gain(gain)
        if coefficients.is_complex() and not gain.is_complex():
            gain = gain.to(coefficients.dtype)
        return self.synthesize(coefficients * gain)


class NaiveSpectralFilter(nn.Module):
    """The naive control of the frequency-replacement table.

    One transform, a learnable per-band diagonal gain, the inverse transform,
    and a pointwise projection. Nothing else. This is the textbook learnable
    spectral filter -- the diagonal-in-frequency operator of FNO / GFNet -- and
    it is a **separate class from the Expert operator on purpose**: if a naive
    cell inherits any part of the proposal, the table stops measuring the
    proposal. It therefore has no dual-domain path, no lifting wavelet, no
    soft-threshold, no routing, no cross-basis term, and no learned local
    convolution.

    Local neighbourhood structure is the graph branch's job and is identical
    across every frequency cell, so withholding it from this class is not a
    handicap: the table varies only the spectral operator in the swapped slot.
    """

    TRANSFORMS = SpectralTransform.NAMES

    def __init__(
        self,
        seq_len,
        feat_dim,
        num_bands,
        dropout=0.1,
        transform="fft",
        keep_dc=False,
        stft_n_fft=32,
    ):
        super().__init__()
        self.transform = SpectralTransform(
            seq_len, transform, stft_n_fft=stft_n_fft
        )
        self.register_buffer(
            "band_masks",
            self.transform.band_masks(
                int(num_bands), drop_dc=not bool(keep_dc)
            ),
        )
        self.band_gain = nn.Parameter(
            _init_band_gain(torch.empty(int(num_bands), int(feat_dim)))
        )
        self.projection = nn.Linear(int(feat_dim), int(feat_dim))
        self.dropout = nn.Dropout(float(dropout))
        self.last_band_gain = None

    def coefficient_gain(self):
        """Per-(coefficient, channel) real gain tied within bands."""
        return torch.einsum(
            "bc,bd->cd", self.band_masks.to(self.band_gain.dtype),
            self.band_gain,
        )

    def forward(self, x):
        self.last_band_gain = self.band_gain.detach()
        filtered = self.transform(x, self.coefficient_gain().to(x.dtype))
        return self.dropout(self.projection(filtered))


class LearnableLiftingTransform(nn.Module):
    """A learnable wavelet transform built from lifting steps.

    One level of the classical lifting scheme (Sweldens' second-generation
    wavelets) splits the signal by parity and then runs two steps:

        d = x_odd  - P(x_even)          (predict: detail is what P misses)
        c = x_even + U(d)               (update:  approximation keeps the mean)

    ``P`` and ``U`` are depthwise convolutions along the energy axis, so each
    channel learns its own wavelet, **one filter pair reused at every scale** as
    in any wavelet basis, and they are **initialised to the Haar predictor and
    updater** -- ``P = identity``, ``U = 1/2`` -- which makes the
    whole cascade bit-for-bit the Haar transform of :func:`haar_forward` up to a
    per-coefficient scale. That scale cancels in ``synthesize(gain * analyze(x))``
    because diagonal matrices commute, so a filter built on this transform
    *starts exactly as the naive Haar filter* and can only move away from it.
    That matters because Haar is the strongest of the four fixed transforms on
    this data: the proposal begins at the best baseline instead of having to
    rediscover it.

    Two properties hold for *every* value of ``P`` and ``U``, not just the
    initial one, because they are structural to lifting:

    *Invertibility.* ``x_even = c - U(d)`` and ``x_odd = d + P(x_even)`` undo the
    two steps in the opposite order, whatever the convolutions contain. A fixed
    transform can only ever attenuate what its basis fails to align with; this
    one adapts the basis instead, with no risk of a non-invertible analysis.

    *Coefficient layout.* The cascade recurses on the approximation until length
    one, and coefficients are concatenated coarse-to-fine exactly as
    :func:`haar_forward` orders them, so :func:`_haar_scale_masks` groups them
    into scale bands and the gain is parameterised identically to the naive
    control's.

    ``steps`` stacks several predict/update pairs, which is how the biorthogonal
    wavelets used in practice are built -- CDF 9/7, the JPEG-2000 wavelet, is two
    lifting pairs plus a scaling. Only the first pair carries the Haar seed; the
    rest start at zero, which is a no-op, so extra depth costs nothing at
    initialisation and the Haar equivalence above survives any ``steps``. Longer
    kernels are the other axis: Haar's two-tap filter has poor frequency
    selectivity, and a wider learned predictor is what buys the smoothness that a
    fixed short basis cannot represent.

    *The coefficients are bounded, for conditioning rather than for capacity.*
    Invertibility is exact in real arithmetic for any ``P`` and ``U``, but the
    analysis cascade multiplies across levels, so far from the seed it stops
    being invertible in *floating point*: perturbing every tap by 0.5 drives the
    coefficients of a length-256 signal to 3e4 and the round-trip error to 0.5,
    which over a long run is a route to NaN rather than to a better wavelet.
    Each tap is therefore ``seed + m * tanh(raw / m)``, which pins the deviation
    inside ``+-m`` where the round-trip error stays near 1e-6. ``raw`` starts at
    zero, where the map is the identity to first order, so the Haar seed is exact
    and the gradient arrives unscaled.
    """

    def __init__(
        self, seq_len, feat_dim, kernel_size=5, steps=2, max_drift=0.25,
        hierarchical_response=False,
        contrast_rms_floor=0.0,
    ):
        super().__init__()
        seq_len = int(seq_len)
        if seq_len < 2 or seq_len & (seq_len - 1):
            raise ValueError("lifting requires a power-of-two sequence")
        self.seq_len = seq_len
        self.num_coefficients = seq_len
        self.num_levels = int(seq_len.bit_length() - 1)
        self.steps = max(1, int(steps))
        self.max_drift = float(max_drift)
        # Kept in the common constructor signature so the static lifting
        # diagnostic can occupy the same slot as adaptive Fourier lifting.
        # A classical mother wavelet is intentionally scale-stationary.
        del hierarchical_response
        del contrast_rms_floor
        feat_dim = int(feat_dim)
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            raise ValueError("lifting kernel must be odd to stay centred")
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        # ``[step, channel, 1, tap]``: one filter pair per lifting step, reused at
        # every scale. That is what a wavelet *is* -- Haar, Daubechies and CDF
        # 9/7 are each a single filter pair applied down the whole cascade -- and
        # the alternative was measured to be the expensive kind of freedom: a
        # per-level copy is eight unrelated filters, 15k parameters a block
        # against this 2k, fitted from the same data, for a basis that no longer
        # corresponds to one mother wavelet.
        shape = (self.steps, feat_dim, 1, kernel_size)
        self.predict_drift = nn.Parameter(torch.zeros(shape))
        self.update_drift = nn.Parameter(torch.zeros(shape))
        self.register_buffer(
            "predict_seed", self._seed(shape, 1.0), persistent=False
        )
        self.register_buffer(
            "update_seed", self._seed(shape, 0.5), persistent=False
        )

    @staticmethod
    def _seed(shape, centre):
        """Haar's value on the first lifting pair, zero on any extra pair."""
        seed = torch.zeros(shape)
        seed[0, :, 0, shape[-1] // 2] = float(centre)
        return seed

    def _taps(self, dtype):
        """The filter pair of every lifting step: two ``[step, D, 1, tap]``.

        Built once per transform rather than once per lifting step. The cascade
        calls ``_lift`` ``2 * levels * steps`` times -- 32 for a length-256
        signal -- and reparameterising inside it made 32 tanh launches on
        slivers of a tensor that costs one launch whole.
        """
        margin = self.max_drift
        return tuple(
            (
                getattr(self, f"{which}_seed")
                + margin
                * torch.tanh(getattr(self, f"{which}_drift") / margin)
            ).to(dtype)
            for which in ("predict", "update")
        )

    def _lift(self, taps, step, values):
        """Depthwise convolution of ``[B, D, T]`` over the position axis."""
        weight = taps[step]
        return F.conv1d(
            values, weight, padding=self.padding, groups=weight.shape[0]
        )

    def analyze(self, x):
        predict, update = self._taps(x.dtype)
        # Channels-first for the whole cascade: conv1d wants ``[B, D, T]``, and
        # transposing per lifting step meant 32 transposes and 32 copies where
        # one at each end suffices.
        current = x.transpose(1, 2)
        details = []
        for _ in range(self.num_levels):
            approximation, detail = current[..., 0::2], current[..., 1::2]
            for step in range(self.steps):
                detail = detail - self._lift(predict, step, approximation)
                approximation = approximation + self._lift(
                    update, step, detail
                )
            current = approximation
            details.append(detail)
        return torch.cat([current] + details[::-1], dim=-1).transpose(1, 2)

    def synthesize(self, coefficients):
        predict, update = self._taps(coefficients.dtype)
        banded = coefficients.transpose(1, 2)
        current = banded[..., :1]
        offset = 1
        # Coarse-to-fine, consuming the detail blocks in the order analysis
        # produced them, reversed.
        for _ in range(self.num_levels):
            length = current.shape[-1]
            detail = banded[..., offset:offset + length]
            approximation = current
            # Undo the pairs in the opposite order: update first, then predict.
            for step in reversed(range(self.steps)):
                approximation = approximation - self._lift(
                    update, step, detail
                )
                detail = detail + self._lift(predict, step, approximation)
            current = torch.stack([approximation, detail], dim=-1).reshape(
                banded.shape[0], banded.shape[1], length * 2
            )
            offset += length
        return current.transpose(1, 2)

    def forward(self, x, gain):
        return self.synthesize(self.analyze(x) * gain)


class AdaptiveFourierLiftingTransform(nn.Module):
    """Confidence-calibrated hierarchical Fourier lifting (CC-H2-SC-AFLD).

    Fixed FFT uses one stationary response over the whole XAS spectrum, while a
    fixed wavelet uses one local basis for every sample.  AFLD keeps FFT as the
    primitive but embeds it in a perfect-reconstruction lifting cascade.  At
    every scale, the odd samples are predicted from the even samples and the
    coarse signal is updated from the detail through a Fourier correction::

        P(v) = irFFT(rFFT(v) * H_p(v))
        U(v) = irFFT(rFFT(v) * H_u(v))
        d = odd  - (even + P(even))
        c = even + (0.5 d + U(d))

    ``H(v)`` is a bounded complex response represented by five coarse anchors
    on the normalized frequency axis and four learned residual midpoints.  A
    midpoint is the average of its coarse neighbours plus a zero-initialized
    residual, so the hierarchy initially reproduces the original coarse curve
    *exactly* and bends locally only when the data supports it. At each lifting
    scale the refined anchors are interpolated directly onto all native rFFT
    bins. The
    intercept learns a long-range Fourier filter and its slope is conditioned
    on the current sample's log-magnitude contrast at that exact bin. A small
    RMS floor carries the contrast's confidence into the response instead of
    standardizing numerical fluctuations on two-bin coarse spectra to unit
    strength. A shifted white line or changed post-edge oscillation therefore
    receives a different response, which no input-independent diagonal FFT
    filter can express. Unlike hard bands, the continuous representation stays
    defined at the coarsest scales instead of silently reverting to Haar. One
    compact response is shared across lifting scales; the cascade supplies
    localization without a separate STFT or wavelet path.

    All responses are zero-initialized, making the first lifting pair exactly
    Haar and every extra pair a no-op at step zero.  Both intercept and adaptive
    slope still receive first-order gradients.  Invertibility is algebraic even
    though the transform is input-dependent: synthesis knows ``d`` before
    recomputing ``U(d)``, then knows the recovered even stream before recomputing
    ``P(even)``.  No inverse matrix, router, or reconstruction loss is needed.
    """

    def __init__(
        self,
        seq_len,
        feat_dim,
        kernel_size=5,
        steps=2,
        max_response=0.25,
        hierarchical_response=False,
        contrast_rms_floor=0.0,
    ):
        super().__init__()
        seq_len = int(seq_len)
        if seq_len < 2 or seq_len & (seq_len - 1):
            raise ValueError("Fourier lifting requires a power-of-two sequence")
        self.seq_len = seq_len
        self.num_coefficients = seq_len
        self.num_levels = int(seq_len.bit_length() - 1)
        self.steps = max(1, int(steps))
        # The ``lifting_kernel`` configuration sets the number of normalized
        # response anchors.  Interpolation makes the same compact parameter set
        # meaningful for every dyadic length, including one- and two-bin coarse
        # spectra; no scale has to be disabled.
        self.num_response_anchors = max(1, int(kernel_size))
        # Compatibility alias used by response diagnostics.
        self.num_response_bands = self.num_response_anchors
        self.max_response = float(max_response)
        self.hierarchical_response = bool(hierarchical_response)
        self.contrast_rms_floor = float(contrast_rms_floor)
        # Opt-in numerical compatibility fix; SPT propagates its config flag.
        self.hermitian_irfft = False
        shape = (self.steps, int(feat_dim), self.num_response_anchors)
        for kind in ("predict", "update"):
            for component in ("real", "imag"):
                setattr(
                    self,
                    f"{kind}_{component}_base",
                    nn.Parameter(torch.zeros(shape)),
                )
                setattr(
                    self,
                    f"{kind}_{component}_slope",
                    nn.Parameter(torch.zeros(shape)),
                )
                if self.hierarchical_response:
                    # Four midpoint residuals refine five coarse anchors. At
                    # zero the interleaved nine-anchor curve is exactly the
                    # same piecewise-linear response as the five-anchor
                    # curve, while weight decay gives the fine residual an
                    # explicit preference to stay smooth unless data supports
                    # a local bend.
                    midpoint_shape = (
                        self.steps,
                        int(feat_dim),
                        self.num_response_anchors - 1,
                    )
                    for role in ("base", "slope"):
                        setattr(
                            self,
                            f"{kind}_{component}_{role}_midpoint",
                            nn.Parameter(torch.zeros(midpoint_shape)),
                        )

    @staticmethod
    def _native_response(anchors, frequencies):
        """Interpolate ``[channel, anchor]`` responses to native rFFT bins."""
        if frequencies == 1:
            # The single coefficient is DC (normalised frequency zero), so its
            # response is the first anchor rather than an average over the axis.
            return anchors[:, :1]
        if anchors.shape[-1] == 1:
            return anchors.expand(-1, frequencies)
        return F.interpolate(
            anchors.unsqueeze(0),
            size=frequencies,
            mode="linear",
            align_corners=True,
        ).squeeze(0)

    def _hierarchical_anchors(self, anchors, midpoint):
        """Interleave coarse anchors with learned midpoint residuals."""
        if anchors.shape[-1] < 2:
            return anchors
        middle = 0.5 * (anchors[..., :-1] + anchors[..., 1:]) + midpoint
        refined = anchors.new_empty(
            *anchors.shape[:-1], 2 * anchors.shape[-1] - 1
        )
        refined[..., 0::2] = anchors
        refined[..., 1::2] = middle
        return refined

    def _correction(self, kind, step, values):
        """Sample-conditioned complex Fourier correction at one lifting scale."""
        length = values.shape[-1]
        spectrum = torch.fft.rfft(values, dim=-1)
        magnitude = torch.log1p(spectrum.abs())
        centered = magnitude - magnitude.mean(dim=-1, keepdim=True)
        variance = centered.square().mean(dim=-1, keepdim=True)
        if self.contrast_rms_floor > 0.0:
            # Standardising every dyadic scale to variance one turns tiny
            # two-bin fluctuations into full-strength sample modulation.  A
            # small RMS floor suppresses only low-evidence contrast while
            # converging to the original z-score on ordinary spectra.
            denominator = variance.add(
                self.contrast_rms_floor ** 2 + 1e-6
            ).sqrt()
        else:
            denominator = variance.add(1e-6).sqrt()
        contrast = centered / denominator

        roles = ["base", "slope"]
        # Interpolate every real/imag response family in one kernel launch.
        # The old nested loop launched F.interpolate four times per lifting
        # correction, at 16 corrections per block and six blocks. Stacking is
        # exactly the same linear operation and removes that avoidable
        # small-FFT orchestration overhead.
        anchors = torch.stack(
            [
                torch.stack(
                    [
                        getattr(self, f"{kind}_{component}_{role}")[step]
                        for component in ("real", "imag")
                    ],
                    dim=0,
                )
                for role in roles
            ],
            dim=0,
        )
        role_count, component_count, channels, anchor_count = anchors.shape
        if self.hierarchical_response:
            midpoint = torch.stack(
                [
                    torch.stack(
                        [
                            getattr(
                                self,
                                f"{kind}_{component}_{role}_midpoint",
                            )[step]
                            for component in ("real", "imag")
                        ],
                        dim=0,
                    )
                    for role in roles
                ],
                dim=0,
            )
            anchors = self._hierarchical_anchors(anchors, midpoint)
            anchor_count = anchors.shape[-1]
        native = self._native_response(
            anchors.reshape(-1, anchor_count), spectrum.shape[-1]
        ).reshape(
            role_count, component_count, channels, spectrum.shape[-1]
        )
        contrast = contrast.unsqueeze(1)
        raw_response = (
            native[0].unsqueeze(0) + native[1].unsqueeze(0) * contrast
        )
        # Bounded like the static lifting taps, but identity-scaled at the
        # origin: d[m*tanh(raw/m)]/d raw = 1.
        band_response = self.max_response * torch.tanh(
            raw_response / self.max_response
        )
        response = torch.complex(
            band_response[:, 0], band_response[:, 1]
        )
        filtered = spectrum * response
        if self.hermitian_irfft:
            filtered = project_real_fft_endpoints(filtered, length)
        return torch.fft.irfft(filtered, n=length, dim=-1)

    def _predict(self, step, approximation):
        base = approximation if step == 0 else 0.0
        return base + self._correction("predict", step, approximation)

    def _update(self, step, detail):
        base = 0.5 * detail if step == 0 else 0.0
        return base + self._correction("update", step, detail)

    def analyze(self, x):
        current = x.transpose(1, 2)
        details = []
        for _ in range(self.num_levels):
            approximation, detail = current[..., 0::2], current[..., 1::2]
            for step in range(self.steps):
                detail = detail - self._predict(step, approximation)
                approximation = approximation + self._update(step, detail)
            current = approximation
            details.append(detail)
        return torch.cat([current] + details[::-1], dim=-1).transpose(1, 2)

    def synthesize(self, coefficients):
        banded = coefficients.transpose(1, 2)
        current = banded[..., :1]
        offset = 1
        for _ in range(self.num_levels):
            length = current.shape[-1]
            detail = banded[..., offset:offset + length]
            approximation = current
            for step in reversed(range(self.steps)):
                approximation = approximation - self._update(step, detail)
                detail = detail + self._predict(step, approximation)
            current = torch.stack([approximation, detail], dim=-1).reshape(
                banded.shape[0], banded.shape[1], length * 2
            )
            offset += length
        return current.transpose(1, 2)

    def forward(self, x, gain):
        return self.synthesize(self.analyze(x) * gain)


class LiftingSpectralOperator(nn.Module):
    """The Expert spectral module: a learned wavelet with learned shrinkage.

        c    = analyze(x)                        learned lifting wavelet
        s    = mean |c| within each scale band   per sample and channel
        c'   = g * sign(c) * relu(|c| - tau * s) shrink, then gain
        out  = projection( synthesize(c') )

    Structurally this is the naive control with two substitutions, and it is
    identical to it everywhere else: the gain ``g`` is band-tied over Haar scale
    bands exactly as the control's is, and the read-out is the control's single
    pointwise projection. Both substitutions are inert at initialisation --
    ``tau`` starts at zero, where shrinkage is the identity, and the wavelet
    starts at Haar -- so **the proposed cell and the naive Haar cell are the same
    function before training**. That is the cleanest form the frequency claim can
    take: not "more capacity wins" but "a basis that is learned, and used
    nonlinearly, beats every fixed one used linearly".

    *Why shrinkage, and not a larger gain.* A diagonal gain in an invertible
    basis is ``S diag(g) A``. If ``g`` is constant the basis cancels exactly and
    the operator is ``g * x`` whatever the wavelet is, so the *only* thing that
    carries gradient to a learned basis is the variation in ``g``, which at
    initialisation is a spread of 0.05 around a mean of 0.10. Measured, the
    lifting parameters received a gradient 100x weaker than the gain's: the basis
    was learnable in principle and starved in practice. Soft shrinkage is the
    classical wavelet denoiser (Donoho-Johnstone) and it fixes this at the root,
    because it *rewards a basis in which the signal is sparse*. Sparsity is a
    property of the basis, so now the wavelet has a first-order effect on the
    loss instead of a second-order one.

    *The threshold is relative to the band's own coefficient scale.* An absolute
    threshold cannot work here: orthonormal wavelet coefficients fall away by
    orders of magnitude from the coarsest band to the finest, so one number is
    simultaneously far too large for the fine scales and far too small for the
    coarse ones. Measured, a fixed threshold of 0.5 against coefficients of
    magnitude 6 removed nothing at all and left the basis's gradient exactly
    where it was. Scaling by the mean absolute coefficient of that band -- the
    same quantity classical BayesShrink and SureShrink estimate from the data --
    makes ``tau`` a dimensionless "fraction of typical magnitude", well
    conditioned for every band at once. The scale is detached, so it acts as a
    normaliser rather than as another route for gradient.

    It also buys the two things no cell of the frequency table can imitate.
    Every naive control is a linear, input-independent operator -- a fixed
    transform, a diagonal gain, an inverse -- so all four are the same *kind* of
    object and differ only in which basis they are diagonal in. Shrinkage is
    nonlinear, and it is the nonlinearity that makes wavelets the tool of choice
    for signals with sharp localised features: it keeps coefficients that stand
    above the threshold and removes the rest, which preserves an absorption edge
    while suppressing what surrounds it. And because the scale is measured per
    sample, the operator adapts to the spectrum in front of it. No diagonal
    linear filter can do either at any gain.

    This replaced a routed convex mixture of four fixed bases, which failed for
    a reason worth recording. A convex mixture cannot exceed its best member by
    much, and measured on the 300-epoch grid it did not: clamping the router
    one-hot showed the favoured basis alone scoring within 0.003 of the whole
    mixture. Worse, each basis's gradient is scaled by its own mixture weight,
    so the unfavoured ones trained at a fraction of the rate and ended far
    weaker inside the mixture than the same transform trained alone -- by 0.04
    on metal:CN2 and up to 0.16 on metal:MOOD. Averaging one healthy operator
    with three starved ones lands at or below a single sharp filter. Mixing was
    the wrong axis.
    """

    def __init__(
        self,
        seq_len,
        feat_dim,
        num_bands,
        dropout=0.1,
        keep_dc=False,
        kernel_size=5,
        steps=2,
        shrinkage=True,
        hierarchical_response=False,
        contrast_rms_floor=0.0,
        transform_class=LearnableLiftingTransform,
    ):
        super().__init__()
        feat_dim = int(feat_dim)
        self.shrinkage = bool(shrinkage)
        self.transform = transform_class(
            seq_len,
            feat_dim,
            kernel_size=kernel_size,
            steps=steps,
            hierarchical_response=hierarchical_response,
            contrast_rms_floor=contrast_rms_floor,
        )
        self.register_buffer(
            "band_masks",
            _haar_scale_masks(
                self.transform.seq_len,
                int(num_bands),
                drop_dc=not bool(keep_dc),
            ),
        )
        self.band_gain = nn.Parameter(
            _init_band_gain(torch.empty(int(num_bands), feat_dim))
        )
        # ``shrinkage=False`` is the proposal's own ablation: a learned basis used
        # linearly, which separates "the wavelet is adapted" from "the wavelet is
        # used nonlinearly". Not a cell of either published table.
        self.threshold_logit = (
            nn.Parameter(torch.zeros(int(num_bands), feat_dim))
            if self.shrinkage
            else None
        )
        self.register_buffer(
            "band_sizes", self.band_masks.sum(dim=1).clamp_min(1.0)
        )
        self.projection = nn.Linear(feat_dim, feat_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.last_band_gain = None
        self.last_band_threshold = None

    def _per_coefficient(self, banded):
        """Spread a ``[band, channel]`` parameter over coefficients."""
        return torch.einsum(
            "bc,bd->cd", self.band_masks.to(banded.dtype), banded
        )

    def coefficient_gain(self):
        """Per-(coefficient, channel) real gain tied within scale bands."""
        return self._per_coefficient(self.band_gain)

    def band_threshold(self):
        """Threshold per band, as a fraction of that band's typical magnitude.

        Bounded to ``(-1, 1)`` rather than left free. Shrinkage at one mean
        absolute coefficient already removes most of a band, and beyond that the
        operator starts deleting bands outright -- a zeroed coefficient has no
        gradient, so a threshold that overshoots is not recoverable. ``tanh`` is
        the identity to first order at zero, so the bound costs neither the exact
        Haar initialisation nor the gradient that has to carry ``tau`` off it,
        which a softplus or a clamp at zero would both destroy.
        """
        if self.threshold_logit is None:
            return None
        return torch.tanh(self.threshold_logit)

    def coefficient_threshold(self, coefficients):
        """Per-(sample, coefficient, channel) threshold, in units of the band."""
        masks = self.band_masks.to(coefficients.dtype)
        band_scale = torch.einsum(
            "bc,ncd->nbd", masks, coefficients.abs().detach()
        ) / self.band_sizes.to(coefficients.dtype).view(1, -1, 1)
        return torch.einsum(
            "bc,nbd->ncd", masks, self.band_threshold() * band_scale
        )

    def shrink(self, coefficients):
        if self.threshold_logit is None:
            return coefficients
        return torch.sign(coefficients) * F.relu(
            coefficients.abs() - self.coefficient_threshold(coefficients)
        )

    def forward(self, x):
        self.last_band_gain = self.band_gain.detach()
        threshold = self.band_threshold()
        self.last_band_threshold = (
            None if threshold is None else threshold.detach()
        )
        filtered = self.transform.synthesize(
            self.shrink(self.transform.analyze(x))
            * self.coefficient_gain().to(x.dtype)
        )
        return self.dropout(self.projection(filtered))


class AdaptiveFourierLiftingSpectralOperator(LiftingSpectralOperator):
    """The proposed XAS-adaptive Fourier frequency decomposition.

    This is one perfect-reconstruction multiscale operator, not a collection of
    FFT/STFT/wavelet branches. It differs from a naive FFT or Haar cell through
    one mechanism: sample-conditioned complex Fourier lifting responses create
    a localized, nonlinear, perfectly reconstructing transform. Scale-relative
    shrinkage is the coefficient estimator used on that transform. At
    initialization both extensions are inert, so with a shared gain and
    projection the whole cell is exactly the naive Haar function.
    """

    def __init__(self, *args, **kwargs):
        kwargs["transform_class"] = AdaptiveFourierLiftingTransform
        super().__init__(*args, **kwargs)


class FramedOscillatoryBank(nn.Module):
    """Multi-window STFT bank with sample-adaptive complementary shaping.

    Not a naive STFT cell: two fixed Hann windows always run in parallel, each
    with two complex per-bin responses. A bounded magnitude-contrast term makes
    their complementary mask depend on the sample and frame, so

    ``m(x) * g_trend + (1 - m(x)) * g_seasonal``

    cannot collapse algebraically into the one static real gain used by the
    naive STFT control. The imaginary responses start at zero and learn local
    phase shifts, which are useful when an absorption edge or white-line peak
    moves along the energy axis. There is no soft-threshold and no path router:
    both windows and both coefficient components receive gradients every step.
    """

    def __init__(self, seq_len, feat_dim, stft_n_fft=64, keep_dc=False):
        super().__init__()
        feat_dim = int(feat_dim)
        base = max(8, min(int(stft_n_fft), int(seq_len)))
        wide = min(2 * base, int(seq_len))
        narrow = max(8, base // 2)
        sizes = []
        for size in (base, wide if wide != base else narrow):
            if size not in sizes:
                sizes.append(size)
        if len(sizes) < 2:
            sizes.append(max(8, sizes[0] // 2 if sizes[0] > 16 else sizes[0]))
        self.transforms = nn.ModuleList(
            [
                SpectralTransform(seq_len, "stft", stft_n_fft=size)
                for size in sizes
            ]
        )
        self.trend_gain = nn.ParameterList()
        self.seasonal_gain = nn.ParameterList()
        self.trend_phase = nn.ParameterList()
        self.seasonal_phase = nn.ParameterList()
        self.disentangle_logit = nn.ParameterList()
        self.magnitude_slope = nn.ParameterList()
        self.drop_dc = not bool(keep_dc)
        for transform in self.transforms:
            bins = transform.num_coefficients
            self.trend_gain.append(
                nn.Parameter(_init_band_gain(torch.empty(bins, feat_dim)))
            )
            self.seasonal_gain.append(
                nn.Parameter(_init_band_gain(torch.empty(bins, feat_dim)))
            )
            # A complex response strictly contains the naive real diagonal
            # response. Its phase starts at zero, so the extension is inert at
            # initialization rather than injecting an arbitrary shift.
            self.trend_phase.append(
                nn.Parameter(torch.zeros(bins, feat_dim))
            )
            self.seasonal_phase.append(
                nn.Parameter(torch.zeros(bins, feat_dim))
            )
            # At initialization m=0.5 and the sample-adaptive term is zero. The
            # two independently seeded responses make the slope's first
            # derivative non-zero, so it does not sit in a dead zone.
            logit = torch.zeros(bins, feat_dim)
            if self.drop_dc:
                logit[0].fill_(-8.0)
            self.disentangle_logit.append(nn.Parameter(logit))
            self.magnitude_slope.append(
                nn.Parameter(torch.zeros(bins, feat_dim))
            )

    @staticmethod
    def _complex_response(real, imag, dtype):
        return (real + 1j * imag).to(dtype)

    @staticmethod
    def _magnitude_contrast(coefficients):
        """Standardized log-magnitude within each framed frequency bin."""
        magnitude = torch.log1p(coefficients.abs())
        centered = magnitude - magnitude.mean(dim=2, keepdim=True)
        scale = centered.square().mean(dim=2, keepdim=True).add(1e-6).sqrt()
        return centered / scale

    def forward(self, x):
        outputs = []
        for transform, trend, seasonal, trend_phase, seasonal_phase, logit, slope in zip(
            self.transforms,
            self.trend_gain,
            self.seasonal_gain,
            self.trend_phase,
            self.seasonal_phase,
            self.disentangle_logit,
            self.magnitude_slope,
        ):
            coefficients = transform.analyze(x)
            contrast = self._magnitude_contrast(coefficients)
            mask = torch.sigmoid(
                logit[None, :, None, :]
                + torch.tanh(slope)[None, :, None, :] * contrast
            ).to(coefficients.real.dtype)
            trend_response = self._complex_response(
                trend, trend_phase, coefficients.dtype
            )[None, :, None, :]
            seasonal_response = self._complex_response(
                seasonal, seasonal_phase, coefficients.dtype
            )[None, :, None, :]
            # The split is coefficient-wise and input-dependent, not a
            # sample-level choice between paths. Both responses remain active.
            gain = mask * trend_response + (1.0 - mask) * seasonal_response
            if self.drop_dc:
                gain = gain.clone()
                gain[:, 0].zero_()
            outputs.append(transform.synthesize(coefficients * gain))
        # Primary window at full strength (naive-STFT neighbourhood); extra
        # windows add as half-scale residuals so the bank remains a strict
        # superset rather than an equal average that dilutes the best window.
        primary = outputs[0]
        if len(outputs) == 1:
            return primary
        return primary + 0.5 * sum(outputs[1:])


class ComplementarySpectralOperator(nn.Module):
    """Complementary Time-Frequency Spectral Decomposition (CSD-TF).

    Three paths always run in parallel without a router:

        y_t = Proj_t( LN( multi_window_STFT_bank(x) ) )
        y_l = Proj_l( LN( synthesize( shrink( lifting(x) ) * g_l ) ) )
        y_g = Proj_g( LN( irFFT( rFFT(x) * g_g ) ) )
        out = y_t + y_l + y_g

    Per-path LayerNorm puts the framed bank, the Haar-seeded lifting path and
    the linear Fourier path on comparable scales. The two localized paths start
    at full strength; the redundant global path starts as a half-scale residual
    but its full matrix remains learnable. This is a fixed initialization, not a
    softmax or convex path router. Soft-thresholding is local-only and every path
    receives a gradient from the first step.
    """

    def __init__(
        self,
        seq_len,
        feat_dim,
        num_bands,
        dropout=0.1,
        keep_dc=False,
        kernel_size=5,
        steps=2,
        shrinkage=True,
        stft_n_fft=64,
    ):
        super().__init__()
        feat_dim = int(feat_dim)
        num_bands = int(num_bands)
        self.shrinkage = bool(shrinkage)

        self.local_transform = LearnableLiftingTransform(
            seq_len, feat_dim, kernel_size=kernel_size, steps=steps
        )
        self.register_buffer(
            "local_band_masks",
            _haar_scale_masks(
                self.local_transform.seq_len,
                num_bands,
                drop_dc=not bool(keep_dc),
            ),
        )
        self.local_band_gain = nn.Parameter(
            _init_band_gain(torch.empty(num_bands, feat_dim))
        )
        self.local_threshold_logit = (
            nn.Parameter(torch.zeros(num_bands, feat_dim))
            if self.shrinkage
            else None
        )
        self.register_buffer(
            "local_band_sizes",
            self.local_band_masks.sum(dim=1).clamp_min(1.0),
        )

        self.stft_bank = FramedOscillatoryBank(
            seq_len,
            feat_dim,
            stft_n_fft=stft_n_fft,
            keep_dc=keep_dc,
        )

        self.global_transform = SpectralTransform(seq_len, "fft")
        self.register_buffer(
            "global_band_masks",
            self.global_transform.band_masks(
                num_bands, drop_dc=not bool(keep_dc)
            ),
        )
        self.global_band_gain = nn.Parameter(
            _init_band_gain(torch.empty(num_bands, feat_dim))
        )
        # Real gains cannot change phase. A zero-initialized imaginary response
        # keeps the cold operator unchanged while making learned translations
        # and asymmetric edge responses reachable.
        self.global_band_phase = nn.Parameter(
            torch.zeros(num_bands, feat_dim)
        )

        self.tf_norm = nn.LayerNorm(feat_dim)
        self.local_norm = nn.LayerNorm(feat_dim)
        self.global_norm = nn.LayerNorm(feat_dim)
        self.tf_projection = nn.Linear(feat_dim, feat_dim)
        self.local_projection = nn.Linear(feat_dim, feat_dim)
        self.global_projection = nn.Linear(feat_dim, feat_dim)
        for projection, scale in (
            (self.tf_projection, 1.0),
            (self.local_projection, 1.0),
            (self.global_projection, 0.5),
        ):
            nn.init.eye_(projection.weight)
            projection.weight.data.mul_(scale)
            nn.init.zeros_(projection.bias)
        self.dropout = nn.Dropout(float(dropout))
        self.last_band_gain = None
        self.last_band_threshold = None

    @staticmethod
    def _spread(masks, banded):
        return torch.einsum("bc,bd->cd", masks.to(banded.dtype), banded)

    def _threshold(self, masks, sizes, logit, coefficients):
        if logit is None:
            return None
        tau = torch.tanh(logit)
        band_scale = torch.einsum(
            "bc,ncd->nbd", masks.to(coefficients.dtype),
            coefficients.abs().detach(),
        ) / sizes.to(coefficients.dtype).view(1, -1, 1)
        return torch.einsum("bc,nbd->ncd", masks.to(coefficients.dtype), tau * band_scale)

    def _shrink(self, coefficients, masks, sizes, logit):
        threshold = self._threshold(masks, sizes, logit, coefficients)
        if threshold is None:
            return coefficients
        return torch.sign(coefficients) * F.relu(
            coefficients.abs() - threshold
        )

    def _local_path(self, x):
        coefficients = self.local_transform.analyze(x)
        shrunk = self._shrink(
            coefficients,
            self.local_band_masks,
            self.local_band_sizes,
            self.local_threshold_logit,
        )
        gain = self._spread(
            self.local_band_masks, self.local_band_gain
        ).to(x.dtype)
        return self.local_transform.synthesize(shrunk * gain)

    def _tf_path(self, x):
        return self.stft_bank(x)

    def _global_path(self, x):
        gain_real = self._spread(
            self.global_band_masks, self.global_band_gain
        ).to(x.dtype)
        gain_imag = self._spread(
            self.global_band_masks, self.global_band_phase
        ).to(x.dtype)
        gain = torch.complex(gain_real, gain_imag)
        return self.global_transform(x, gain)

    def forward(self, x):
        self.last_band_gain = torch.stack(
            [
                self.local_band_gain.detach(),
                self.global_band_gain.detach(),
            ]
        )
        if self.shrinkage and self.local_threshold_logit is not None:
            self.last_band_threshold = torch.tanh(
                self.local_threshold_logit
            ).detach()
        else:
            self.last_band_threshold = None
        return self.dropout(
            self.tf_projection(self.tf_norm(self._tf_path(x)))
            + self.local_projection(self.local_norm(self._local_path(x)))
            + self.global_projection(self.global_norm(self._global_path(x)))
        )


class NestedSpectralAssociation(nn.Module):
    """The spectral branch: the Expert operator, or one naive filter in its place.

    ``decomposition='complementary'`` builds the single-path
    :class:`AdaptiveFourierLiftingSpectralOperator` (AFLD). ``lifting`` keeps
    the earlier static learned wavelet for diagnostics. The other four values
    build :class:`NaiveSpectralFilter` on that fixed transform -- always that
    class -- so the replacement table swaps the frequency method without
    handing the control any part of the proposal.
    """

    DECOMPOSITIONS = (
        "complementary",
        "lifting",
    ) + NaiveSpectralFilter.TRANSFORMS

    def __init__(
        self,
        seq_len,
        feat_dim,
        num_bands,
        stft_n_fft=32,
        dropout=0.1,
        decomposition="complementary",
        keep_dc=False,
        lifting_kernel=5,
        lifting_steps=2,
        lifting_shrinkage=True,
        lifting_contrast_rms_floor=0.0,
    ):
        super().__init__()
        self.decomposition = str(decomposition).lower()
        if self.decomposition not in self.DECOMPOSITIONS:
            raise ValueError(
                "spectral decomposition must be one of "
                f"{self.DECOMPOSITIONS}, got {decomposition}"
            )
        self.complementary = None
        self.lifting = None
        for transform in NaiveSpectralFilter.TRANSFORMS:
            setattr(self, transform, None)
        shared = dict(
            seq_len=seq_len,
            feat_dim=feat_dim,
            num_bands=num_bands,
            dropout=dropout,
            keep_dc=keep_dc,
            kernel_size=lifting_kernel,
            steps=lifting_steps,
            shrinkage=lifting_shrinkage,
            contrast_rms_floor=lifting_contrast_rms_floor,
        )
        if self.decomposition == "complementary":
            self.complementary = AdaptiveFourierLiftingSpectralOperator(
                **shared, hierarchical_response=True
            )
        elif self.decomposition == "lifting":
            self.lifting = LiftingSpectralOperator(**shared)
        else:
            setattr(
                self,
                self.decomposition,
                NaiveSpectralFilter(
                    seq_len=seq_len,
                    feat_dim=feat_dim,
                    num_bands=num_bands,
                    dropout=dropout,
                    transform=self.decomposition,
                    keep_dc=keep_dc,
                    stft_n_fft=stft_n_fft,
                ),
            )

    def forward(self, x):
        return getattr(self, self.decomposition)(x)


class SpecBlock(nn.Module):
    """Spectral and physical-graph branches with an identity-safe residual path.

    ``force_expert`` selects which branches run: ``identity``, ``fft``,
    ``graph`` or ``fft_graph``. Those four values are exactly the four cells of
    the module ablation, and they share one implementation so the cells differ
    only in which branch is applied.

    ``branch_fusion="allocated"`` preserves the existing channel allocation and
    sample-adaptive gates. The optional ``"residual"`` candidate removes those
    fusion parameters and lets both branches write full-width updates. Both
    operators return update messages, not identity-plus-update tensors: their
    outputs must therefore NOT have the input subtracted. With parallel
    composition and zero spectral context the candidate is simply
    ``x + scale_s * spectral(LN(x)) + scale_g * graph(LN(x))``,
    with residual dropout and stochastic depth retained. Zero spectral context
    is fixed off in this candidate; nonzero context remains learnable.

    **The spectral branch is the frequency operator and nothing else.** It used
    to also carry a multi-scale depthwise convolution (kernels 3/5/9), which cost
    30k parameters against the operator's 26k -- so more than half of the branch
    was a local convolution that every cell of the frequency-replacement table
    shared identically. That table then swapped 16% of a block while 19% of it
    sat alongside doing much the same local work, which is enough to explain why
    the five cells came out within noise of each other and the winner changed
    from descriptor to descriptor. Local structure now belongs to the graph
    branch, which is where a neighbourhood operator belongs, and it is constant
    across the frequency table by construction.
    """

    def __init__(
        self,
        leng,
        seq_len,
        pred_len,
        top_k,
        d_model,
        d_ff,
        num_kernels,
        device,
        lowpass_bins=None,
        spectral_decomposition="complementary",
        spectral_num_bands=6,
        stft_n_fft=32,
        branch_gate_init=0.05,
        spectral_gate_init=None,
        graph_gate_init=None,
        residual_dropout=0.1,
        graph_dropout=0.1,
        graph_neighbors=3,
        graph_heads=4,
        graph_dilations=None,
        graph_max_dilation=8,
        graph_global_heads=None,
        graph_global_k=8,
        drop_path=0.0,
        keep_dc=False,
        compose_mode="parallel",
        graph_spectral_context=0.1,
        disjoint_branch_channels=True,
        graph_channel_fraction=0.5,
        allocation_logit_init=2.0,
        lifting_kernel=5,
        lifting_steps=2,
        lifting_shrinkage=True,
        lifting_contrast_rms_floor=0.0,
        branch_fusion="allocated",
    ):
        super().__init__()
        self.branch_fusion = str(branch_fusion).lower()
        if self.branch_fusion not in {"allocated", "residual"}:
            raise ValueError("branch_fusion must be 'allocated' or 'residual'")
        decomp = str(spectral_decomposition).lower()
        self.compose_mode = str(compose_mode).lower()
        if self.compose_mode not in {"sequential", "parallel"}:
            raise ValueError(
                "compose_mode must be 'sequential' or 'parallel'"
            )
        self.spectral_norm = nn.LayerNorm(d_model)
        self.graph_norm = nn.LayerNorm(d_model)
        # The frequency table changes only this slot. Construct it in a forked
        # RNG stream so a larger Expert operator cannot shift initialization of
        # the graph, later blocks, pooling head, or any other shared parameter.
        # With the same training seed those common tensors are now bit-identical
        # in all five cells, rather than merely having the same architecture.
        with torch.random.fork_rng(devices=[]):
            self.spectral = NestedSpectralAssociation(
                seq_len=leng,
                feat_dim=d_model,
                num_bands=int(spectral_num_bands),
                stft_n_fft=int(stft_n_fft),
                dropout=float(residual_dropout),
                decomposition=decomp,
                keep_dc=bool(keep_dc),
                lifting_kernel=int(lifting_kernel),
                lifting_steps=int(lifting_steps),
                lifting_shrinkage=bool(lifting_shrinkage),
                lifting_contrast_rms_floor=float(lifting_contrast_rms_floor),
            )
        self.graph = MultiHeadEnergyGraph(
            feat_dim=d_model,
            num_neighbors=int(graph_neighbors),
            device=device,
            dropout=float(graph_dropout),
            num_heads=int(graph_heads),
            dilations=graph_dilations,
            max_dilation=int(graph_max_dilation),
            global_heads=graph_global_heads,
            global_k=int(graph_global_k),
        )
        # Sample-adaptive *modulation* around 1.0, not an on/off gate.
        # Multiplying two near-zero-initialized factors (sigmoid gate x ReZero
        # scale) created a gradient dead zone that permanently froze the graph
        # branch, so the modulation is centred at 1 and the branch strength is
        # carried solely by ``spectral_scale`` / ``graph_scale``.
        # The residual candidate is deliberately a smaller fusion mechanism:
        # full-width updates with per-channel LayerScale, and no sample gate.
        # Keep construction order unchanged for legacy allocated checkpoints.
        self.residual_gate = None
        if self.branch_fusion == "allocated":
            self.residual_gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 2),
            )
            nn.init.zeros_(self.residual_gate[-1].weight)
            nn.init.zeros_(self.residual_gate[-1].bias)
        spectral_init = float(
            1.0 if spectral_gate_init is None else spectral_gate_init
        )
        # Graph starts small but ALIVE so it receives usable gradient. Kept
        # deliberately below the spectral scale so near-saturated descriptors
        # cannot have the graph rewrite what the spectral branch already
        # explained (the failure mode behind metal:CN's graph_only > full).
        graph_init = float(
            0.15 if graph_gate_init is None else graph_gate_init
        )
        if graph_init == 0.0:
            graph_init = 0.15
        # Read-full / write-complementary: both branches see the whole stream
        # and may write anywhere, but their per-channel shares are tied to sum
        # to one, so a channel can never be claimed by both at full strength.
        self.disjoint_channels = (
            bool(disjoint_branch_channels) and self.branch_fusion == "allocated"
        )
        if self.disjoint_channels:
            self.spectral_width, self.graph_width = channel_split_sizes(
                d_model, graph_channel_fraction
            )
            self.spectral_write = nn.Linear(d_model, d_model)
            self.graph_write = nn.Linear(d_model, d_model)
            # Start polarised along the nominal boundary so the block begins
            # where a hard split begins, then let the data move the boundary.
            sharpness = float(allocation_logit_init)
            logit = torch.full((d_model,), -sharpness)
            logit[: self.spectral_width] = sharpness
            self.allocation_logit = nn.Parameter(logit)
        else:
            self.spectral_width = self.graph_width = d_model
            self.spectral_write = None
            self.graph_write = None
            self.allocation_logit = None
        self.spectral_scale = nn.Parameter(
            torch.full((d_model,), spectral_init)
        )
        self.graph_scale = nn.Parameter(torch.full((d_model,), graph_init))
        # Per-channel amount of spectral residual the graph branch reads as
        # context. Unlike a multiplicative output gate this has no dead zone at
        # zero: its gradient is proportional to the spectral residual itself.
        self.spectral_context = (
            None
            if graph_spectral_context is None or (
                self.branch_fusion == "residual"
                and float(graph_spectral_context) == 0.0
            )
            else nn.Parameter(
                torch.full((d_model,), float(graph_spectral_context))
            )
        )
        self.residual_dropout = nn.Dropout(float(residual_dropout))
        self.spectral_drop_path = DropPath(float(drop_path))
        self.graph_drop_path = DropPath(float(drop_path))
        self.last_residual_gate = None

    def _gates(self, x):
        if self.residual_gate is None:
            # Preserve the diagnostic shape without a trainable gate.
            gates = x.new_ones((x.shape[0], 2))
        else:
            gates = 1.0 + torch.tanh(self.residual_gate(x.mean(dim=1)))
        self.last_residual_gate = gates.detach()
        return gates

    def _spectral_candidate(self, x, gate):
        delta = self.spectral(self.spectral_norm(x))
        if self.spectral_write is not None:
            delta = self.spectral_allocation() * self.spectral_write(delta)
        return x + self.spectral_drop_path(
            gate[:, None, 0:1]
            * self.spectral_scale
            * self.residual_dropout(delta)
        )

    def _graph_candidate(self, x, energy, gate, spectral_residual=None):
        # Under parallel composition both branches read the same stream, so the
        # graph has to rediscover from ``x`` alone which structure the spectral
        # branch already explained. On descriptors where the spectral branch
        # nearly saturates the task there is nothing left for it to find, and
        # its contribution collapses. Letting it read the spectral residual
        # makes it a genuine residual learner while it still *writes* into the
        # identity path, which is what keeps it from blurring spectral peaks.
        context = x
        if spectral_residual is not None and self.spectral_context is not None:
            context = x + self.spectral_context * spectral_residual
        graph_delta = self.graph(self.graph_norm(context), energy=energy)
        if self.graph_write is not None:
            graph_delta = (
                1.0 - self.spectral_allocation()
            ) * self.graph_write(graph_delta)
        return x + self.graph_drop_path(
            gate[:, None, 1:2]
            * self.graph_scale
            * self.residual_dropout(graph_delta)
        )

    def spectral_allocation(self):
        """Per-channel share of the spectral branch; the graph gets ``1 - m``."""
        if self.allocation_logit is None:
            return None
        return torch.sigmoid(self.allocation_logit)

    def forward(self, x, energy=None, force_expert="fft_graph"):
        expert = str(force_expert).lower()
        if expert not in {"identity", "fft", "graph", "fft_graph"}:
            raise ValueError(
                "force_expert must be identity, fft, graph, or fft_graph"
            )
        sequence_out = x
        if expert == "identity":
            return sequence_out

        gates = self._gates(sequence_out)
        if expert == "graph":
            return self._graph_candidate(sequence_out, energy, gates)
        spectral_out = self._spectral_candidate(sequence_out, gates)
        if expert == "fft":
            return spectral_out
        if self.compose_mode == "parallel":
            # Parallel residual fusion (LOGLO-FNO / AFNO style): spectral and
            # graph both correct the same identity path, so the graph cannot
            # blur the spectral peak representation the way it would if the two
            # were composed in sequence.
            graph_from_identity = self._graph_candidate(
                sequence_out,
                energy,
                gates,
                spectral_residual=spectral_out - sequence_out,
            )
            return spectral_out + (graph_from_identity - sequence_out)
        return self._graph_candidate(spectral_out, energy, gates)
