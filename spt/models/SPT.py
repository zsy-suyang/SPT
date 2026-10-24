import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from spt.models.specnet import AdaptiveFourierLiftingTransform, SpecBlock, channel_split_sizes
import torch.nn.init as init 


def ordinal_cdf_squared_distance(prediction, target, class_values):
    """Per-example squared CDF distance in physical label units.

    Class indices need not follow numeric order. Integrating over adjacent
    physical-value gaps handles nonconsecutive count classes without treating
    an encoded class index as a physical count. The result is not normalized by
    the source class span. This compares the full predicted distribution with
    the exact target, not only their expected values.
    """
    values = class_values.to(device=prediction.device, dtype=torch.float32)
    if values.ndim != 1 or values.numel() != prediction.shape[-1]:
        raise ValueError("CDF ordinal loss requires one physical value per class")
    ordered_values, order = values.sort()
    gaps = ordered_values[1:] - ordered_values[:-1]
    if not torch.isfinite(values).all() or values.numel() < 2 or (gaps <= 0).any():
        raise ValueError("CDF ordinal loss requires distinct finite class values")
    target = target.reshape(-1).long().to(prediction.device)
    probabilities = F.softmax(prediction.float(), dim=-1).index_select(-1, order)
    predicted_cdf = probabilities.cumsum(dim=-1)[..., :-1]
    target_cdf = (
        values[target].unsqueeze(-1) <= ordered_values[:-1].unsqueeze(0)
    ).to(predicted_cdf.dtype)
    return ((predicted_cdf - target_cdf).square() * gaps).sum(dim=-1)


def ordinal_adjacent_cross_entropy(prediction, target, class_values):
    """Exact-label binary CE against physical +/-1 count alternatives.

    This auxiliary loss reinforces the true class against each available
    immediate count neighbor. It does not count neighboring labels as correct
    and does not replace the full multiclass CE. Classes without an available
    physical +/-1 neighbor contribute zero, retaining a valid gradient graph.
    """
    values = class_values.to(device=prediction.device, dtype=torch.float32)
    if values.ndim != 1 or values.numel() != prediction.shape[-1]:
        raise ValueError("Adjacent ordinal loss requires one physical value per class")
    if not torch.isfinite(values).all() or values.unique().numel() != values.numel():
        raise ValueError("Adjacent ordinal loss requires distinct finite class values")
    target = target.reshape(-1).long().to(prediction.device)
    logits = prediction.float()
    true_logits = logits.gather(-1, target.unsqueeze(-1))
    neighbors = (
        values.unsqueeze(0) - values[target].unsqueeze(-1)
    ).abs().eq(1.0)
    pairwise_ce = F.softplus(logits - true_logits)
    weights = neighbors.to(pairwise_ce.dtype)
    return (pairwise_ce * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


class Prediction(nn.Module):
    """Descriptor read-out with a configurable hidden stack.

    Hidden widths control the capacity and cost of the shared descriptor
    read-out. The same read-out configuration is used by all module and
    frequency controls. Empty hidden_dims gives a linear probe.
    """

    def __init__(self, input_dim, hidden_dims, out_dim, dropout=0.1):
        super(Prediction, self).__init__()
        dropout = float(dropout)
        widths = [int(input_dim)] + [
            int(width) for width in hidden_dims if int(width) > 0
        ]
        layers = []
        for index, (fan_in, fan_out) in enumerate(zip(widths, widths[1:])):
            layers.append(nn.Linear(fan_in, fan_out))
            if index == 0:
                layers.append(nn.LayerNorm(fan_out))
            layers.append(nn.Dropout(p=dropout, inplace=False))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(widths[-1], out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.empty(num_numerical_types, dim))
        init.xavier_uniform_(self.weights)
        init.constant_(self.biases, 0.)

    def forward(self, x):
        x = x.unsqueeze(-1)  
        return x * self.weights + self.biases


class SpectralPointEmbedder(nn.Module):
    """Embed each (energy, intensity) pair as one token, position-blind.

    The projection is shared across energy bins. Non-affine intensity
    normalization avoids adding learned per-bin calibration parameters.
    Ordered regional read-out can retain coarse energy-location information.
    """

    def __init__(self, dim, num_points, normalize_intensity=True):
        super().__init__()
        self.intensity_norm = (
            nn.LayerNorm(num_points, elementwise_affine=False)
            if bool(normalize_intensity)
            else None
        )
        self.projection = nn.Linear(2, dim)
        init.xavier_uniform_(self.projection.weight)
        init.zeros_(self.projection.bias)

    def forward(self, interleaved_spectrum):
        if interleaved_spectrum.shape[1] % 2:
            raise ValueError("interleaved spectrum must contain energy/intensity pairs")
        points = interleaved_spectrum.reshape(
            interleaved_spectrum.shape[0], -1, 2
        )
        if self.intensity_norm is not None:
            energy = points[..., 0]
            intensity = self.intensity_norm(points[..., 1])
            points = torch.stack([energy, intensity], dim=-1)
        return self.projection(points)


class PointwiseEmbedding(nn.Module):
    """Channel projection with no neighbourhood mixing and no position code.

    The shared 1x1 projection mixes channels independently at each point;
    neighbourhood interactions are handled by the subsequent blocks.
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(p=float(dropout))
        init.xavier_uniform_(self.projection.weight)
        init.zeros_(self.projection.bias)

    def forward(self, x, x_mark=None):
        return self.dropout(self.projection(x))


class SPT(nn.Module):

    def __init__(self, args, device):
        super(SPT, self).__init__()
        self.args = args 
        dim = args.dim
        num_special_tokens = args.num_special_tokens
        categories = args.categories
        out = args.out
        self.out = out
        self.num_cont = args.num_cont
        num_cont = args.num_cont

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        total_tokens = self.num_unique_categories + num_special_tokens

        if self.num_unique_categories > 0:
            categories_offset = torch.nn.functional.pad(torch.tensor(categories), (1, 0), value=num_special_tokens)
            self.register_buffer('categories_offset', categories_offset.cumsum(-1)[:-1])
            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        self.spectral_pair_tokens = bool(
            getattr(args, 'spectral_pair_tokens', False)
        )
        self.input_preprocessing = str(
            getattr(args, 'input_preprocessing', 'none')
        ).lower()
        self.lowpass_bins = int(getattr(args, 'lowpass_bins', 16))
        if num_cont > 0:
            if self.spectral_pair_tokens:
                if num_cont % 2:
                    raise ValueError("num_cont must be even for spectral_pair_tokens")
                self.num_spectral_points = num_cont // 2
                self.numerical_embedder = SpectralPointEmbedder(
                    dim,
                    self.num_spectral_points,
                    normalize_intensity=bool(
                        getattr(args, 'intensity_norm', True)
                    ),
                )
            else:
                self.numerical_embedder = NumericalEmbedder(dim, num_cont)

        e_layers = args.e_layers
        self.layer = e_layers
        dropout = args.ff_dropout
        self.seq = (
            self.num_spectral_points
            if self.spectral_pair_tokens
            else self.num_cont + self.num_categories
        )
        # Which branches the blocks apply. The four ablation cells are exactly
        # the four values of this one field.
        self.force_expert = str(
            getattr(args, 'force_expert', 'fft_graph') or 'fft_graph'
        ).lower()
        self.label_smoothing = float(getattr(args, 'label_smoothing', 0.0))
        self.cls_loss = str(getattr(args, 'cls_loss', 'ce')).lower()
        self.focal_gamma = float(getattr(args, 'focal_gamma', 1.0))
        self.ordinal_penalty_weight = float(
            getattr(args, 'ordinal_penalty_weight', 0.0) or 0.0
        )
        self.ordinal_penalty_power = float(
            getattr(args, 'ordinal_penalty_power', 1.0) or 1.0
        )
        self.ordinal_penalty_kind = str(
            getattr(args, 'ordinal_penalty_kind', 'expected_distance')
        ).lower()
        if self.ordinal_penalty_kind not in {'expected_distance', 'cdf_squared', 'adjacent_ce'}:
            raise ValueError(
                "ordinal_penalty_kind must be expected_distance, cdf_squared or adjacent_ce"
            )
        if (
            not math.isfinite(self.ordinal_penalty_weight)
            or self.ordinal_penalty_weight < 0
            or not math.isfinite(self.ordinal_penalty_power)
            or self.ordinal_penalty_power <= 0
        ):
            raise ValueError(
                "ordinal penalty weight must be nonnegative and power positive"
            )
        ordinal_values = None
        if self.ordinal_penalty_weight > 0:
            task_name = getattr(args, 'task', None)
            task_spec = (
                getattr(args, task_name)
                if isinstance(task_name, str) and hasattr(args, task_name)
                else None
            )
            if not isinstance(task_spec, dict) or task_spec.get('mode') != 'Cls':
                raise ValueError(
                    "ordinal penalty is available only for single-task classification"
                )
            values = task_spec.get('value', [])
            labels = task_spec.get('label', [])
            if (
                len(values) != self.out or len(labels) != self.out
                or sorted(labels) != list(range(self.out))
            ):
                raise ValueError("ordinal penalty requires a complete class mapping")
            ordinal_values = torch.empty(self.out, dtype=torch.float32)
            for value, label in zip(values, labels):
                ordinal_values[int(label)] = float(value)
            if (
                not torch.isfinite(ordinal_values).all()
                or float(ordinal_values.max() - ordinal_values.min()) <= 0
                or (
                    self.ordinal_penalty_kind in {'cdf_squared', 'adjacent_ce'}
                    and ordinal_values.unique().numel() != self.out
                )
            ):
                raise ValueError(
                    "ordinal penalty requires distinct finite numeric class values"
                )
        # Nonpersistent keeps strict loading of all historical checkpoints.
        self.register_buffer(
            '_ordinal_class_values', ordinal_values, persistent=False
        )
        self.regression_output = str(
            getattr(args, 'regression_output', 'raw')
        ).lower()
        if self.regression_output not in {'raw', 'standardized'}:
            raise ValueError("regression_output must be raw or standardized")
        self.regression_target_mean = float(
            getattr(args, 'regression_target_mean', 0.0)
        )
        self.regression_target_std = max(
            float(getattr(args, 'regression_target_std', 1.0)), 1e-6
        )
        if not (
            math.isfinite(self.regression_target_mean)
            and math.isfinite(self.regression_target_std)
        ):
            raise ValueError("regression target statistics must be finite")

        # Statistics come from training labels, never validation/test labels.
        # A nonpersistent buffer preserves strict loading of legacy weights.
        class_weights = None
        if self.cls_loss == 'balanced_ce' and self.out > 1:
            counts = torch.as_tensor(
                getattr(args, 'class_counts', []), dtype=torch.float32
            )
            power = float(getattr(args, 'class_balance_power', 0.5))
            if (
                counts.ndim != 1 or counts.numel() != self.out
                or not torch.isfinite(counts).all() or (counts < 0).any()
                or not (counts > 0).any() or not math.isfinite(power)
                or power < 0
            ):
                raise ValueError(
                    "balanced_ce requires nonnegative training class_counts "
                    "of length out and a finite nonnegative class_balance_power"
                )
            observed = counts > 0
            observed_weights = counts[observed].pow(-power)
            class_weights = torch.ones_like(counts)
            # Unit expected weight over training examples keeps balancing
            # from silently shrinking the overall loss / effective step size.
            mean_weight = (
                observed_weights * counts[observed]
            ).sum() / counts[observed].sum()
            class_weights[observed] = observed_weights / mean_weight
            # A class absent from training retains neutral weight 1, so an
            # unexpected label is not silently ignored. Eval loss is unweighted.
        self.register_buffer(
            '_class_weights', class_weights, persistent=False
        )
        self.head_hidden = tuple(getattr(args, 'head_hidden', (64,)))
        lowpass_bins = getattr(args, 'lowpass_bins', None)
        spectral_decomposition = getattr(
            args, 'spectral_decomposition', 'complementary'
        )
        spectral_num_bands = getattr(args, 'spectral_num_bands', 5)
        stft_n_fft = getattr(args, 'stft_n_fft', 32)
        branch_gate_init = getattr(args, 'branch_gate_init', 0.05)
        spectral_gate_init = getattr(args, 'spectral_gate_init', None)
        graph_gate_init = getattr(args, 'graph_gate_init', None)
        residual_dropout = getattr(args, 'residual_dropout', 0.1)
        graph_dropout = getattr(args, 'graph_dropout', 0.1)
        graph_neighbors = getattr(args, 'graph_neighbors', 3)
        graph_heads = getattr(args, 'graph_heads', 4)
        graph_dilations = getattr(args, 'graph_dilations', None)
        graph_max_dilation = int(getattr(args, 'graph_max_dilation', 8))
        graph_global_heads = getattr(args, 'graph_global_heads', None)
        graph_global_k = int(getattr(args, 'graph_global_k', 8))
        drop_path = getattr(args, 'drop_path', 0.0)
        # How much of the spectral residual the graph branch reads as context
        # under parallel composition. None disables the path entirely.
        graph_spectral_context = getattr(args, 'graph_spectral_context', 0.1)
        disjoint_branch_channels = bool(
            getattr(args, 'disjoint_branch_channels', True)
        )
        graph_channel_fraction = float(
            getattr(args, 'graph_channel_fraction', 0.5)
        )
        allocation_logit_init = float(
            getattr(args, 'allocation_logit_init', 2.0)
        )
        lifting_kernel = int(getattr(args, 'lifting_kernel', 5))
        lifting_steps = int(getattr(args, 'lifting_steps', 2))
        lifting_shrinkage = bool(getattr(args, 'lifting_shrinkage', True))
        lifting_contrast_rms_floor = float(
            getattr(args, 'lifting_contrast_rms_floor', 0.0)
        )
        head_dropout = float(getattr(args, 'head_dropout', 0.1))
        keep_dc = bool(getattr(args, 'keep_dc', False))
        compose_mode = str(getattr(args, 'compose_mode', 'parallel')).lower()
        branch_fusion = str(getattr(args, 'branch_fusion', 'allocated')).lower()
        self.model = nn.ModuleList([SpecBlock(self.seq, self.seq // 2, self.seq // 2, top_k=args.top_k,
                                            d_model=dim, d_ff=dim, num_kernels=args.num_kernels, device=device,
                                            lowpass_bins=lowpass_bins,
                                            spectral_decomposition=spectral_decomposition,
                                            spectral_num_bands=spectral_num_bands,
                                            stft_n_fft=stft_n_fft,
                                            branch_gate_init=branch_gate_init,
                                            spectral_gate_init=spectral_gate_init,
                                            graph_gate_init=graph_gate_init,
                                            residual_dropout=residual_dropout,
                                            graph_dropout=graph_dropout,
                                            graph_neighbors=graph_neighbors,
                                            graph_heads=graph_heads,
                                            graph_dilations=graph_dilations,
                                            graph_max_dilation=graph_max_dilation,
                                            graph_global_heads=graph_global_heads,
                                            graph_global_k=graph_global_k,
                                            drop_path=drop_path,
                                            keep_dc=keep_dc,
                                            compose_mode=compose_mode,
                                            branch_fusion=branch_fusion,
                                            graph_spectral_context=graph_spectral_context,
                                            disjoint_branch_channels=disjoint_branch_channels,
                                            graph_channel_fraction=graph_channel_fraction,
                                            allocation_logit_init=allocation_logit_init,
                                            lifting_kernel=lifting_kernel,
                                            lifting_steps=lifting_steps,
                                            lifting_shrinkage=lifting_shrinkage,
                                            lifting_contrast_rms_floor=
                                            lifting_contrast_rms_floor) for _ in
                                    range(e_layers)])
        self.hermitian_irfft = bool(getattr(args, 'hermitian_irfft', False))
        for transform in self.model.modules():
            if isinstance(transform, AdaptiveFourierLiftingTransform):
                transform.hermitian_irfft = self.hermitian_irfft

        self.enc_embedding = PointwiseEmbedding(
            args.dim, dropout=float(getattr(args, 'ff_dropout', 0.1))
        )
        # A single LayerNorm over all channels lets the branch with the larger
        # update inflate the shared denominator and shrink the other branch's
        # relative magnitude — a normalization-mediated winner-take-all.
        # Normalizing the two nominal groups separately decouples the two
        # magnitudes. The groups stay fixed even though the branches' learned
        # per-channel shares drift away from this boundary during training.
        if disjoint_branch_channels:
            self.spectral_width, self.graph_width = channel_split_sizes(
                dim, graph_channel_fraction
            )
            self.layer_norm = None
            self.group_norms = nn.ModuleList([
                nn.LayerNorm(self.spectral_width),
                nn.LayerNorm(self.graph_width),
            ])
        else:
            self.spectral_width, self.graph_width = dim, dim
            self.layer_norm = nn.LayerNorm(dim)
            self.group_norms = None

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        self.is_multi_task = (
            hasattr(args, 'multi_task')
            and isinstance(args.multi_task, list)
            and len(args.multi_task) > 0
        )
        self.tl_logit_adapter_feature = str(
            getattr(args, 'tl_logit_adapter_feature', 'none')
        ).lower()
        if self.tl_logit_adapter_feature not in {'none', 'pooled'}:
            raise ValueError('tl_logit_adapter_feature must be none or pooled')
        self.tl_logit_adapter = None
        if self.tl_logit_adapter_feature != 'none':
            task_name = getattr(args, 'task', None)
            task_spec = getattr(args, task_name, None) if isinstance(task_name, str) else None
            if (self.is_multi_task or not isinstance(task_spec, dict)
                    or task_spec.get('mode') != 'Cls' or self.out <= 1):
                raise ValueError(
                    'TL logit adapter supports only single-task classification'
                )

        # Legacy defaults retain the exact head shape and state-dict keys.
        # Optional ordered regional means preserve coarse energy localization
        # without a learned pooling encoder. Apply the same options to controls.
        self.readout_bins = int(getattr(args, 'readout_bins', 0))
        if self.readout_bins < 0:
            raise ValueError("readout_bins must be nonnegative")
        self.head_use_category = bool(getattr(args, 'head_use_category', False))
        feature_dim = (2 + self.readout_bins + int(self.head_use_category)) * dim
        self.feature_dim = feature_dim

        if self.is_multi_task:
            if self.regression_output != 'standardized':
                raise ValueError(
                    "Joint SPT requires regression_output='standardized' so "
                    "regression heads share a scale-balanced training objective"
                )
            self.heads = nn.ModuleDict()
            self.multi_task_modes = {}
            self.multi_task_regression_stats = {}
            configured_stats = getattr(
                args, 'multi_task_regression_stats', {}
            ) or {}
            configured_weights = getattr(
                args, 'multi_task_loss_weights', {}
            ) or {}
            self.multi_task_loss_weights = {}
            if self.cls_loss == 'balanced_ce':
                raise ValueError(
                    "Joint SPT does not use one anchor task's class weights; "
                    "use cls_loss='ce' or provide a dedicated per-head policy"
                )
            for t_name in args.multi_task:
                if not hasattr(args, t_name):
                    raise ValueError(
                        f"Configuration for joint task {t_name!r} is missing"
                    )
                t_conf = getattr(args, t_name)
                mode = str(t_conf['mode'])
                if mode not in {'Cls', 'Reg'}:
                    raise ValueError(
                        f"Unsupported mode for joint task {t_name}: {mode}"
                    )
                t_out = 1 if mode == 'Reg' else int(t_conf['num'])
                self.multi_task_modes[t_name] = mode
                weight = float(configured_weights.get(t_name, 1.0))
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError(
                        f"Joint loss weight for {t_name} must be finite and positive"
                    )
                self.multi_task_loss_weights[t_name] = weight
                if mode == 'Reg':
                    if t_name not in configured_stats:
                        raise ValueError(
                            f"Joint regression task {t_name} requires train-only "
                            "mean/std statistics"
                        )
                    stats = configured_stats[t_name]
                    mean = float(stats['mean'])
                    std = float(stats['std'])
                    if (
                        not math.isfinite(mean)
                        or not math.isfinite(std)
                        or std <= 1e-6
                    ):
                        raise ValueError(
                            f"Invalid joint regression statistics for {t_name}"
                        )
                    self.multi_task_regression_stats[t_name] = {
                        'mean': mean,
                        'std': std,
                    }

                self.heads[t_name] = Prediction(
                    feature_dim, self.head_hidden, t_out, dropout=head_dropout
                )
                if mode == 'Reg':
                    init.zeros_(self.heads[t_name].layers[-1].bias)
        else:
            self.projection = Prediction(
                feature_dim, self.head_hidden, out, dropout=head_dropout
            )
            if self.out == 1 and self.regression_output == 'standardized':
                # Keep a nonzero final weight for immediate upstream gradients,
                # but do not introduce an arbitrary standardized output bias.
                init.zeros_(self.projection.layers[-1].bias)
            if self.tl_logit_adapter_feature == 'pooled':
                reference = self.projection.layers[-1].weight
                cuda_devices = ([reference.device.index]
                                if reference.device.type == 'cuda' else [])
                # Linear's default random initializer must not advance the
                # model's subsequent dropout/training RNG stream. The adapter
                # starts at exact zero and leaves source logits unchanged.
                with torch.random.fork_rng(devices=cuda_devices):
                    self.tl_logit_adapter = nn.Linear(
                        feature_dim, out,
                        device=reference.device, dtype=reference.dtype,
                    )
                init.zeros_(self.tl_logit_adapter.weight)
                init.zeros_(self.tl_logit_adapter.bias)

    def model_name(self):
        return 'SPT'

    def _preprocess_numerical(self, x_numer):
        if self.input_preprocessing in {'none', 'raw'}:
            return x_numer
        if self.input_preprocessing != 'lowpass':
            raise ValueError(
                "input_preprocessing must be none, raw, or lowpass"
            )
        if not self.spectral_pair_tokens:
            signal = x_numer
            spectrum = torch.fft.rfft(signal, dim=1)
            spectrum[:, self.lowpass_bins:] = 0
            return torch.fft.irfft(
                spectrum, n=signal.shape[1], dim=1
            )
        paired = x_numer.reshape(x_numer.shape[0], -1, 2)
        energy = paired[..., 0]
        intensity = paired[..., 1]
        spectrum = torch.fft.rfft(intensity, dim=1)
        spectrum[:, self.lowpass_bins:] = 0
        filtered = torch.fft.irfft(
            spectrum, n=intensity.shape[1], dim=1
        )
        return torch.stack([energy, filtered], dim=-1).reshape_as(x_numer)

    def _normalize_stream(self, hidden):
        if self.group_norms is None:
            return self.layer_norm(hidden)
        spectral_part, graph_part = hidden.split(
            [self.spectral_width, self.graph_width], dim=-1
        )
        return torch.cat(
            [
                self.group_norms[0](spectral_part),
                self.group_norms[1](graph_part),
            ],
            dim=-1,
        )

    def _run_expert_hidden(self, embedded, energy, expert):
        enc_out = embedded
        for i in range(self.layer):
            enc_out = self._normalize_stream(
                self.model[i](
                    enc_out, energy=energy, force_expert=expert
                )
            )
        return self.dropout(self.act(enc_out))

    def _pool_features(self, hidden, categorical_context=None):
        features = [hidden.mean(dim=1), hidden.amax(dim=1)]
        if self.readout_bins:
            regions = F.adaptive_avg_pool1d(
                hidden.transpose(1, 2), self.readout_bins
            ).transpose(1, 2)
            # Flatten region-major: bins remain in input energy order.
            features.append(regions.flatten(start_dim=1))
        if self.head_use_category:
            if categorical_context is None:
                categorical_context = hidden.new_zeros(
                    hidden.shape[0], hidden.shape[-1]
                )
            elif categorical_context.ndim == 3:
                categorical_context = categorical_context.mean(dim=1)
            features.append(categorical_context)
        return torch.cat(features, dim=-1)

    def _run_expert(self, embedded, energy, expert, categorical_context=None):
        output = self._run_expert_hidden(embedded, energy, expert)
        return self._pool_features(output, categorical_context)

    def get_shared_features(
        self, x_enc, energy=None, force_expert=None, categorical_context=None
    ):
        embedded = self.enc_embedding(x_enc, None)
        return self._run_expert(
            embedded, energy, force_expert or self.force_expert,
            categorical_context=categorical_context,
        )

    def _decode_prediction(self, prediction):
        # Every external inference caller receives physical/original units.
        if (
            not self.is_multi_task and self.out == 1
            and self.regression_output == 'standardized'
        ):
            return (
                prediction.float() * self.regression_target_std
                + self.regression_target_mean
            )
        return prediction

    def _decode_multi_task_prediction(self, task_name, prediction):
        if self.multi_task_modes[task_name] != 'Reg':
            return prediction
        stats = self.multi_task_regression_stats[task_name]
        return prediction.float() * stats['std'] + stats['mean']

    def _multi_task_loss(self, task_name, raw_prediction, label):
        mode = self.multi_task_modes[task_name]
        if mode == 'Reg':
            target = label.float().reshape(-1)
            prediction = raw_prediction.float().squeeze(-1)
            valid = torch.isfinite(target)
            if not valid.any():
                return None
            stats = self.multi_task_regression_stats[task_name]
            standardized_target = (target[valid] - stats['mean']) / stats['std']
            return (prediction[valid] - standardized_target).square().mean()

        target = label.reshape(-1).long()
        valid = (target >= 0) & (target < raw_prediction.shape[1])
        if not valid.any():
            return None
        prediction = raw_prediction[valid]
        target = target[valid]
        if self.cls_loss == 'focal':
            log_prob = F.log_softmax(prediction, dim=-1)
            per_sample = F.nll_loss(log_prob, target, reduction='none')
            focal_weight = (1.0 - log_prob.exp().gather(
                1, target.unsqueeze(1)
            ).squeeze(1)).clamp_min(0.0) ** self.focal_gamma
            return (focal_weight * per_sample).mean()
        return F.cross_entropy(
            prediction,
            target,
            label_smoothing=self.label_smoothing,
        )

    def _single_task_loss(self, prediction, label):
        return self._per_sample_loss(prediction, label).mean()

    def _per_sample_loss(self, prediction, label):
        if self.out == 1:
            # Keep small descriptor residuals in FP32 under AMP: casting
            # labels/decoded means to FP16 can erase sub-milliscale errors.
            target = label.float().reshape(-1)
            values = prediction.float().squeeze(-1)
            if target.numel() != values.numel():
                raise ValueError("regression requires one target per prediction")
            # Explicit flattening prevents [B, 1] labels broadcasting to [B, B].
            return ((values - target) / self.regression_target_std).square()

        target = label.reshape(-1).long()
        if self.cls_loss == 'focal':
            log_prob = F.log_softmax(prediction, dim=-1)
            per_sample = F.nll_loss(log_prob, target, reduction='none')
            focal_weight = (1.0 - log_prob.exp().gather(
                1, target.unsqueeze(1)
            ).squeeze(1)).clamp_min(0.0) ** self.focal_gamma
            per_sample = focal_weight * per_sample
        else:
            weights = (
                self._class_weights
                if self.training and self.cls_loss == 'balanced_ce' else None
            )
            per_sample = F.cross_entropy(
                prediction, target, weight=weights, reduction='none',
                label_smoothing=self.label_smoothing,
            )
        if self.ordinal_penalty_weight > 0:
            values = self._ordinal_class_values.to(prediction.device)
            if self.ordinal_penalty_kind == 'cdf_squared':
                expected_distance = ordinal_cdf_squared_distance(
                    prediction, target, values
                )
            elif self.ordinal_penalty_kind == 'adjacent_ce':
                expected_distance = ordinal_adjacent_cross_entropy(
                    prediction, target, values
                )
            else:
                # Historical expected-distance loss is intentionally unchanged.
                span = (values.max() - values.min()).clamp_min(1e-6)
                distances = (
                    (values.unsqueeze(0) - values[target].unsqueeze(1)).abs()
                    / span
                ).pow(self.ordinal_penalty_power)
                expected_distance = (
                    F.softmax(prediction.float(), dim=-1) * distances
                ).sum(dim=-1)
            per_sample = (
                per_sample
                + self.ordinal_penalty_weight * expected_distance
            )
        return per_sample

    def get_channel_allocation(self):
        """Per-block spectral share ``m`` of the learned channel partition.

        ``m`` near 1 means the channel carries the spectral branch, near 0 the
        graph branch, near 0.5 that the two share it.
        """
        allocation = []
        for block in self.model:
            if getattr(block, 'allocation_logit', None) is None:
                allocation.append(None)
            else:
                allocation.append(
                    block.spectral_allocation().detach().cpu()
                )
        return allocation

    @staticmethod
    def _snapshot(module, attribute):
        """Detached copy of a module's recorded tensor(s), or None.

        Every slot is optional: a module that has not run yet has recorded
        nothing, and the graph branch records one tensor per neighbourhood kind,
        which arrives as a tuple.
        """
        value = getattr(module, attribute, None) if module is not None else None
        if value is None:
            return None
        if isinstance(value, tuple):
            return tuple(item.detach().cpu() for item in value)
        return value.detach().cpu()

    def get_diagnostics(self):
        diagnostics = []
        for idx, block in enumerate(self.model):
            # Whichever operator occupies the spectral slot: the proposed
            # learned wavelet, or the naive filter that replaced it.
            operator = getattr(
                block.spectral, block.spectral.decomposition, None
            )
            diagnostics.append({
                'layer': idx,
                'residual_gate': self._snapshot(block, 'last_residual_gate'),
                'band_gain': self._snapshot(operator, 'last_band_gain'),
                'graph_attention': self._snapshot(
                    block.graph, 'last_attention'
                ),
                'graph_energy_distance': self._snapshot(
                    block.graph, 'last_energy_distance'
                ),
            })
        return {
            'force_expert': self.force_expert,
            'blocks': diagnostics,
        }

    def forward(self, x_categ, x_numer, label, step=None, return_attn=False, delta=False):
        xs = []
        raw_x_numer = x_numer
        x_numer = self._preprocess_numerical(x_numer)
        categorical_context = None
        head_category = None
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset
            x_categ_embed = self.categorical_embeds(x_categ)
            head_category = x_categ_embed.mean(dim=1)
            if self.spectral_pair_tokens:
                categorical_context = x_categ_embed.mean(dim=1, keepdim=True)
            else:
                xs.append(x_categ_embed)
        
        if self.num_cont > 0:
            x_numer_embed = self.numerical_embedder(x_numer)
            if categorical_context is not None:
                x_numer_embed = x_numer_embed + categorical_context
            xs.append(x_numer_embed)

        x_enc = torch.cat(xs, dim=1)
        energy = (
            raw_x_numer[:, 0::2] if self.spectral_pair_tokens else None
        )
        embedded = self.enc_embedding(x_enc, None)
        enc_feature = self._run_expert(
            embedded, energy, self.force_expert,
            categorical_context=head_category,
        )

        if self.is_multi_task:
            outs = {}
            task_losses = []
            task_weights = []
            for t_name, head_module in self.heads.items():
                raw_prediction = head_module(enc_feature)
                outs[t_name] = self._decode_multi_task_prediction(
                    t_name, raw_prediction
                )
                if label is not None and isinstance(label, dict) and t_name in label:
                    task_loss = self._multi_task_loss(
                        t_name, raw_prediction, label[t_name]
                    )
                    if task_loss is not None:
                        weight = self.multi_task_loss_weights[t_name]
                        task_losses.append(task_loss * weight)
                        task_weights.append(weight)
            if label is None:
                total_loss = None
            elif task_losses:
                total_loss = torch.stack(task_losses).sum() / sum(task_weights)
            else:
                raise ValueError(
                    "Joint SPT batch contains no valid target for any task"
                )

            if return_attn:
                return outs, total_loss, self.get_diagnostics()
            return outs, total_loss

        raw_prediction = self.projection(enc_feature)
        if self.tl_logit_adapter is not None:
            raw_prediction = raw_prediction + self.tl_logit_adapter(enc_feature)
        dec_out = self._decode_prediction(raw_prediction)
        loss = (
            self._single_task_loss(dec_out, label)
            if label is not None else None
        )
        if return_attn:
            return dec_out, loss, self.get_diagnostics()
        return dec_out, loss
