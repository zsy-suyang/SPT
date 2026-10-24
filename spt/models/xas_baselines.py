"""XAS-specific spectrum-to-descriptor neural baselines.

The numerical input interleaves energy and intensity. Historical checkpoints
use intensity only. The opt-in same_information variant additionally gives
the supervised readout the energy coordinates and one-hot categorical metadata.
It is an input-matched adaptation, not an exact published-method reproduction.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def _positive_int(value, name):
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class _SpectrumBaseline(nn.Module):
    """Common trainer-compatible behaviour for single-task baselines."""

    architecture_name = "xas_baseline"

    def __init__(self, args):
        super().__init__()
        self.out = _positive_int(args.out, "out")
        self.num_cont = _positive_int(args.num_cont, "num_cont")
        self.interleaved_input = bool(
            getattr(args, "xas_interleaved_input", True)
        )
        if self.interleaved_input and self.num_cont % 2:
            raise ValueError(
                "num_cont must be even when xas_interleaved_input is enabled"
            )
        self.spectrum_length = (
            self.num_cont // 2 if self.interleaved_input else self.num_cont
        )

        self.input_mode = str(getattr(args, "xas_input_mode", "intensity"))
        if self.input_mode not in {"intensity", "same_information"}:
            raise ValueError("xas_input_mode must be intensity or same_information")
        if self.input_mode == "same_information" and not self.interleaved_input:
            raise ValueError("same_information requires interleaved energy/intensity")
        self.category_sizes = tuple(int(size) for size in getattr(args, "categories", ()))
        if any(size <= 0 for size in self.category_sizes):
            raise ValueError("category sizes must be positive")
        self.context_dim = (
            self.spectrum_length + sum(self.category_sizes)
            if self.input_mode == "same_information" else 0
        )
        self.regression_output = str(getattr(args, "xas_regression_output", "raw"))
        if self.regression_output not in {"raw", "standardized"}:
            raise ValueError("xas_regression_output must be raw or standardized")
        self.target_mean = float(getattr(args, "regression_target_mean", 0.0))
        self.target_std = float(getattr(args, "regression_target_std", 1.0))
        if not math.isfinite(self.target_mean) or not math.isfinite(self.target_std) or self.target_std <= 0:
            raise ValueError("regression target statistics must be finite with positive std")
        # Plain attributes preserve the exact state_dict layout of old models.
        # The opt-in scale and training-only statistics are stored in run YAML.

    def _spectrum(self, x_numer):
        if x_numer.ndim != 2 or x_numer.shape[1] != self.num_cont:
            raise ValueError(
                "x_numer must have shape "
                f"(batch, {self.num_cont}), got {tuple(x_numer.shape)}"
            )
        if self.interleaved_input:
            return x_numer[:, 1::2].float()
        return x_numer.float()

    def _with_context(self, features, x_categ, x_numer):
        if self.input_mode == "intensity":
            return features
        context = [x_numer[:, 0::2].float()]
        if self.category_sizes:
            if x_categ is None or x_categ.shape != (features.shape[0], len(self.category_sizes)):
                raise ValueError("categorical input does not match configured categories")
            context.extend(
                F.one_hot(x_categ[:, column].long(), num_classes=size).float()
                for column, size in enumerate(self.category_sizes)
            )
        return torch.cat([features] + context, dim=-1)

    def _prediction_units(self, prediction):
        if self.out == 1 and self.regression_output == "standardized":
            return prediction.float() * self.target_std + self.target_mean
        return prediction

    def _task_loss(self, prediction, label):
        if label is None:
            return prediction.new_zeros(())
        target = label.reshape(-1)
        if prediction.shape[0] != target.numel():
            raise ValueError("requires exactly one target per prediction")
        if self.out == 1:
            values = prediction.float().reshape(-1)
            scale = self.target_std if self.regression_output == "standardized" else 1.0
            return ((values - target.float()) / scale).square().mean()
        return F.cross_entropy(prediction, target.long())

    def num_parameters(self, trainable_only=True):
        """Return the model parameter count for benchmark manifests/logging."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad or not trainable_only
        )

    def model_name(self):
        return self.architecture_name

    def _finish_initialization(self):
        print(
            f"{self.architecture_name} parameters: "
            f"{self.num_parameters():,}"
        )


class DirectSpectrumMLP(_SpectrumBaseline):
    """Direct spectrum-to-descriptor multilayer perceptron."""

    architecture_name = "DirectNN"

    def __init__(self, args, device=None):
        super().__init__(args)
        hidden = _positive_int(
            getattr(args, "xas_mlp_hidden_dim", 256),
            "xas_mlp_hidden_dim",
        )
        bottleneck = _positive_int(
            getattr(args, "xas_mlp_bottleneck_dim", 128),
            "xas_mlp_bottleneck_dim",
        )
        dropout = float(getattr(args, "xas_dropout", 0.15))
        self.network = nn.Sequential(
            nn.LayerNorm(self.spectrum_length + self.context_dim),
            nn.Linear(self.spectrum_length + self.context_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, self.out),
        )
        self._finish_initialization()

    def forward(
        self,
        x_categ,
        x_numer,
        label,
        step=None,
        return_attn=False,
        delta=False,
        **kwargs,
    ):
        features = self._with_context(self._spectrum(x_numer), x_categ, x_numer)
        prediction = self._prediction_units(self.network(features))
        return prediction, self._task_loss(prediction, label)


class _ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super().__init__()
        padding = kernel_size // 2
        self.convolutions = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, 1, bias=False)
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.activation(self.convolutions(x) + self.skip(x)))


class SpectrumCNN1D(_SpectrumBaseline):
    """Local multiscale 1D CNN for an XAS intensity trace."""

    architecture_name = "CNN1D"

    def __init__(self, args, device=None):
        super().__init__(args)
        width = _positive_int(
            getattr(args, "xas_cnn_width", 32), "xas_cnn_width"
        )
        dropout = float(getattr(args, "xas_dropout", 0.15))
        self.features = nn.Sequential(
            _ResidualConvBlock(1, width, 9, dropout),
            nn.MaxPool1d(2),
            _ResidualConvBlock(width, width * 2, 7, dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(width * 2 + self.context_dim),
            nn.Linear(width * 2 + self.context_dim, self.out),
        )
        self._finish_initialization()

    def forward(
        self,
        x_categ,
        x_numer,
        label,
        step=None,
        return_attn=False,
        delta=False,
        **kwargs,
    ):
        spectrum = self._spectrum(x_numer).unsqueeze(1)
        features = self.features(spectrum).flatten(1)
        features = self._with_context(features, x_categ, x_numer)
        prediction = self._prediction_units(self.head(features))
        return prediction, self._task_loss(prediction, label)


class _GradientReversal(Function):
    """Gradient reversal, following the official RankAAE discriminator."""

    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.beta * grad_output, None


class RankAAELike(_SpectrumBaseline):
    """RankAAE-style adversarial autoencoder baseline.

    This follows the method of the official RankAAE implementation
    (``RankAAE/sc/clustering/model.py``): a convolutional encoder maps a
    spectrum to a low-dimensional style latent, a convolutional decoder
    reconstructs the spectrum, and a discriminator trained through a gradient
    reversal layer pushes the latent toward a Gaussian prior (adversarial
    autoencoder). A rank/style constraint aligns the latent with the target
    descriptor, and a supervised readout head produces the descriptor
    prediction used for the benchmark. This is a simplified supervised
    adaptation: encoder/decoder blocks and losses differ from the bundled
    upstream implementation. It must not be reported as an exact reproduction.
    The optional input-matched variant adds metadata only to the readout.
    """

    architecture_name = "RankAAE"

    def __init__(self, args, device=None):
        super().__init__(args)
        hidden = _positive_int(
            getattr(args, "xas_rankaae_hidden_dim", 256),
            "xas_rankaae_hidden_dim",
        )
        latent = _positive_int(
            getattr(args, "xas_rankaae_latent_dim", 64),
            "xas_rankaae_latent_dim",
        )
        channels = _positive_int(
            getattr(args, "xas_rankaae_channels", 8),
            "xas_rankaae_channels",
        )
        dropout = float(getattr(args, "xas_dropout", 0.2))
        self.reconstruction_weight = float(
            getattr(args, "xas_reconstruction_weight", 0.1)
        )
        self.rank_weight = float(getattr(args, "xas_rank_weight", 0.05))
        self.rank_margin = float(getattr(args, "xas_rank_margin", 1.0))
        self.adversarial_weight = float(
            getattr(args, "xas_adversarial_weight", 0.05)
        )
        self.adversarial_beta = float(
            getattr(args, "xas_adversarial_beta", 0.3)
        )
        self.latent_dim = latent
        self.decoder_channels = channels
        self.decoder_seed_len = 16

        self.input_norm = nn.LayerNorm(self.spectrum_length)
        # Convolutional style encoder (spectrum -> Gaussian style latent).
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, channels, 11, stride=2, padding=5,
                      padding_mode="replicate"),
            nn.BatchNorm1d(channels),
            nn.PReLU(channels),
            nn.Conv1d(channels, channels, 7, stride=2, padding=3,
                      padding_mode="replicate"),
            nn.BatchNorm1d(channels),
            nn.PReLU(channels),
            nn.Conv1d(channels, channels, 5, stride=2, padding=2,
                      padding_mode="replicate"),
            nn.BatchNorm1d(channels),
            nn.PReLU(channels),
            nn.AdaptiveAvgPool1d(4),
        )
        self.encoder_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, latent),
            nn.BatchNorm1d(latent, affine=False),
        )
        # Convolutional decoder (style latent -> reconstructed spectrum).
        self.decoder_seed = nn.Linear(
            latent, channels * self.decoder_seed_len
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1),
            nn.BatchNorm1d(channels),
            nn.PReLU(channels),
            nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1),
            nn.BatchNorm1d(channels),
            nn.PReLU(channels),
            nn.Conv1d(channels, 1, 5, padding=2, padding_mode="replicate"),
        )
        # Discriminator for the Gaussian prior (trained via gradient reversal).
        self.discriminator = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.PReLU(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.PReLU(hidden),
            nn.Linear(hidden, 2),
        )
        self.predictor = nn.Sequential(
            nn.PReLU(latent + self.context_dim),
            nn.Dropout(dropout),
            nn.Linear(latent + self.context_dim, self.out),
        )
        self.rank_head = nn.Linear(latent, 1)
        self._finish_initialization()

    def _encode(self, normalized):
        features = self.encoder_conv(normalized.unsqueeze(1))
        return self.encoder_head(features)

    def _decode(self, latent):
        seed = self.decoder_seed(latent).view(
            latent.shape[0], self.decoder_channels, self.decoder_seed_len
        )
        decoded = self.decoder_conv(seed).squeeze(1)
        return F.interpolate(
            decoded.unsqueeze(1),
            size=self.spectrum_length,
            mode="linear",
            align_corners=False,
        ).squeeze(1)

    def _adversarial_loss(self, latent):
        if latent.shape[0] < 2:
            return latent.new_zeros(())
        gaussian = torch.randn_like(latent)
        reversed_latent = _GradientReversal.apply(
            latent, self.adversarial_beta
        )
        real_logits = self.discriminator(gaussian)
        fake_logits = self.discriminator(reversed_latent)
        real_target = latent.new_ones(
            latent.shape[0], dtype=torch.long
        )
        fake_target = latent.new_zeros(
            latent.shape[0], dtype=torch.long
        )
        return 0.5 * (
            F.cross_entropy(real_logits, real_target)
            + F.cross_entropy(fake_logits, fake_target)
        )

    def _rank_loss(self, latent, label):
        if label is None or latent.shape[0] < 2:
            return latent.new_zeros(())
        label = label.reshape(-1)
        if label.numel() != latent.shape[0]:
            raise ValueError("rank loss requires one target per latent")
        left = latent[0::2]
        right = latent[1::2]
        pair_count = min(left.shape[0], right.shape[0])
        if pair_count == 0:
            return latent.new_zeros(())
        left = left[:pair_count]
        right = right[:pair_count]
        left_label = label[0::2][:pair_count]
        right_label = label[1::2][:pair_count]

        if self.out == 1:
            direction = torch.sign(
                left_label.float() - right_label.float()
            )
            informative = direction != 0
            if not informative.any():
                return latent.new_zeros(())
            left_score = self.rank_head(left).squeeze(-1)
            right_score = self.rank_head(right).squeeze(-1)
            signed_gap = direction[informative] * (
                left_score[informative] - right_score[informative]
            )
            return F.softplus(-signed_gap).mean()

        distance = (left - right).pow(2).sum(dim=1).sqrt()
        same_class = left_label.long() == right_label.long()
        same_loss = (
            distance[same_class].pow(2).mean()
            if same_class.any()
            else distance.new_zeros(())
        )
        different = ~same_class
        different_loss = (
            F.relu(self.rank_margin - distance[different]).pow(2).mean()
            if different.any()
            else distance.new_zeros(())
        )
        return same_loss + different_loss

    def forward(
        self,
        x_categ,
        x_numer,
        label,
        step=None,
        return_attn=False,
        delta=False,
        **kwargs,
    ):
        spectrum = self._spectrum(x_numer)
        normalized = self.input_norm(spectrum)
        latent = self._encode(normalized)
        features = self._with_context(latent, x_categ, x_numer)
        prediction = self._prediction_units(self.predictor(features))
        reconstruction = self._decode(latent)
        loss = self._task_loss(prediction, label)
        loss = loss + self.reconstruction_weight * F.mse_loss(
            reconstruction, normalized
        )
        loss = loss + self.rank_weight * self._rank_loss(latent, label)
        if self.training:
            loss = loss + self.adversarial_weight * self._adversarial_loss(
                latent
            )
        return prediction, loss
