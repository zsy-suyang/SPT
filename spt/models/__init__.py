"""Model construction with optional comparison models imported on demand."""


def build_model(args, device):
    architecture = str(getattr(args, "architecture", "spt")).lower()
    if architecture == "spt":
        from spt.models.SPT import SPT
        return SPT(args, device)
    if architecture == "amformer":
        from spt.baselines.amformer.model import FTTransformer as amformer
        return amformer(args)
    if architecture in {"fttransformer", "ft_transformer"}:
        from spt.models.ft_transformer import FTTransformer
        return FTTransformer(args, device)
    if architecture == "direct_nn":
        from spt.models.xas_baselines import DirectSpectrumMLP
        return DirectSpectrumMLP(args, device)
    if architecture == "cnn1d":
        from spt.models.xas_baselines import SpectrumCNN1D
        return SpectrumCNN1D(args, device)
    if architecture == "rankaae":
        from spt.models.xas_baselines import RankAAELike
        return RankAAELike(args, device)
    raise ValueError(f"unknown architecture: {architecture}")
