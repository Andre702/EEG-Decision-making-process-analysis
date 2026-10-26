"""Architektura: ekstraktor cech, Temporal Transformer i warstwy pomocnicze."""
import torch
import torch.nn as nn

# (...) importy dopasujemy po wklejeniu ciał klas


class GaussianNoise(nn.Module):
    """Layer adding Gaussian noise only during training."""

    def __init__(self, std=0.01):
        ...  # (...)

    def forward(self, x):
        ...  # (...)


class FeatureExtractor(nn.Module):
    def __init__(self, model_class_name="TemporalTransformer", method="raw", noise_std=0.01):
        ...  # (...)

    def forward(self, x):
        ...  # (...)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=3000):
        ...  # (...)

    def forward(self, x):
        ...  # (...)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        ...  # (...)

    def forward(self, x):
        ...  # (...)


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        input_size,
        d_model=128,
        nhead=8,
        num_classes=2,
        feature_method="raw",
        feature_extractor=None,
        noise_std=0.01,
    ):
        ...  # (...)

    def forward(self, x):
        ...  # (...)
