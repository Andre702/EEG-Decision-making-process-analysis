"""Reprezentacja Riemannowska: okienkowa Tangent Space Projection (TSP)."""
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

# (...) importy dopasujemy po wklejeniu ciał metod


class RiemannianTokenizer:
    def __init__(self, window_size_sec=0.25, step_size_sec=0.1, sfreq=None):
        ...  # (...)

    def _extract_sliding_windows(self, X_epochs):
        ...  # (...)

    def fit_transform_all(self, training_subject_data, global_test_dataset):
        ...  # (...)

    def get_feature_dim(self):
        ...  # (...)


def generate_tsp_labels(channel_names):
    """
    Generuje poprawne mapowanie nazw dla wektora TSP z macierzy kowariancji.
    PyRiemann wykorzystuje układ górnego trójkąta macierzy (Upper Triangular).
    Zwraca listę stringów w stylu: "C3" (wariancja) lub "C3-C4" (kowariancja).
    """
    ...  # (...)
