"""Analiza XAI: Integrated Gradients (Captum) i agregacja atrybucji do map (kanały x czas)."""
import numpy as np
import torch
from captum.attr import IntegratedGradients

# (...) importy dopasujemy po wklejeniu ciał funkcji

KEY_MOTOR_CHANNELS = [
    "FC3", "FC1", "FCZ", "FC2", "FC4", "C5", "C3", "C1",
    "CZ", "C2", "C4", "C6", "CP3", "CP1", "CPZ", "CP2", "CP4",
]  # fmt: skip


def compute_captum_analysis(
    model,
    test_loader,
    device,
    max_samples=640,
    sfreq=160.0,
    sequence_first=True,
    feature_method="raw",
    tsp_step_size=0.1,
):
    """
    Computes model attributions for given samples upt to the max_samples count using IntegratedGradients.
    Returns a dictionary containing results with appropriate tags for later analysis.
    """
    ...  # (...)


def extract_global_heatmap_data(analysis_results, mode="all", subject_id=None, target_class=None):
    """
    Extracts and averages the (channels x time) attribution matrix across all samples.
    If 'subject_id' is provided, it filters the samples and computes the heatmap
    only for the specified patient.

    Modes:
    - 'all': Absolute sum of all attributions.
    - 'correct_direction': Only features that pushed the model towards the true class (attr > 0).
    - 'incorrect_direction': Only features that pushed the model towards the opposing class (attr < 0).
    """
    ...  # (...)


def get_electrode_total_importance(feature_importances, all_tsp_labels, channel_names):
    """
    Skanuje wszystkie wygenerowane pary TSP i sumuje siłę atencji dla każdej
    fizycznej elektrody, która wzięła w nich udział.
    """
    ...  # (...)
