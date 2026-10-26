"""Preprocessing surowych nagrań EEG (Physionet / GigaScience) do plików .fif z epokami."""
import os
from pathlib import Path

import mne
import numpy as np

from .config import DatasetConfig, ExperimentConfig

# Parametry przetwarzania biorą się z cfg.preprocess (l_freq, h_freq, resample_to,
# selected_channels, segment_type, tmin, tmax). Podłączymy je przy wklejeniu ciał funkcji.


# --- Helpers
def _create_safe_directory(path: str) -> str:
    """Creates a directory safely without deleting existing content."""
    os.makedirs(path, exist_ok=True)
    return path


def _apply_bandpass_filter(raw: mne.io.BaseRaw, l_freq: float, h_freq: float):
    """Applies a bandpass filter to continuous raw data if frequencies are specified."""
    if l_freq is not None or h_freq is not None:
        print(f"Applying bandpass filter: {l_freq} - {h_freq} Hz")
        raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)
    return raw


def _select_channels(epochs: mne.Epochs, selected_channels: list):
    """Filters channels safely, standardizing names to uppercase."""
    if not selected_channels:
        return epochs

    clean_selected = [str(ch).upper().replace(".", "").strip() for ch in selected_channels]
    channels_to_keep = [
        orig_name
        for orig_name in epochs.ch_names
        if str(orig_name).upper().replace(".", "").strip() in clean_selected
    ]

    if len(channels_to_keep) == 0:
        raise ValueError("[ERROR]: None of the selected channels were found in the MNE data!")

    epochs.pick(channels_to_keep, verbose=False)
    return epochs


def _extract_and_save_epochs(
    raw: mne.io.BaseRaw,
    mapped_events: np.ndarray,
    event_id: dict,
    subject_name: str,
    save_dir: str,
):
    """Core function to extract epochs, filter channels, and save to disk."""
    ...  # (...)


# --- Per-dataset pipelines
def process_physionet(cfg: ExperimentConfig, dataset: DatasetConfig):
    print(f"--- Starting Physionet Pipeline from: {dataset.data_path} ---")
    save_directory = _create_safe_directory(str(cfg.preprocessed_dir(dataset)))
    ...  # (...)


def process_gigascience(cfg: ExperimentConfig, dataset: DatasetConfig):
    print(f"--- Starting GigaScience Pipeline from: {dataset.data_path} ---")
    data_dir_path = Path(dataset.data_path)
    save_directory = _create_safe_directory(str(cfg.preprocessed_dir(dataset)))
    ...  # (...)


# --- General Methods
def run_preprocessing(cfg: ExperimentConfig, dataset: DatasetConfig):
    name = dataset.name.upper()
    if name == "PHYSIONET":
        process_physionet(cfg, dataset)
    elif name == "GIGASCIENCE":
        process_gigascience(cfg, dataset)
    else:
        raise ValueError(f"Unknown dataset: {dataset.name}. Please select PHYSIONET or GIGASCIENCE.")
