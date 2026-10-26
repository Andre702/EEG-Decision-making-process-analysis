"""Datasety PyTorch, wczytywanie preprocesowanych epok i podział na zbiór treningowy/globalny test."""
import gc
import os

import mne
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

FEATURE_METHODS = ("raw", "tsp")


def _check_feature_method(feature_method: str) -> None:
    if feature_method not in FEATURE_METHODS:
        raise ValueError(
            f"Input method {feature_method}. Is not supported. Only allowed methods are 'tsp' and 'raw'."
        )


def _prepare_input(x_val: np.ndarray, feature_method: str, sequence_first: bool) -> torch.Tensor:
    """Wspólna transformacja pojedynczej epoki dla obu datasetów."""
    if feature_method == "raw":
        # Z-SCORE Normalization (Per channel)
        # Should prevent data leakage and MNE scaling issues
        mean = x_val.mean(axis=-1, keepdims=True)
        std = x_val.std(axis=-1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        x_norm = (x_val - mean) / std

        x_tensor = torch.tensor(x_norm, dtype=torch.float32)

        # Architecture Shape Requirements
        # Transformers: (Time, Channels) -> e.g. (512, 64)
        if sequence_first:
            x_tensor = x_tensor.permute(1, 0)
        return x_tensor

    if feature_method == "tsp":
        # TSP Data is already (Windows, TSP_Features) and centered.
        # No need for Z-score per epoch or permutation.
        return torch.tensor(x_val, dtype=torch.float32)

    _check_feature_method(feature_method)
    raise AssertionError("unreachable")


class GlobalTestDataset(Dataset):
    """
    Used ONLY for the globally isolated test set.
    """

    def __init__(self, x, y, subject_ids=None, sequence_first=True, feature_method="raw"):
        """
        Args:
            x: Numpy array of shape (N_epochs, Channels, Time)
            y: Numpy array of labels
            sequence_first: If True, permutes shape to (Time, Channels) for Transformers.
            feature_method: 'raw' or 'tsp'. Defines how to process the tensor in __getitem__
        """
        _check_feature_method(feature_method)
        self.sequence_first = sequence_first
        self.subject_ids = subject_ids
        self.feature_method = feature_method

        # Ensure labels are explicitly 0 and 1 (Technically not necessary here but safety first).
        y_array = np.array(y)
        unique_y = np.unique(y_array)
        if not np.isin(unique_y, (0, 1)).all():
            print(f"WARNING: Unexpected labels found: {unique_y}. Forcing mapping via modulo 2.")
            y_array = y_array % 2

        self.y = torch.tensor(y_array, dtype=torch.long)
        self.X = x  # Keep as numpy initially to save RAM

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Load single epoch
        x_val = self.X[idx].astype(np.float32)
        label = self.y[idx]
        x_tensor = _prepare_input(x_val, self.feature_method, self.sequence_first)
        return x_tensor, label


class LazySubjectDataset(Dataset):
    """
    RAM-efficient PyTorch Dataset for training/validation.
    Points directly to the loaded subject_data dictionary.
    """

    def __init__(self, subject_data, subjects_list, sequence_first=False, feature_method="raw"):
        _check_feature_method(feature_method)
        self.subject_data = subject_data
        self.subjects_list = subjects_list
        self.sequence_first = sequence_first
        self.feature_method = feature_method

        # Create an index map (subject_name, local_epoch_idx)
        self.index_map = []
        for subj in self.subjects_list:
            num_epochs = len(self.subject_data[subj][0])
            for epoch_idx in range(num_epochs):
                self.index_map.append((subj, epoch_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        subj, epoch_idx = self.index_map[idx]

        # Extract epoch from dictionary
        x_val = self.subject_data[subj][0][epoch_idx].astype(np.float32)
        y_val = self.subject_data[subj][1][epoch_idx]
        y_tensor = torch.tensor(y_val, dtype=torch.long)

        # Ensure label is 0 or 1
        if y_tensor > 1:
            y_tensor = y_tensor % 2

        x_tensor = _prepare_input(x_val, self.feature_method, self.sequence_first)
        return x_tensor, y_tensor


def load_preprocessed_patient_data(
    data_dir: str,
    segment_type: str = "4s",
    prefix: str = "",
    global_expected_time_points=None,
):
    """
    Loads preprocessed EEG epochs (.fif files) for either Physionet or GigaScience.
    Assigns the data to separate keys (patient names) in a dictionary.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} does not exist. Run preprocessing first.")

    subject_data = {}
    subjects = sorted(os.listdir(data_dir))
    print(f"Loading preprocessed data from {data_dir} (segment type: {segment_type})...")

    time_array = None
    t0_idx = None
    expected_time_points = global_expected_time_points

    for subj_folder in subjects:
        subj_path = os.path.join(data_dir, subj_folder)
        if not os.path.isdir(subj_path):
            continue

        # Look for any file ending with the specific segment type
        file_path = None
        for filename in os.listdir(subj_path):
            if filename.endswith(f"{segment_type}-epo.fif"):
                file_path = os.path.join(subj_path, filename)
                break

        if file_path is None:
            continue

        try:
            epochs = mne.read_epochs(file_path, preload=True, verbose=False)

            # Extract time array and t=0 index only once
            if time_array is None:
                time_array = epochs.times
                t0_idx = np.argmin(np.abs(time_array - 0.0))
                if expected_time_points is None:
                    expected_time_points = len(time_array)

            # Labels should be standarized to 0 and 1 !!!
            x = epochs.get_data(copy=True).astype(np.float32)  # Shape: (epochs, channels, time)
            y = epochs.events[:, -1]  # Target class

            actual_time_points = x.shape[2]
            if actual_time_points != expected_time_points:
                print(
                    f"[WARNING!]: Patient {subj_folder} has {actual_time_points} time points instead of expected {expected_time_points}. Skipping to prevent crash."
                )
                continue

            # Sanity check to ensure labels are correct (0 and 1)
            valid_labels = np.isin(y, [0, 1])
            if not np.all(valid_labels):
                print(f"[WARNING!]: Patient {subj_folder} has unexpected labels: {np.unique(y)}. Skipping.")
                continue

            # Save to dictionary
            final_subj_id = f"{prefix}_{subj_folder}" if prefix else subj_folder
            subject_data[final_subj_id] = (x, y)
            del epochs

        except Exception as e:
            print(f"Skipping patient {subj_folder}. Error: {e}")

    gc.collect()

    if not subject_data:
        raise ValueError(f"No valid data loaded for segment type {segment_type}. Check paths.")

    print(f"Successfully loaded data for {len(subject_data)} patients.")
    if time_array is not None:
        print(f"Sample time bounds: {time_array[0]:.2f}s to {time_array[-1]:.2f}s.")
        print(f"Event stimulus (t=0) located at index: {t0_idx}")

    return subject_data, time_array, t0_idx, expected_time_points


def create_train_test_split(
    subject_data: dict,
    save_dir: str,
    test_patient_count: int = 15,
    batch_size: int = 16,
    test_subjects_list: list | None = None,
    *,
    sequence_first: bool,
    feature_method: str = "raw",
) -> tuple[dict, DataLoader]:
    """
    Splits the dataset into a strict, isolated Global Test Set (hold-out) and a Training Set.
    Returns the training dictionary and the test DataLoader.
    Saves the global test raw data to disk safely.

    UWAGA: funkcja zużywa `subject_data`. Pacjenci testowi są z niego usuwani (pop),
    żeby nie trzymać w RAM podwójnej kopii danych.
    """
    print(" --- SPLITTING DATASET (Strict Cross-Subject Isolation) ")

    all_subjects = np.array(list(subject_data.keys()))

    if test_subjects_list is not None:
        # User defined exactly which subjects should be the test set
        missing_subjects = [subj for subj in test_subjects_list if subj not in all_subjects]
        if missing_subjects:
            raise ValueError(f"Missing subjects requested for testing: {missing_subjects}")

        global_test_subjects = np.array(test_subjects_list)
        training_subjects = np.array([s for s in all_subjects if s not in global_test_subjects])
        print(f"Using defined list of {len(global_test_subjects)} patients for Global Test.")

    else:
        # Randomly sample hold-out test set
        np.random.seed(42)  # Fixed seed
        np.random.shuffle(all_subjects)

        if len(all_subjects) <= test_patient_count:
            raise ValueError(f"Not enough patients ({len(all_subjects)}) to hold out {test_patient_count}.")

        global_test_subjects = all_subjects[:test_patient_count]
        training_subjects = all_subjects[test_patient_count:]
        print(f"Randomly held out {len(global_test_subjects)} patients for Global Test.")

    # --- Process Global test set
    x_test_list, y_test_list, subject_id_test_list = [], [], []

    for subj in global_test_subjects:
        x, y = subject_data.pop(subj)
        x_test_list.append(x)
        y_test_list.append(y)
        subject_id_test_list.extend([subj] * len(x))

    x_global_test = np.concatenate(x_test_list, axis=0)
    y_global_test = np.concatenate(y_test_list, axis=0)

    del x_test_list, y_test_list
    gc.collect()

    print(f"Global test data shape: X = {x_global_test.shape}, Y = {y_global_test.shape}")

    # Save tensors/arrays
    save_path = os.path.join(save_dir, "global_test_data.pt")
    os.makedirs(save_dir, exist_ok=True)

    test_data_dict = {
        "x_test": x_global_test,
        "y_test": y_global_test,
        "subjects": subject_id_test_list,
    }
    torch.save(test_data_dict, save_path)
    print(f"Global test data saved to: '{save_path}'")

    # Create Dataloader for immediate use in current run
    global_test_dataset = GlobalTestDataset(
        x_global_test,
        y_global_test,
        subject_ids=subject_id_test_list,
        sequence_first=sequence_first,
        feature_method=feature_method,
    )

    global_test_loader = DataLoader(global_test_dataset, batch_size=batch_size, shuffle=False)

    # --- Prepare remaining training set
    training_subject_data = {subj: subject_data[subj] for subj in training_subjects}

    print(f"Returning remaining set of {len(training_subject_data)} patients.\n")

    return training_subject_data, global_test_loader


def get_channel_names_from_header(data_dir: str, segment_type: str) -> list:
    """
    Looks for any .fif file created by preprocessing.
    Reads the header (preload=False) and returns the names of channels from epochs.
    """
    for root, _dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(f"{segment_type}-epo.fif"):
                filepath = os.path.join(root, file)
                # reads only metadata
                epochs_info = mne.read_epochs(filepath, preload=False, verbose=False)
                return epochs_info.ch_names

    raise FileNotFoundError(f"Nie znaleziono plików *{segment_type}-epo.fif w {data_dir}")
