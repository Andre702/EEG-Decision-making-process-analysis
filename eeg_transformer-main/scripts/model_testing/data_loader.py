import os
import numpy as np
import torch
import mne
from torch.utils.data import Dataset, DataLoader


# Ładowanie danych =====================================================================================
class SimpleEEGDataset(Dataset):
    def __init__(self, x, y, subject_ids=None, is_train=False):
        self.is_train = is_train
        self.X = torch.tensor(x, dtype=torch.float32)
        if self.X.ndim == 3:
            self.X = self.X.unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)
        self.subject_ids = subject_ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Per-channel Z-score normalization along the time dimension
        x = self.X[idx].clone()
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / torch.clamp(std, min=1e-5)
        return x, self.y[idx]

class LazySubjectDataset(Dataset):
    """
    PyTorch Dataset pointing directly to the loaded subject_data dictionary.
    Eliminates the need to call np.concatenate and duplicate gigabytes of RAM.
    """
    def __init__(self, subject_data, subjects_list):
        self.subject_data = subject_data
        self.subjects_list = subjects_list

        # Tworzymy mapę indeksów: (nazwa_pacjenta, numer_epoki_u_tego_pacjenta)
        self.index_map = []
        for subj in self.subjects_list:
            num_epochs = len(self.subject_data[subj][0])
            for epoch_idx in range(num_epochs):
                self.index_map.append((subj, epoch_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        subj, epoch_idx = self.index_map[idx]

        # Pobieramy dane i wymuszamy konwersję do float32 (oszczędność pamięci)
        x = self.subject_data[subj][0][epoch_idx].astype(np.float32)
        y = self.subject_data[subj][1][epoch_idx]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def load_physionet_data(data_dir, segment_type="6s"): #Bazowo tylko 6-sekundowe próbki
    all_x, all_y = [], []
    subjects = sorted(os.listdir(data_dir))
    print(f"Loading normalized data from {data_dir} (variant: {segment_type})...")

    # Variables for capturing timestamps from MNE
    time_array = None
    t0_idx = None

    for subj_folder in subjects:
        subj_path = os.path.join(data_dir, subj_folder)
        if not os.path.isdir(subj_path): continue

        expected_filename = f"PA{subj_folder[1:4]}-{segment_type}-epo.fif"
        file_path = os.path.join(subj_path, expected_filename)

        if not os.path.exists(file_path):
            continue

        try:
            epochs = mne.read_epochs(file_path, preload=True, verbose=False)

            if time_array is None:
                time_array = epochs.times
                t0_idx = np.argmin(np.abs(time_array - 0.0))

            # PHYSIONET SPECIFIC:
            # 2 = left hand
            # 3 = right hand
            events = epochs.events[:, -1]
            target_events = [2, 3]

            # Safety in case of no events present
            epochs = epochs[np.isin(epochs.events[:, -1], target_events)]
            X = epochs.get_data(copy=True)
            y = epochs.events[:, -1]

            # Mapping left hand (2) = 0, right hand (3) = 1 -----------------------------------
            y = np.where(y == 2, 0, 1)

            all_x.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"Omitting {subj_folder}: Encountering error -> {e}")

    if not all_x:
        raise ValueError(f"Could not find files in format: {segment_type}")

    print(f"Dataset time bounds: {time_array[0]:.2f}s to {time_array[-1]:.2f}s")
    if t0_idx is not None:
        print(f"Event time (t=0) in sample index: {t0_idx}")

    return np.concatenate(all_x), np.concatenate(all_y)


def load_physionet_separate_patient_data(data_dir, segment_type="4s"):
    """
    Loads EEG samples from Physionet data assigning it to the separate keys (patient names)
    """
    subject_data = {}
    subjects = sorted(os.listdir(data_dir))
    print(f"Loading Physionet preprocessed data from {data_dir} (variant: {segment_type})...")

    time_array = None
    t0_idx = None

    for subj_folder in subjects:
        subj_path = os.path.join(data_dir, subj_folder)
        if not os.path.isdir(subj_path):
            continue

        expected_filename = f"PA{subj_folder[1:4]}-{segment_type}-epo.fif"
        file_path = os.path.join(subj_path, expected_filename)

        if not os.path.exists(file_path):
            continue

        try:
            epochs = mne.read_epochs(file_path, preload=True, verbose=False)

            if time_array is None:
                time_array = epochs.times
                # Find index where time is closest to 0.0
                t0_idx = np.argmin(np.abs(time_array - 0.0))

            # PHYSIONET SPECIFIC:
            # 2 = left hand
            # 3 = right hand
            target_events = [2, 3]
            mask = np.isin(epochs.events[:, -1], target_events)

            if not np.any(mask):
                continue  # Patient does not have event daya, omit patient

            epochs = epochs[mask]
            x = epochs.get_data(copy=True)
            y = epochs.events[:, -1]

            # Mapping left hand (2) = 0, right hand (3) = 1
            y = np.where(y == 2, 0, 1)

            # Save to dictionary
            subject_data[subj_folder] = (x, y)

        except Exception as e:
            print(f"Skipping patient {subj_folder}. Error: {e}")

    if not subject_data:
        raise ValueError(f"No files for {segment_type}")

    print(f"Loaded {len(subject_data)} patients data.")
    if time_array is not None:
        print(f"Sample time starts at {time_array[0]:.2f}s, ends at {time_array[-1]:.2f}s.")
        print(f"Test beginning (t=0) is at index: {t0_idx}")

    return subject_data, time_array, t0_idx

def load_gigadb_separate_patient_data(data_dir, segment_type="4s"):
    """
    Loads EEG samples from GigaDB data assigning it to the separate keys (patient names)
    Dynamically resolves event mapping from MNE .fif metadata
    """
    subject_data = {}
    subjects = sorted(os.listdir(data_dir))
    print(f"Loading GigaDB preprocessed data from {data_dir} (variant: {segment_type})...")

    time_array = None
    t0_idx = None

    for subj_folder in subjects:
        subj_path = os.path.join(data_dir, subj_folder)
        if not os.path.isdir(subj_path):
            continue

        # Look for files based on naming convention (PA001-4s-epo.fif)
        expected_filename = f"PA{subj_folder[1:4]}-{segment_type}-epo.fif"
        file_path = os.path.join(subj_path, expected_filename)

        if not os.path.exists(file_path):
            continue

        try:
            epochs = mne.read_epochs(file_path, preload=True, verbose=False)

            if time_array is None:
                time_array = epochs.times
                t0_idx = np.argmin(np.abs(time_array - 0.0))

            # GigaDB set has event ids saved in metadata, so we take the index from there
            left_code = epochs.event_id.get("left_hand")
            right_code = epochs.event_id.get("right_hand")

            if left_code is None or right_code is None:
                left_code, right_code = 1, 2

            target_events = [left_code, right_code]
            mask = np.isin(epochs.events[:, -1], target_events)

            if not np.any(mask):
                continue  # Patient did not have class markings for standard IDs

            epochs = epochs[mask]
            x = epochs.get_data(copy=True)
            y = epochs.events[:, -1]

            # Map events to 0 and 1
            y = np.where(y == left_code, 0, 1)

            # Save to dictionary
            subject_data[subj_folder] = (x, y)

        except Exception as e:
            print(f"Skipping patient {subj_folder}. Error: {e}")

    if not subject_data:
        raise ValueError(f"No files for {segment_type} in {data_dir}")

    print(f"Loaded {len(subject_data)} patients data from GigaDB.")
    if time_array is not None:
        print(f"Sample time starts at {time_array[0]:.2f}s, ends at {time_array[-1]:.2f}s.")
        print(f"Test beginning (t=0) is at index: {t0_idx}")

    return subject_data, time_array, t0_idx

def load_bci_separate_patient_data(data_dir):
    """
    Loads EEG samples From BCI data assigning it to the separate keys (patient names)
    """
    subject_data = {}
    subjects = sorted(os.listdir(data_dir))
    print(f"Loading BCI preprocessed data from {data_dir}")

    time_array = None
    t0_idx = None

    for subj_folder in subjects:
        subj_path = os.path.join(data_dir, subj_folder)
        if not os.path.isdir(subj_path):
            continue

        expected_filename = f"PA{subj_folder[1:3]}T-epo.fif"
        file_path = os.path.join(subj_path, expected_filename)

        if not os.path.exists(file_path):
            continue

        try:
            epochs = mne.read_epochs(file_path, preload=True, verbose=False)

            if time_array is None:
                time_array = epochs.times
                # Find index where time is closest to 0.0
                t0_idx = np.argmin(np.abs(time_array - 0.0))

            # This assumes that events (left, right) are mapped to 0 and 1
            target_events = [0, 1]
            mask = np.isin(epochs.events[:, -1], target_events)
            if not np.any(mask):
                continue  # Patient does not have event daya, omit patient

            epochs = epochs[mask]
            x = epochs.get_data(copy=True)
            y = epochs.events[:, -1]  # Values are strictly 0 and 1 now

            subject_data[subj_folder] = (x, y)

        except Exception as e:
            print(f"Skipping {subj_folder}. Error: {e}")

    if not subject_data:
        raise ValueError(f"No valid BCI files found in {data_dir}")

    print(f"Loaded {len(subject_data)} patients data.")
    if time_array is not None:
        print(f"Sample time starts at: {time_array[0]:.2f}s, ends at {time_array[-1]:.2f}s.")
        print(f"Test beginning (t=0) is at index: {t0_idx}")

    return subject_data, time_array, t0_idx

def split_dataset_return_training(subject_data, test_patient_count=15, batch_size=32, test_subjects_list=None):
    """
    Splits the dataset into a separate, isolated global test set
    and a remaining set for model training/cross-validation.
    If 'test_subjects_list' is provided, those exact subjects will be used for testing.
    Otherwise, it randomly splits 'test_patient_count' subjects.
    Saves the global test dataloader.
    Returns remaining data for training.
    """
    print("\n" + "=" * 50)
    print("Splitting dataset into training and global test set")
    print("=" * 50)

    # Gets all patient IDs
    all_subjects = np.array(list(subject_data.keys()))

    if test_subjects_list is not None:
        missing_subjects = [subj for subj in test_subjects_list if subj not in all_subjects]
        if missing_subjects:
            raise ValueError(
                f"The following subjects from test list are missing in the dataset!: {missing_subjects}")

        global_test_subjects = np.array(test_subjects_list)
        # other test subjects are for training
        training_subjects = np.array([subj for subj in all_subjects if subj not in global_test_subjects])

        print(f"Using custom list of {len(global_test_subjects)} patients for the Global Test Set.")
    else:
        np.random.seed(702)
        np.random.shuffle(all_subjects)

        if len(all_subjects) <= test_patient_count:
            raise ValueError(
                f"Not enough patients to create the hold-out set of set size {test_patient_count}.")

        global_test_subjects = all_subjects[:test_patient_count]
        training_subjects = all_subjects[test_patient_count:]

        print(f"Randomly selected {len(global_test_subjects)} patients for the Global Test Set.")
    # -----------------------------------------

    print(f"Global test set: {len(global_test_subjects)} patients.")
    print(f"Returning remaining: {len(training_subjects)} patients.")

    # Extract data for the Global test set
    x_test_list, y_test_list = [], []
    subject_id_test_list = []
    for subj in global_test_subjects:
        x, y = subject_data[subj]
        x_test_list.append(x)
        y_test_list.append(y)

        num_epochs_for_subject = len(x)
        subject_id_test_list.extend([subj] * num_epochs_for_subject)

    x_global_test = np.concatenate(x_test_list, axis=0)
    y_global_test = np.concatenate(y_test_list, axis=0)

    # Save global data loader for all tests
    global_test_dataset = SimpleEEGDataset(x_global_test, y_global_test, is_train=False, subject_ids=subject_id_test_list)
    global_test_loader = DataLoader(global_test_dataset, batch_size=batch_size, shuffle=False)

    loader_path = f"./saved_model_states/global_data_loader/test_loader_batch_{batch_size}.pth"
    os.makedirs(os.path.dirname(loader_path), exist_ok=True)

    torch.save(global_test_loader, loader_path)

    print(f"Global test data loader saved to: '{loader_path}'")

    # Prepare the remaining data to be returned for training loop
    remaining_subject_data = {subj: subject_data[subj] for subj in training_subjects}

    return remaining_subject_data

def load_bci3a_data(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads preprocessed MNE Epochs for the BCI III 3a dataset.
    Returns standardized features X and binary labels y meaning: 0 - Left Hand, 1 - Right Hand
    """
    all_x, all_y = [], []

    # Filter only directories ('S1', 'S2', 'S3') as generated by preprocessing
    subjects = [f for f in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, f))]
    print(f"Loading and normalizing data from {data_dir}...")

    for subj_folder in subjects:
        subj_path = os.path.join(data_dir, subj_folder)

        expected_filename = f"{subj_folder[1:3]}-epo.fif"
        file_path = os.path.join(subj_path, expected_filename)

        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue

        try:
            epochs = mne.read_epochs(file_path, preload=True, verbose=False)

            # Extract raw signal and event labels
            X = epochs.get_data(copy=True)
            y = epochs.events[:, -1]

            # Base BCI III 3a Mapping:
            # 3 = left hand (Event id: 769)
            # 4 = right hand (Event id: 770)

            # Filter everything else (although I think there should not be any Tongue or Foot experiments in this dataset)
            valid_indices = np.isin(y, [3, 4])
            X = X[valid_indices]
            y = y[valid_indices]

            # Convert to binary target values
            y = np.where(y == 3, 0, 1)

            # Apply per-channel z-score scaling across the time axis (for Neural Networks)
            mean = np.mean(X, axis=2, keepdims=True)
            std = np.std(X, axis=2, keepdims=True)
            std[std == 0] = 1.0
            X = (X - mean) / std

            all_x.append(X)
            all_y.append(y)

            print(f"Subject {subj_folder} loaded: {X.shape[0]} samples.")

        except Exception as e:
            print(f"Skipped {subj_folder} due to error: {e}")

    if not all_x:
        raise ValueError("No matching .fif files found in the directory.")

    # Convert to appropriate numpy types for PyTorch
    final_x = np.concatenate(all_x).astype(np.float32)
    final_y = np.concatenate(all_y).astype(np.int64)

    print(f"Dataset successfully loaded. Total shape: X={final_x.shape}, y={final_y.shape}")
    return final_x, final_y

# Do późniejszego przetestowania
def load_bci2a_data_normalized(data_dir):
    all_X, all_y = [], []
    subjects = sorted(os.listdir(data_dir))
    print(f"Ładowanie danych z {data_dir}...")

    for subj_folder in subjects:
        subj_path = os.path.join(data_dir, subj_folder)
        if not os.path.isdir(subj_path): continue
        files = [f for f in os.listdir(subj_path) if f.endswith('.fif')]
        if not files: continue

        try:
            epochs = mne.read_epochs(os.path.join(subj_path, files[0]), preload=True, verbose=False)

            events = epochs.events[:, -1]
            unique_ev = np.unique(events)
            target_events = unique_ev[:2]

            epochs = epochs[np.isin(epochs.events[:, -1], target_events)]
            X = epochs.get_data(copy=True)
            y = epochs.events[:, -1]

            # Mapowanie etykiet na 0 i 1
            y = np.where(y == target_events[0], 0, 1)

            # NORMALIZACJA (Z-SCORE)
            mean = np.mean(X, axis=2, keepdims=True)
            std = np.std(X, axis=2, keepdims=True)
            std[std == 0] = 1.0
            X = (X - mean) / std

            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"Skip {subj_folder}: {e}")

    if not all_X: return np.array([]), np.array([])
    return np.concatenate(all_X), np.concatenate(all_y)

