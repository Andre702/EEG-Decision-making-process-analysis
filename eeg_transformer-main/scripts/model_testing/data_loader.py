import os
import numpy as np
import torch
import mne
from torch.utils.data import Dataset

# Ładowanie danych =====================================================================================
class SimpleEEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        if self.X.ndim == 3:
            self.X = self.X.unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_physionet_data(data_dir, segment_type="6s"): #Bazowo tylko 6-sekundowe próbki
    all_x, all_y = [], []
    subjects = sorted(os.listdir(data_dir))
    print(f"Loading normalized data from {data_dir} (variant: {segment_type})...")

    for subj_folder in subjects:
        subj_path = os.path.join(data_dir, subj_folder)
        if not os.path.isdir(subj_path): continue

        # hardcodowana struktura plików
        expected_filename = f"PA{subj_folder[1:4]}-{segment_type}-epo.fif"
        file_path = os.path.join(subj_path, expected_filename)

        if not os.path.exists(file_path):
            continue

        try:
            epochs = mne.read_epochs(file_path, preload=True, verbose=False)

            # PHYSIONET SPECIFIC:
            # 2 = left hand
            # 3 = right hand
            events = epochs.events[:, -1]
            target_events = [2, 3]

            # Zabezpieczenie przed brakiem zdarzeń
            epochs = epochs[np.isin(epochs.events[:, -1], target_events)]
            X = epochs.get_data(copy=True)
            y = epochs.events[:, -1]

            # Mapowanie lewa ręka (2) = 0, prawa ręka (3) = 1 -----------------------------------
            y = np.where(y == 2, 0, 1)

            # Skalowanie danych do wykresów ----------------------------------------------------
            mean = np.mean(X, axis=2, keepdims=True)
            std = np.std(X, axis=2, keepdims=True)
            std[std == 0] = 1.0
            X = (X - mean) / std

            all_x.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"Pominięto {subj_folder}: Z powodu błędu -> {e}")

    if not all_x:
        raise ValueError(f"Nie znaleziono plików w formacie {segment_type}")

    return np.concatenate(all_x), np.concatenate(all_y)

# Do późniejszego przetestowania (ładowanie bci2a)
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

