import os
import numpy as np
import torch
import torch.nn as nn
import mne
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
import math


# ==============================================================================
# 1. IMPLEMENTACJA MODELU (TemporalTransformer - Uniwersalny)
# ==============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=0.3)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Time, Batch, Channels/Features)
        return x + self.pe[:x.size(0), :, :]


class TemporalTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=8, num_classes=2):
        super(TemporalTransformer, self).__init__()
        print(f"--> Inicjalizacja modelu dla {input_size} kanałów (elektrod)")
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Sequential(
            TransformerBlock(d_model, nhead),
            TransformerBlock(d_model, nhead),
            TransformerBlock(d_model, nhead)
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Wejście x: (Batch, 1, Channels, Time)
        # Transformacja pod Transformera: (Time, Batch, Channels)
        x = x.squeeze(1).permute(2, 0, 1)

        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        # Pooling (średnia po czasie)
        x = x.mean(dim=0)
        return self.fc(x)


# ==============================================================================
# 2. IMPLEMENTACJA LAZY DATASET (Zrekonstruowana z PDF str. 29-30)
# ==============================================================================

class LazyPhysionetWindowDataset(Dataset):
    """
    Odtworzona klasa z PDF-a. Przyjmuje listę macierzy (per pacjent) i pozwala
    na dzielenie ich na okna.
    """

    def __init__(self, X_list, y_list, window_size=None, n_windows=1):
        self.X_list = X_list  # Lista arrayów [(Epoki, Kanały, Czas), ...]
        self.y_list = y_list  # Lista etykiet
        self.n_windows = n_windows

        # Jeśli nie podano rozmiaru okna, bierzemy całą długość sygnału (bez cięcia)
        # To adaptacja pod BCI 2a, gdzie epoki są już pocięte poprawnie
        if window_size is None:
            # Sprawdzamy długość pierwszego pacjenta
            self.window_size = X_list[0].shape[2]
        else:
            self.window_size = window_size

        self.samples = []
        # Precompute indices: (indeks_pacjenta, indeks_epoki, start_okna)
        for subj_idx, X in enumerate(X_list):
            for ep_idx in range(len(X)):
                n_times = X.shape[2]

                # Logika z PDF: wyznaczanie startów okien
                if n_windows > 1:
                    step = max(1, n_times // n_windows)
                    starts = [i * step for i in range(n_windows)]
                else:
                    starts = [0]  # Tylko jedno okno (cała epoka)

                for start in starts:
                    if start + self.window_size <= n_times:
                        self.samples.append((subj_idx, ep_idx, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subj_idx, ep_idx, start = self.samples[idx]

        # Pobieramy dane
        X_full = self.X_list[subj_idx][ep_idx]  # (Kanały, Czas)
        y = self.y_list[subj_idx][ep_idx]

        # Wycinamy okno (Slice)
        # Jeśli n_windows=1, to bierze cały sygnał
        window = X_full[:, start: start + self.window_size]

        # Dodajemy wymiar (1) aby pasowało do (1, C, T) jak w modelu
        window = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.long)

        return window, y


# ==============================================================================
# 3. ŁADOWANIE DANYCH BCI IV 2a
# ==============================================================================

def load_bci2a_data(data_dir):
    all_X_list = []
    all_y_list = []

    if not os.path.exists(data_dir):
        raise ValueError(f"Katalog {data_dir} nie istnieje!")

    subjects = sorted(os.listdir(data_dir))
    print(f"Znaleziono foldery pacjentów: {subjects}")

    for subj_folder in subjects:
        subj_path = os.path.join(data_dir, subj_folder)
        if not os.path.isdir(subj_path): continue

        # Szukamy pierwszego pliku .fif
        files = [f for f in os.listdir(subj_path) if f.endswith('.fif')]
        if not files: continue

        file_path = os.path.join(subj_path, files[0])
        print(f"Wczytywanie i normalizacja: {file_path}")

        try:
            epochs = mne.read_epochs(file_path, preload=True, verbose=False)

            # --- POPRAWKA 1: MAPOWANIE KLAS 7 i 8 ---
            # Twoje dane mają klasy 7 i 8. Zakładamy, że:
            # 7 = Lewa Ręka -> zmapuj na 0
            # 8 = Prawa Ręka -> zmapuj na 1

            # Filtrujemy tylko zdarzenia 7 i 8
            target_events = [7, 8]
            epochs = epochs[np.isin(epochs.events[:, -1], target_events)]

            X = epochs.get_data(copy=True)  # (N, Channels, Time) - w mikrowoltach
            y = epochs.events[:, -1]

            # Mapowanie na 0 i 1
            y = np.where(y == 7, 0, 1)  # Jeśli 7 to 0, w przeciwnym razie 1 (czyli 8)

            # --- POPRAWKA 2: NORMALIZACJA (Z-SCORE) ---
            # X ma kształt (Epoki, 22, Czas)
            # Chcemy normalizować każdą próbkę w każdym kanale niezależnie
            # Wzór: (wartość - średnia) / odchylenie

            # Obliczamy średnią i std dla każdej epoki i każdego kanału (axis=2 to czas)
            mean = np.mean(X, axis=2, keepdims=True)
            std = np.std(X, axis=2, keepdims=True)

            # Zabezpieczenie przed dzieleniem przez zero (bardzo rzadkie, ale warto)
            std[std == 0] = 1.0

            # Aplikacja normalizacji - teraz dane będą miały zakres ok. -3 do 3 zamiast 1e-6
            X = (X - mean) / std

            print(f"   -> Statystyki po naprawie: Std={np.std(X):.2f}, Etykiety={np.unique(y)}")

            all_X_list.append(X)
            all_y_list.append(y)

        except Exception as e:
            print(f"Błąd przy {subj_folder}: {e}")

    return all_X_list, all_y_list


# ==============================================================================
# 4. GŁÓWNA PĘTLA TRENINGOWA
# ==============================================================================

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    # Ścieżka do danych BCI 2a (z preprocess.py)
    DATA_PATH = "./preprocessed_data/BCI_IV_2a"

    print(f"--- EKSPERYMENT 2: Zbiór BCI IV 2a z LazyDataset ---")

    # 1. Wczytaj dane jako listę (niezłączone)
    try:
        X_list, y_list = load_bci2a_data(DATA_PATH)
    except Exception as e:
        print(f"Krytyczny błąd: {e}")
        return

    if len(X_list) == 0:
        print("Brak danych.")
        return

    # Sprawdzenie parametrów danych
    n_channels = X_list[0].shape[1]  # Dla BCI 2a powinno być 22
    n_timepoints = X_list[0].shape[2]  # Zależy od preprocessingu (np. 1000 dla 4s przy 250Hz)
    print(f"Parametry danych: {n_channels} kanałów, {n_timepoints} próbek czasu")

    # 2. Walidacja Krzyżowa (Leave-One-Subject-Out lub KFold)
    # Tu zrobimy proste 5-Fold na liście pacjentów (lub danych scalonych, dla uproszczenia LazyDataset spłaszcza strukturę)

    # Dla LazyDataset musimy przekazać listy. Zrobimy podział na poziomie pacjentów.
    # Jeśli mamy 9 pacjentów, zróbmy proste Train/Test split (np. 7 pacjentów train, 2 test)
    # żeby pokazać działanie klasy Lazy.

    split_idx = int(len(X_list) * 0.8)
    X_train, y_train = X_list[:split_idx], y_list[:split_idx]
    X_test, y_test = X_list[split_idx:], y_list[split_idx:]

    print(f"Trening na {len(X_train)} pacjentach, test na {len(X_test)} pacjentach.")

    # 3. Użycie LazyPhysionetWindowDataset
    # Tu nie tniemy okien (window_size=None, n_windows=1), bierzemy całe epoki
    train_ds = LazyPhysionetWindowDataset(X_train, y_train, window_size=None, n_windows=1)
    test_ds = LazyPhysionetWindowDataset(X_test, y_test, window_size=None, n_windows=1)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 4. Inicjalizacja Modelu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # WAŻNE: input_size dynamicznie pobrane z danych (22 dla BCI2a)
    model = TemporalTransformer(input_size=n_channels, d_model=64, nhead=8).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 5. Trening
    epochs = 15
    for epoch in range(epochs):
        loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss:.4f} | Train Acc: {train_acc:.2%} | Test Acc: {test_acc:.2%}")


if __name__ == "__main__":
    main()