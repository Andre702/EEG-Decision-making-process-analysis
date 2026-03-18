import os
import sys
import numpy as np
import torch
import torch.nn as nn
import mne
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
import math


# --- 1. KOMPONENTY MODELU (z PDF str. 5, 6, 16) ---

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
        return x + self.pe[:x.size(0), :, :]


class FeatureExtractor(nn.Module):
    # Uproszczona wersja - dla metody 'raw' FeatureExtractor jest trywialny
    def __init__(self, method='raw'):
        super(FeatureExtractor, self).__init__()
        self.method = method

    def forward(self, x):
        if self.method == 'raw':
            # x: (B, 1, C, T) -> (B, 1, C, T) - Identity (można dodać unsqueeze jeśli input 3D)
            return x
        return x


class TemporalTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=8, num_classes=2):
        super(TemporalTransformer, self).__init__()

        # input_size = liczba kanałów EEG (dla Physionet = 64)
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # 3 bloki transformera
        self.transformer = nn.Sequential(
            TransformerBlock(d_model, nhead),
            TransformerBlock(d_model, nhead),
            TransformerBlock(d_model, nhead)
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Oczekiwane wejście x: (Batch, 1, Channels, Time)

        # 1. Transformacja do (Time, Batch, Channels)
        # squeeze(1) usuwa wymiar 1 -> (Batch, C, T)
        # permute(2, 0, 1) -> (T, Batch, C)
        x = x.squeeze(1).permute(2, 0, 1)

        # 2. Embedding (C -> d_model)
        x = self.embedding(x)

        # 3. Pozycja + Transformer
        x = self.pos_encoder(x)
        x = self.transformer(x)

        # 4. Pooling po czasie (średnia) i klasyfikacja
        x = x.mean(dim=0)
        return self.fc(x)


# --- 2. DATASET I ŁADOWANIE DANYCH ---

# Wykorzystamy prostszy dataset (podobny do tego co wkleiłeś w wiadomości),
# ale dopasowany do formatu Physionet (Batch, 1, C, T)
class SimpleEEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # Dodajemy wymiar '1' dla zgodności z konwencją kodu (B, 1, C, T)
        if self.X.ndim == 3:
            self.X = self.X.unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_subject_data(file_path):
    # Wczytanie epok z pliku .fif (użycie MNE)
    if not os.path.exists(file_path):
        return None, None

    epochs = mne.read_epochs(file_path, preload=True, verbose=False)
    X = epochs.get_data(copy=True)  # (N_epochs, Channels, Time)
    y = epochs.events[:, -1]  # Event codes

    # Mapowanie klas Physionet:
    # Oryginalnie: T1 (lewa) = 2, T2 (prawa) = 3 (wg kodu w physionet.py)
    # Musimy to zmapować na 0 i 1
    # Zakładamy, że w plikach .fif masz już eventy 2 i 3
    mask = (y == 2) | (y == 3)
    X = X[mask]
    y = y[mask]

    # 2 -> 0, 3 -> 1
    y = np.where(y == 2, 0, 1)
    return X, y


def load_all_data(data_dir):
    all_X, all_y = [], []
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('S')])

    print(f"Znaleziono {len(subjects)} pacjentów w {data_dir}")

    for subject in subjects:
        # Szukamy pliku 6s
        # Struktura: ./preprocessed_data/Physionet/S001/PA001-6s-epo.fif
        file_path = os.path.join(data_dir, subject, f"PA{subject[1:]}-6s-epo.fif")

        X, y = load_subject_data(file_path)
        if X is not None:
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        raise ValueError("Nie wczytano żadnych danych! Sprawdź ścieżki.")

    return np.concatenate(all_X), np.concatenate(all_y)


# --- 3. PĘTLA TRENINGOWA ---

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    return total_loss / total, correct / total


# --- 4. MAIN ---

def main():
    # USTAWIENIA
    PREPROCESSED_PATH = "./preprocessed_data/Physionet"  # Twoja lokalna ścieżka
    BATCH_SIZE = 32
    EPOCHS = 15  # Możesz zwiększyć do 50 jak w PDF
    LR = 0.0001
    N_SPLITS = 5  # 5-fold CV

    # 1. Wczytaj dane
    print("Ładowanie danych...")
    X_all, y_all = load_all_data(PREPROCESSED_PATH)
    print(f"Dane wczytane. Kształt X: {X_all.shape}, y: {y_all.shape}")
    # X shape spodziewany: (Total_Epochs, 64, 960) dla 6s

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    # 2. Walidacja krzyżowa (5-Fold)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        print(f"\n=== FOLD {fold + 1}/{N_SPLITS} ===")

        # Przygotowanie datasetów
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        train_ds = SimpleEEGDataset(X_train, y_train)
        test_ds = SimpleEEGDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # Inicjalizacja modelu
        # input_size = liczba kanałów (np. 64)
        n_channels = X_all.shape[1]
        model = TemporalTransformer(input_size=n_channels, d_model=64, nhead=8).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        # Trening
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

        # Wynik końcowy foldu
        _, final_acc = evaluate(model, test_loader, criterion, device)
        fold_results.append(final_acc)
        print(f"--> Wynik Fold {fold + 1}: {final_acc * 100:.2f}%")

    print(f"\nŚrednia dokładność (Accuracy): {np.mean(fold_results) * 100:.2f}%")


if __name__ == "__main__":
    main()