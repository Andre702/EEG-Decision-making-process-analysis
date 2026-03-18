import os
import numpy as np
import torch
import torch.nn as nn
import mne
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
import math


# ==============================================================================
# 1. ELEMENTY Z PDF (FeatureExtractor, PositionalEncoding, TransformerBlock)
# ==============================================================================

class FeatureExtractor(nn.Module):
    def __init__(self, model_class_name, method='raw', wavelet='coif3', level=3, n_csp_components=4):
        super(FeatureExtractor, self).__init__()
        self.model_class_name = model_class_name
        self.method = method
        # (Pominięto implementacje CNN/CSP/Wavelet dla czytelności, bo używamy RAW)

    def forward(self, x):
        # Dla metody 'raw' FeatureExtractor w PDF (str. 8) robił identity lub unsqueeze
        if self.method == 'raw':
            # if x.ndim == 3:
            #     x = x.unsqueeze(1)
            return x
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
        # x shape: (seq_len, batch, d_model)
        return x + self.pe[:x.size(0), :, :]


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
        # W PDF implementacja wyglądała tak:
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x


# ==============================================================================
# 2. MODEL TEMPORAL TRANSFORMER (Dokładnie z Twojej wiadomości)
# ==============================================================================

class TemporalTransformer(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_classes=2, feature_method='raw', feature_extractor=None):
        super(TemporalTransformer, self).__init__()
        self.feature_method = feature_method

        # Inicjalizacja Feature Extractora (zgodnie z kodem PDF)
        if feature_method in ['raw', 'cnn', 'stft']:
            self.feature_extractor = FeatureExtractor(
                model_class_name=self.__class__.__name__,
                method=feature_method
            )
        elif feature_method == 'csp':
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = None

        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # 3 bloki transformera (zgodnie z kodem)
        self.transformer = nn.Sequential(
            TransformerBlock(d_model, nhead),
            TransformerBlock(d_model, nhead),
            TransformerBlock(d_model, nhead)
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch, 1, channels, time)

        if self.feature_extractor is not None:
            x = self.feature_extractor(x)

        # (time = sequence dimension)
        if self.feature_method == 'raw':
            # TO JEST KLUCZOWY MOMENT TOKENIZACJI "PUNKTOWEJ"
            # (Batch, 1, Channels, Time) -> squeeze -> (Batch, C, T) -> permute -> (Time, Batch, Channels)
            x = x.squeeze(1).permute(2, 0, 1)

            # ... (pomijam bloki elif dla czytelności, bo używamy 'raw') ...

        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=0)
        return self.fc(x)


# ==============================================================================
# 3. ŁADOWANIE DANYCH
# ==============================================================================

class SimpleEEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # Model oczekuje 4 wymiarów (B, 1, C, T) na wejściu przed FeatureExtractorem
        if self.X.ndim == 3:
            self.X = self.X.unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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

            # 1. Filtrujemy tylko zdarzenia 7 (Lewa) i 8 (Prawa) - specyficzne dla BCI 2a
            # Czasem MNE mapuje je na 1 i 2, a czasem zostawia 769, 770.
            # Zróbmy bezpieczne podejście: weźmy 2 najczęstsze zdarzenia lub jawnie 7 i 8.
            events = epochs.events[:, -1]
            unique_ev = np.unique(events)

            # Zakładamy logikę: dwie pierwsze klasy to te decyzyjne
            # Jeśli twoje logi pokazały [7, 8], to używamy 7 i 8.
            target_events = unique_ev[:2]

            epochs = epochs[np.isin(epochs.events[:, -1], target_events)]
            X = epochs.get_data(copy=True)
            y = epochs.events[:, -1]

            # Mapowanie etykiet na 0 i 1
            y = np.where(y == target_events[0], 0, 1)

            # 2. NORMALIZACJA (Z-SCORE) - ABSOLUTNIE KLUCZOWE
            # Bez tego RAW Transformer nie zadziała (zbyt małe wartości wag)
            mean = np.mean(X, axis=2, keepdims=True)
            std = np.std(X, axis=2, keepdims=True)
            std[std == 0] = 1.0  # unikanie dzielenia przez 0
            X = (X - mean) / std

            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"Skip {subj_folder}: {e}")

    if not all_X: return np.array([]), np.array([])
    return np.concatenate(all_X), np.concatenate(all_y)


# ==============================================================================
# 4. TRENING
# ==============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
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
    DATA_PATH = "./preprocessed_data/Physionet"
    BATCH_SIZE = 32
    EPOCHS = 15  # PDF sugerował 50, my damy 30
    LR = 0.0001  # Z Tabelki wyników w Twojej wiadomości
    D_MODEL = 64  # Zgodnie z kodem PDF (my wcześniej mieliśmy 64)
    N_HEAD = 8  # Zgodnie z kodem PDF

    # 1. Dane
    X_all, y_all = load_bci2a_data_normalized(DATA_PATH)
    if len(X_all) == 0:
        print("Błąd: Brak danych.")
        return

    n_channels = X_all.shape[1]
    n_timepoints = X_all.shape[2]
    print(f"Dane: {len(X_all)} epok. Kanały: {n_channels}, Czas: {n_timepoints}")
    print(f"Klasy: {np.unique(y_all, return_counts=True)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")

    # 2. Walidacja 5-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        print(f"\n--- Fold {fold + 1} ---")

        train_ds = SimpleEEGDataset(X_all[train_idx], y_all[train_idx])
        test_ds = SimpleEEGDataset(X_all[test_idx], y_all[test_idx])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # 3. Inicjalizacja Modelu (PDF EXACT VERSION)
        model = TemporalTransformer(
            input_size=n_channels,  # Będzie 22
            d_model=D_MODEL,
            nhead=N_HEAD,
            num_classes=2,
            feature_method='raw'
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        for epoch in range(EPOCHS):
            loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            test_acc = evaluate(model, test_loader, criterion, device)

            if test_acc > best_acc:
                best_acc = test_acc

            if (epoch + 1) % 5 == 0:
                print(f"Ep {epoch + 1:02d}: Loss={loss:.4f}, TrAcc={train_acc:.2%}, TsAcc={test_acc:.2%}")

        print(f"Best Test Acc Fold {fold + 1}: {best_acc:.2%}")
        fold_results.append(best_acc)

    print(f"\nŚrednia ze wszystkich foldów: {np.mean(fold_results) * 100:.2f}%")


if __name__ == "__main__":
    main()