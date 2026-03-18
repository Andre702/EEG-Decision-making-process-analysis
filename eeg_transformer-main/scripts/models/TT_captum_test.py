import os
import numpy as np
import torch
import torch.nn as nn
import mne
import copy
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset


# MODEL -----------------------------------------------------------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, model_class_name='TemporalTransformer', method='raw'):
        super(FeatureExtractor, self).__init__()
        self.model_class_name = model_class_name
        self.method = method

    def forward(self, x):
        if self.method == 'raw':
            if x.ndim == 3:
                x = x.unsqueeze(1)
            return x
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=3000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
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
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x

class TemporalTransformer(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_classes=2, feature_method='raw', feature_extractor=None):
        super(TemporalTransformer, self).__init__()
        self.feature_method = feature_method

        # Feature Extractor (for now only raw tested) -----------------
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
            x = x.squeeze(1).permute(2, 0, 1)

        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=0)
        return self.fc(x)


# Data Loading ---------------------------------------------------------------------------------

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


def load_physionet_data(data_dir, segment_type="6s"): # By default 6 second samples
    all_X, all_y = [], []
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

            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"Pominięto {subj_folder}: Z powodu błędu -> {e}")

    if not all_X:
        raise ValueError(f"Nie znaleziono plików w formacie {segment_type}")

    return np.concatenate(all_X), np.concatenate(all_y)

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


# Training =====================================================================================

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

        total_loss += loss.item() * X.size(0)  # Mnożenie dla uśrednionej straty
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)

            total_loss += loss.item() * X.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total # (loss | acc)


# Interpretacja Captum i wyniki ==============================================================================

def analyze_with_captum_bulk2(model, test_loader, device, max_samples=200):
    print("\n" + "=" * 60)
    print(f"Analiza klasyfikacji próbek przez Captum z testowego setu")
    print("=" * 60)

    model.eval()
    ig = IntegratedGradients(model)

    attr_class_0 = []
    attr_class_1 = []

    dual_peak_samples = []
    all_samples_info = []

    processed = 0
    print("Obliczanie całek gradientów")

    for X_batch, y_batch in test_loader:
        for i in range(len(X_batch)):
            if processed >= max_samples:
                break

            X_input = X_batch[i].unsqueeze(0).to(device).requires_grad_()
            target_class = y_batch[i].item()

            output = model(X_input)
            predicted_class = torch.argmax(output, dim=1).item()

            attributions, _ = ig.attribute(X_input, target=target_class, return_convergence_delta=True, n_steps=20)
            attr_np = attributions.cpu().detach().numpy().squeeze()

            # Oś z samymi wartościami absolutnymi
            time_imp_abs = np.mean(np.abs(attr_np), axis=0)

            signed_time_imp = np.mean(attr_np, axis=0)

            total_gradient_strength = np.sum(np.abs(attr_np))
            net_positive_force = np.sum(signed_time_imp)

            all_samples_info.append({
                'strength_abs': total_gradient_strength,
                'strength_net': net_positive_force,
                'time_imp_abs': time_imp_abs,
                'signed_imp': signed_time_imp,
                'heatmap': np.abs(attr_np),
                'true_class': target_class,
                'pred_class': predicted_class,
                'id': processed + 1
            })

            visual_curve_magnitude = np.abs(signed_time_imp)

            time_length = len(visual_curve_magnitude)
            midpoint = time_length // 2

            max_first_half = np.max(visual_curve_magnitude[:midpoint])
            max_second_half = np.max(visual_curve_magnitude[midpoint:])

            global_max = max(max_first_half, max_second_half)
            smaller_max = min(max_first_half, max_second_half)

            if global_max > 1e-6:
                if smaller_max >= (0.5 * global_max):
                    dual_peak_samples.append({
                        'class': target_class,
                        'pred_class': predicted_class,
                        'signed_imp': signed_time_imp,
                        'idx_1': np.argmax(visual_curve_magnitude[:midpoint]),
                        'idx_2': midpoint + np.argmax(visual_curve_magnitude[midpoint:]),
                    })

            # Klasyfikacja do średniej ogólnej (bezwzględna) - aby zapisać bezwzględną chwilową uwagę sieci
            if target_class == 0:
                attr_class_0.append(np.abs(attr_np))
            else:
                attr_class_1.append(np.abs(attr_np))

            processed += 1
            # Test capowany na maksymalnej liczbie próbek. Można będzie zwiększyć przy dużym zbiorze

        if processed >= max_samples:
            break

    count_0 = len(attr_class_0)
    count_1 = len(attr_class_1)
    total_valid = count_0 + count_1

    if total_valid == 0: return

    # Oś Czasu
    n_time = attr_class_0[0].shape[1] if count_0 > 0 else attr_class_1[0].shape[1]
    n_channels = attr_class_0[0].shape[0] if count_0 > 0 else attr_class_1[0].shape[0]

    SFREQ = 160.0
    total_seconds = n_time / SFREQ
    time_sec_array = np.linspace(0, total_seconds, n_time)

    # WYKRESY =================================================================

    # Wykresy podwójnych profili uwagi ---------------------------------------------------------------------
    # Są to wykresy przebiegu klasyfikacji, w której potencjalnie można zauważyć skupienie sieci na dwóch różnych miejscach.
    # Jedno przed drugie po trzeciej sekundzie nagrania

    print("\n" + "=" * 60)
    print(f"Podwójne profile uwagi. Znaleziono {len(dual_peak_samples)} podwójnych skoków (przed i po 3. sekundzie)")
    print("=" * 60)

    limit_draw = 6  #graficzne przedstawienie 6 piewszych
    drawn = 0

    for anomaly in dual_peak_samples:
        if drawn >= limit_draw: break

        sig = anomaly['signed_imp']
        klasa = anomaly['class']
        klasa_pred = anomaly['pred_class']

        # Odczytywane wartości ze znakiem na szczytowych skokach
        idx1, idx2 = anomaly['idx_1'], anomaly['idx_2']
        val1_signed = sig[idx1]
        val2_signed = sig[idx2]

        plt.figure(figsize=(10, 4))

        plt.plot(time_sec_array, sig, color='black', linewidth=1)
        plt.fill_between(time_sec_array, sig, 0, where=(sig >= 0), color='green', alpha=0.5)
        plt.fill_between(time_sec_array, sig, 0, where=(sig < 0), color='red', alpha=0.5)
        plt.axhline(0, color='black', linestyle='--', linewidth=1)

        plt.scatter([time_sec_array[idx1]], [val1_signed], color='darkred' if val1_signed < 0 else 'darkgreen',
                    zorder=5, s=40, label=f'Peak A: {val1_signed:.3f}')
        plt.scatter([time_sec_array[idx2]], [val2_signed], color='darkred' if val2_signed < 0 else 'darkgreen',
                    zorder=5, s=40, label=f'Peak B: {val2_signed:.3f}')

        prawda = 'Prawa Ręka' if klasa == 1 else 'Lewa Ręka'
        wyrok = 'Klasyfikacja poprawna' if klasa == klasa_pred else 'Klasyfikacja BŁĘDNA!'

        plt.title(f"Uchwycony profil podwójny - {prawda} {wyrok}")
        plt.xlabel("Moment zdarzenia [Sekundy]")
        plt.ylabel("Siła gradientu sieci")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        drawn += 1

    # Wykresy Top 3 najlepiej ukierunkowanych próbek ---------------------------------------------------------------------------------
    # Są to wykresy próbek najprostszych dla sieci do sklasyfikowania. Czyli tych, które miały największy bias do jednej konkretnej klasy.
    # Nie oznacza to jednak, że sieć faktycznie poprawnie sklasyfikowała próbkę.

    print("\n" + "=" * 60)
    print("Top 3 próbek pod względem pewność absolutnej jednej z klas")
    print("=" * 60)

    # Sortowanie po absolutnej sumie kierunkowej. |sum(+ i -)|
    all_samples_info.sort(key=lambda x: abs(x['strength_net']), reverse=True)
    top_3_biased = all_samples_info[:3]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

    for rank, sample in enumerate(top_3_biased, 1):
        true_name = 'Prawa Ręka (1)' if sample['true_class'] == 1 else 'Lewa Ręka (0)'
        pred_name = 'Prawa Ręka (1)' if sample['pred_class'] == 1 else 'Lewa Ręka (0)'
        correct_text = "Poprawna" if sample['true_class'] == sample['pred_class'] else "BŁĘDNA"

        print(f"\nPróbka #{rank} | ID Próbki: {sample['id']} | Siła gradientu: {sample['strength_net']:.2f}")
        print(f"Wynik klasyfikacji: {pred_name} -> Klasyfikacja {correct_text}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

        sig = sample['signed_imp']
        ax1.plot(time_sec_array, sig, color='black', linewidth=1)
        ax1.fill_between(time_sec_array, sig, 0, where=(sig >= 0), color='limegreen', alpha=0.6,
                         label="Kierunek poprawny")
        ax1.fill_between(time_sec_array, sig, 0, where=(sig < 0), color='red', alpha=0.6, label="Kierunek błędny")
        ax1.axhline(0, color='black', linestyle='--', linewidth=1)

        ax1.set_title(f"#{rank} Rozkład gradientu sieci silnie ukierunkowanej (Próbka: {true_name})")
        ax1.set_xlabel("Moment zdarzenia [Sekundy]")
        ax1.set_ylabel("Siła gradientu sieci")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')

        # Heatmap
        im = ax2.imshow(sample['heatmap'], aspect='auto', cmap='hot', origin='lower',
                        extent=[0, total_seconds, 0, n_channels])
        ax2.set_title("Heatmapa")
        ax2.set_xlabel("Moment zdarzenia [Sekundy]")
        ax2.set_ylabel("Kanały (Elektrody)")
        plt.colorbar(im, ax=ax2)

        plt.tight_layout()
        plt.show()

    # Wykresy Top 3 próbki najbardziej nieukierunkowanych ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Top 3 próbek pod względem różnicy gradientów obu klas")
    print("=" * 60)

    # Obliczenia największych dwóch przeciwnych gradientów
    for s in all_samples_info:
        sig = s['signed_imp']
        # suma dodatnich aktywacji
        pos_sum = np.sum(sig[sig > 0]) if len(sig[sig > 0]) > 0 else 0
        # suma absolutna ujemnych aktywacji
        neg_sum = np.abs(np.sum(sig[sig < 0])) if len(sig[sig < 0]) > 0 else 0

        # Wynik to wartość słabszego z gradientów. Im wyższa wartość minimalna, tym większe były różnice
        s['conflict_score'] = min(pos_sum, neg_sum)

    # Ponowne sortowanie pod względem conflict_score
    all_samples_info.sort(key=lambda x: x['conflict_score'], reverse=True)
    top_3_conflicted = all_samples_info[:3]

    for rank, sample in enumerate(top_3_conflicted, 1):
        true_name = 'Prawa Ręka (1)' if sample['true_class'] == 1 else 'Lewa Ręka (0)'
        pred_name = 'Prawa Ręka (1)' if sample['pred_class'] == 1 else 'Lewa Ręka (0)'
        correct_text = "Poprawna" if sample['true_class'] == sample['pred_class'] else "BŁĘDNA"

        # Dodatkowe wyliczenie sił do analizy
        s_sig = sample['signed_imp']
        P_val = np.sum(s_sig[s_sig > 0])
        N_val = np.sum(s_sig[s_sig < 0])

        print(f"\nPróbka #{rank} | ID Próbki: {sample['id']}")
        print(f"Gradient ku klasie poprawnej {P_val:.2f} | Gradient ku klasie błędnej {N_val:.2f}")
        print(f"Wynik klasyfikacji: {pred_name} -> Klasyfikacja {correct_text}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

        sig = sample['signed_imp']
        ax1.plot(time_sec_array, sig, color='black', linewidth=1)
        ax1.fill_between(time_sec_array, sig, 0, where=(sig >= 0), color='green', alpha=0.5,
                         label='Kierunek poprawny')
        ax1.fill_between(time_sec_array, sig, 0, where=(sig < 0), color='red', alpha=0.5,
                         label='Kierunek błędny')
        ax1.axhline(0, color='black', linestyle='--', linewidth=1)

        ax1.set_title(f"#{rank} Rozkład gradientu sieci sieci silnie nieukierunkowanej (Próbka: {true_name})")
        ax1.set_xlabel("Moment zdarzenia [Sekundy]")
        ax1.set_ylabel("Siła gradientu sieci")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        im = ax2.imshow(sample['heatmap'], aspect='auto', cmap='hot', origin='lower',
                        extent=[0, total_seconds, 0, n_channels])
        ax2.set_title("Heatmapa")
        ax2.set_xlabel("Moment zdarzenia [Sekundy]")
        ax2.set_ylabel("Kanały (Elektrody)")
        plt.colorbar(im, ax=ax2)

        plt.tight_layout()
        plt.show()

# Trening ===================================================================================================

def main():
    DATA_PATH = "./preprocessed_data/Physionet"
    SEGMENT_TYPE = "6s"

    BATCH_SIZE = 32
    EPOCHS = 60
    LR = 0.0007
    D_MODEL = 128
    N_HEAD = 8

    print(f"Inicjalizacja WARIANT: {SEGMENT_TYPE}, BATCH={BATCH_SIZE}, LR={LR}, D_MODEL={D_MODEL}")

    # Dane -----------------------------------------------------------
    X_all, y_all = load_physionet_data(DATA_PATH, segment_type=SEGMENT_TYPE)
    if len(X_all) == 0:
        print("Błąd: Brak danych w", DATA_PATH)
        return


    n_channels = X_all.shape[1]
    n_timepoints = X_all.shape[2]
    print(f"Dane: {len(X_all)} epok. Kanały: {n_channels}, Czas: {n_timepoints}")
    print(f"Klasy: {np.unique(y_all, return_counts=True)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")

    # Walidacja 5-Fold ------------------------------------------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    best_global_acc = 0.0
    best_model_state = None
    best_test_loader_state = None

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        print(f"\n--- Fold {fold + 1} ---")

        train_ds = SimpleEEGDataset(X_all[train_idx], y_all[train_idx])
        test_ds = SimpleEEGDataset(X_all[test_idx], y_all[test_idx])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # Inicjalizacja Klasy modelu -----------------------------------------------
        model = TemporalTransformer(
            input_size=n_channels,
            d_model=D_MODEL,
            nhead=N_HEAD,
            num_classes=2,
            feature_method='raw'
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0

        # Wewnątrz jednego foldu wykonuje się podana pętla na N epok:
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            if test_acc > best_acc:
                best_acc = test_acc

            if (epoch + 1) % 5 == 0:
                print(f"Ep {epoch + 1:02d}: Loss={train_loss:.4f}, TrAcc={train_acc:.2%}, TsAcc={test_acc:.2%}")

        print(f"Best Test Acc Fold {fold + 1}: {best_acc:.2%}")
        fold_results.append(best_acc)

        # Rejestrowanie modelu dla Captum -------------------------------------------------
        if best_acc > best_global_acc:
            best_global_acc = best_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_test_loader_state = test_loader

    print(f"\nŚrednia ze wszystkich foldów: {np.mean(fold_results) * 100:.2f}%")

    # Zastosowanie Captum na najlepszym modelu ---------------------------------------------
    if best_model_state is not None:

        # Przy okazji zapis do pliku:
        SAVE_PATH = "./saved_model_states/temporal_transformer.pth"
        torch.save(best_model_state, SAVE_PATH)

        # Inicjalizacja wzorca modelu:
        best_model = TemporalTransformer(
            input_size=n_channels,
            d_model=D_MODEL,
            nhead=N_HEAD,
            num_classes=2,
            feature_method='raw'
        ).to(device)

        best_model.load_state_dict(best_model_state)

        # metoda do analizy i wykresów:
        analyze_with_captum_bulk2(best_model, best_test_loader_state, device)


if __name__ == "__main__":
    main()