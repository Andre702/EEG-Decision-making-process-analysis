import os
import numpy as np
import torch
import torch.nn as nn
import mne
import copy
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset

from model_classes import TemporalTransformer
import captum_analysis as captum
import data_loader as data


def count_bci_samples(preprocessed_data_root: str):
    """
    Scans the directories for processed .fif files and counts
    the exact number of epochs per class and per subject.
    """
    base_dir = os.path.join(preprocessed_data_root, "BCI_III_3a")

    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    subjects = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    total_epochs_all = 0

    print("=" * 50)
    print("BCI III 3a - Data Verification")
    print("=" * 50)

    for subject in subjects:
        # Construct the expected filename, e.g., 'k3-epo.fif' or 'k3b-epo.fif'
        # Assuming the folder is 'k3b', subject[1:3] gives '3b'. We just read any .fif in folder.
        subject_folder = os.path.join(base_dir, subject)
        fif_files = [f for f in os.listdir(subject_folder) if f.endswith("-epo.fif")]

        if not fif_files:
            continue

        file_path = os.path.join(subject_folder, fif_files[0])

        try:
            epochs = mne.read_epochs(file_path, preload=True, verbose=False)

            # MNE używa zadeklarowanych KLUCZY słownika jako aliasów klas:
            left_count = len(epochs['left_hand']) if 'left_hand' in epochs.event_id else 0
            right_count = len(epochs['right_hand']) if 'right_hand' in epochs.event_id else 0
            total_subj = len(epochs)
            total_epochs_all += total_subj

            print(f"Subject {subject}: {total_subj} total epochs")
            print(f" -> Left Hand: {left_count} epochs")
            print(f" -> Right Hand: {right_count} epochs\n")

        except Exception as e:
            print(f"Failed to read data for {subject}: {e}")

    print("=" * 50)
    print(f"Total dataset size (all subjects): {total_epochs_all} epochs")
    print("=" * 50)

# Trening  =====================================================================================

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

def main():
    # "./preprocessed_data/Physionet"
    DATA_PATH = "./preprocessed_data/BCI_III_3a"

    count_bci_samples("./preprocessed_data")

    MODEL_PATH = "./saved_model_states/temporal_transformer.pth"

    SEGMENT_TYPE = "6s"

    D_MODEL = 128
    N_HEAD = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")

    # Inicjalizacja pustego modelu do wgrania gotowego
    X_all, y_all = data.load_physionet_data(DATA_PATH, segment_type=SEGMENT_TYPE)
    n_channels = X_all.shape[1]

    model = TemporalTransformer(
        input_size=n_channels,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_classes=2,
        feature_method='raw'
    )

    # Wczytanie modelu
    state_dict = torch.load(MODEL_PATH, map_location=device) # parametr location musi być ustawiony zgodnie z obecnie uruchamianym setupem
    model.load_state_dict(state_dict)

    # Model nałożony, wrzucamy go w tryb oceniania (bardzo ważne przed Captum)
    model.to(device)
    model.eval()
    print(f"Wgrany model z pliku: {MODEL_PATH}")

    # Dane testowe do klasyfikacji dla modelu
    test_ds = data.SimpleEEGDataset(X_all[:300], y_all[:300])
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Test wytłumaczalności:
    wyniki_badania = captum.compute_captum_analysis(model, test_loader, device, sfreq=160.0)

    captum.plot_top_biased(wyniki_badania, top_n=3)
    captum.plot_top_conflicted(wyniki_badania, top_n=2)
    captum.plot_dual_peaks(wyniki_badania, limit=4)
    # captum.analyze_bulk(model, test_loader, device, max_samples=60)

    # Total absolute network attention
    heatmap_all = captum.extract_global_heatmap_data(wyniki_badania, mode='all')
    captum.plot_global_heatmap_and_bars(heatmap_all, wyniki_badania, title_suffix="Global / Total Impact")

    # Attention pointing TOWARDS the correct classification
    heatmap_correct = captum.extract_global_heatmap_data(wyniki_badania, mode='correct_direction')
    captum.plot_global_heatmap_and_bars(heatmap_correct, wyniki_badania, title_suffix="Correct Class Support")

    # Attention pointing AWAY from the correct classification (Conflict/Noise)
    heatmap_wrong = captum.extract_global_heatmap_data(wyniki_badania, mode='incorrect_direction')
    captum.plot_global_heatmap_and_bars(heatmap_wrong, wyniki_badania, title_suffix="Incorrect Class Influence (Noise/Error)")

    # # Dane -----------------------------------------------------------
    # print(f"Dane: {SEGMENT_TYPE}, BATCH={BATCH_SIZE}, LR={LR}, D_MODEL={D_MODEL}")
    # X_all, y_all = load_physionet_data(DATA_PATH, segment_type=SEGMENT_TYPE)
    # if len(X_all) == 0:
    #     print("Błąd: Brak danych w", DATA_PATH)
    #     return
    #
    #
    # n_channels = X_all.shape[1]
    # n_timepoints = X_all.shape[2]
    # print(f"Dane: {len(X_all)} epok. Kanały: {n_channels}, Czas: {n_timepoints}")
    # print(f"Klasy: {np.unique(y_all, return_counts=True)}")
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Urządzenie: {device}")
    #
    # # Walidacja 5-Fold ------------------------------------------------------------
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # fold_results = []
    #
    # best_global_acc = 0.0
    # best_model_state = None
    # best_test_loader_state = None
    #
    # for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
    #     print(f"\n--- Fold {fold + 1} ---")
    #
    #     train_ds = SimpleEEGDataset(X_all[train_idx], y_all[train_idx])
    #     test_ds = SimpleEEGDataset(X_all[test_idx], y_all[test_idx])
    #
    #     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    #     test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    #
    #     # Inicjalizacja Klasy modelu -----------------------------------------------
    #     model = TemporalTransformer(
    #         input_size=n_channels,
    #         d_model=D_MODEL,
    #         nhead=N_HEAD,
    #         num_classes=2,
    #         feature_method='raw'
    #     ).to(device)
    #
    #     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    #     criterion = nn.CrossEntropyLoss()
    #
    #     best_acc = 0.0
    #
    #     # Wewnątrz jednego foldu wykonuje się podana pętla na N epok:
    #     for epoch in range(EPOCHS):
    #         train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    #         test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    #
    #         if test_acc > best_acc:
    #             best_acc = test_acc
    #
    #         if (epoch + 1) % 5 == 0:
    #             print(f"Ep {epoch + 1:02d}: Loss={train_loss:.4f}, TrAcc={train_acc:.2%}, TsAcc={test_acc:.2%}")
    #
    #     print(f"Best Test Acc Fold {fold + 1}: {best_acc:.2%}")
    #     fold_results.append(best_acc)
    #
    #     # Rejestrowanie modelu dla Captum -------------------------------------------------
    #     if best_acc > best_global_acc:
    #         best_global_acc = best_acc
    #         best_model_state = copy.deepcopy(model.state_dict())
    #         best_test_loader_state = test_loader
    #
    # print(f"\nŚrednia ze wszystkich foldów: {np.mean(fold_results) * 100:.2f}%")
    #
    # # Zastosowanie Captum na najlepszym modelu ---------------------------------------------
    # if best_model_state is not None:
    #
    #     # Przy okazji zapis do pliku:
    #     SAVE_PATH = "./saved_model_states/temporal_transformer.pth"
    #     torch.save(best_model_state, SAVE_PATH)
    #
    #     # Inicjalizacja wzorca modelu:
    #     best_model = TemporalTransformer(
    #         input_size=n_channels,
    #         d_model=D_MODEL,
    #         nhead=N_HEAD,
    #         num_classes=2,
    #         feature_method='raw'
    #     ).to(device)
    #
    #     best_model.load_state_dict(best_model_state)
    #
    #     # metoda do analizy i wykresów:
    #     captum.analyze_bulk(best_model, best_test_loader_state, device)


def train_and_save_model(model_class_instantiated, data_dir: str, save_path: str, device: str):
    """
    Full pipeline to load data, train the transformer model,
    and save the best performing weights to disk.
    """
    print("\n" + "=" * 50)
    print("Initiating BCI III 3a Training Pipeline")
    print("=" * 50)

    # 1. Load Data
    X_all, y_all = data.load_bci3a_data(data_dir)

    # 2. Split into Train and Validation sets (80% / 20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Validation set size: {X_val.shape[0]} samples")

    # 3. Convert to PyTorch Tensors and DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    batch_size = 16  # Adjust based on your GPU VRAM, 16 is safe for small EEG datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Model Setup
    model = model_class_instantiated.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    # AdamW is recommended for Transformers due to decoupled weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    num_epochs = 30
    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())

    # 5. Training Loop
    print("\nStarting Training Phase...")

    for epoch in range(1, num_epochs + 1):

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Save model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            is_best = " [BEST SAVED]"
        else:
            is_best = ""

        print(f"Epoch [{epoch}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}{is_best}")

    # 6. Save the ultimate best model to disk
    print("\nTraining completed.")
    torch.save(best_model_weights, save_path)
    print(f"Best model weights saved to {save_path} (Validation Accuracy: {best_val_acc:.4f})")

    # We return loaders if you want to immediately pass them to your Captum functions
    return train_loader, val_loader


if __name__ == "__main__":
    main()