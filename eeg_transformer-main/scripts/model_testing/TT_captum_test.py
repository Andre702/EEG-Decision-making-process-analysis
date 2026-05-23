import os
import numpy as np
import torch
import torch.nn as nn
import mne
import copy
from sklearn.model_selection import KFold, StratifiedKFold
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

# Training  =====================================================================================

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

        total_loss += loss.item() * X.size(0)
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

def train_and_save_model_5fold(X_all, y_all, device, save_path="best_model.pth",
                               epochs=30, batch_size=8, lr=1e-4, d_model=32, n_head=4):
    """
    Trains Temporal Transformer using 5-Fold Cross Validation.
    Finds the best model and saves it.
    Returns the best model and a testing group for captum analysis.
    """

    # Data shape X_all (N, channels, time)
    n_channels = X_all.shape[1]

    print("=" * 60)
    print(f"Training starting (5-Fold CV)")
    print(f"Data: {len(X_all)} samples, {n_channels} channels, D_MODEL: {d_model}")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    best_global_acc = 0.0
    best_model_state = None
    best_test_loader_state = None

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
        print(f"\n--- Fold {fold + 1}/5 ---")

        # Creating collections for current fold:
        train_ds = data.SimpleEEGDataset(X_all[train_idx], y_all[train_idx])
        test_ds = data.SimpleEEGDataset(X_all[test_idx], y_all[test_idx]) #test_idx?

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # In every fold initialize an empty model
        model = TemporalTransformer(
            input_size=n_channels,
            d_model=d_model,
            nhead=n_head,
            num_classes=2,
            feature_method='raw'
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()

        best_fold_acc = 0.0
        best_fold_weights = None

        # Training loop for this fold:
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            # Track the best accuracy in the fold
            if test_acc > best_fold_acc:
                best_fold_acc = test_acc
                # Save deep copy
                best_fold_weights = copy.deepcopy(model.state_dict())
                is_best = "*"
            else:
                is_best = ""

            if epoch % 5 == 0 or is_best == "*":
                print(
                    f"  Ep {epoch:02d}/{epochs}: TrLoss: {train_loss:.4f}, TrAcc: {train_acc:.2%} | ValAcc: {test_acc:.2%} {is_best}")

        print(f"-> Best in fold {fold + 1}: {best_fold_acc:.2%}")
        fold_results.append(best_fold_acc)

        # Comparing best acc in fold with previous folds accuracy
        if best_fold_acc > best_global_acc:
            best_global_acc = best_fold_acc
            best_model_state = copy.deepcopy(best_fold_weights)
            best_test_loader_state = test_loader

        # Cleanup on memory
        del model
        del optimizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # End
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)

    print("\n" + "=" * 60)
    print("Training finished (5-FOLD CV)")
    print(f"Results for folds: {[f'{a:.2%}' for a in fold_results]}")
    print(f"average accuracy: {mean_acc:.2%} ± {std_acc:.2%}")
    print(f"Highest test-set accuracy: {best_global_acc:.2%}")
    print("=" * 60)

    # Recreating the best model state:
    final_model = TemporalTransformer(
        input_size=n_channels,
        d_model=d_model,
        nhead=n_head,
        num_classes=2,
        feature_method='raw'
    ).to(device)
    final_model.load_state_dict(best_model_state)

    # --- Zapisywanie na dysk ---
    torch.save(best_model_state, save_path)
    print(f"\n[SUKCES] Wagi najlepszego modelu zapisano pomyślnie w pliku: {save_path}")

    return final_model, best_test_loader_state


def train_cross_individual(subject_data, model_class, device, epochs=30, batch_size=32, lr=1e-3):
    """
    Performs a 5-fold cross-individual validation.
    Tracks and returns the best performing model across all folds, along with its specific test_loader.
    """
    all_subjects = np.array(list(subject_data.keys()))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_global_acc = 0.0
    best_global_model = None
    best_global_test_loader = None
    fold_accuracies = []

    print(f"\n--- Starting 5-Fold Cross-Individual Validation (Device: {device}) ---")

    for fold, (train_idx, test_idx) in enumerate(kf.split(all_subjects)):
        print(f"Fold {fold + 1}/5")

        train_subjects = all_subjects[train_idx]
        test_subjects = all_subjects[test_idx]

        # Gather training data for current fold
        X_train_list, y_train_list = [], []
        for subj in train_subjects:
            X, y = subject_data[subj]
            X_train_list.append(X)
            y_train_list.append(y)

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        # Gather testing data for current fold
        X_test_list, y_test_list = [], []
        for subj in test_subjects:
            X, y = subject_data[subj]
            X_test_list.append(X)
            y_test_list.append(y)

        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)

        # Create Datasets and DataLoaders
        train_dataset = data.SimpleEEGDataset(X_train, y_train, is_train=True)
        test_dataset = data.SimpleEEGDataset(X_test, y_test, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Input size for the Model
        num_channels = X_train.shape[1] # For temporal transformer forward is applied as squeeze(1).permute(2, 0, 1), so the layer expects Channels as input size

        model = model_class(
            input_size=num_channels,
            d_model=64,
            nhead=8,
            num_classes=2,
            feature_method='raw'
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        best_fold_acc = 0.0
        best_fold_state = None
        total_batches = len(train_loader)

        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(x_batch)

                loss = criterion(outputs, y_batch)
                loss.backward()

                # added gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * x_batch.size(0)
                preds = outputs.argmax(dim=1)
                train_correct += (preds == y_batch).sum().item()
                train_total += y_batch.size(0)

                print(f"Batch [{batch_idx + 1:03d}/{total_batches}]", flush=True)

            # Evaluation for current epoch
            model.eval()
            test_correct, test_total = 0, 0
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    outputs = model(x_batch)
                    preds = outputs.argmax(dim=1)
                    test_correct += (preds == y_batch).sum().item()
                    test_total += y_batch.size(0)

            train_acc = train_correct / train_total
            test_acc = test_correct / test_total

            # Save the best model state inside the current fold
            if test_acc > best_fold_acc:
                best_fold_acc = test_acc
                best_fold_state = copy.deepcopy(model.state_dict())

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1:02d}/{epochs}] "
                      f"Loss: {train_loss / train_total:.4f} "
                      f"Train Acc: {train_acc * 100:.2f}% "
                      f"Test Acc: {test_acc * 100:.2f}%", flush=True)

        print(f"\nBest Test Accuracy for Fold {fold + 1}: {best_fold_acc * 100:.2f}%")
        fold_accuracies.append(best_fold_acc)

        # Check if model from this fold beat best model, if so assign it as best
        if best_fold_acc > best_global_acc:
            best_global_acc = best_fold_acc

            # Reconstruct the best model
            model.load_state_dict(best_fold_state)
            best_global_model = copy.deepcopy(model)
            best_global_test_loader = test_loader

    avg_accuracy = np.mean(fold_accuracies)
    print(f"\n--- Training finished (5-FOLD Cross-Individual) ---")
    print(f"Average 5-Fold Accuracy: {avg_accuracy * 100:.2f}%")
    print(f"Best Global Fold Accuracy: {best_global_acc * 100:.2f}%")

    # Returning the best model and its specific test_loader for further analysis
    return best_global_model, best_global_test_loader


def main():
    DATA_PATH = "./preprocessed_data/Physionet"

    # count_bci_samples("./preprocessed_data")

    MODEL_PATH = "./saved_model_states/temporal_transformer.pth"
    MODEL_PATH_EXTENDED = "./saved_model_states/temporal_transformer_6s_pro.pth"
    SEGMENT_TYPE = "6s"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {DEVICE}")


    # X_all, y_all = data.load_physionet_data(DATA_PATH, segment_type=SEGMENT_TYPE)

    subject_data,time_array, t0_idx = data.load_physionet_separate_patient_data(DATA_PATH, segment_type=SEGMENT_TYPE)
    best_model, best_test_loader = train_cross_individual(subject_data, TemporalTransformer, DEVICE, 10, 16)
    torch.save(best_model, MODEL_PATH_EXTENDED)

    #     = train_and_save_model_5fold(
    #     X_all=X_all,
    #     y_all=y_all,
    #     device=DEVICE,
    #     save_path=MODEL_PATH_EXTENDED,
    #     epochs=10,
    #     batch_size=32,
    #     lr=0.0001,
    #     d_model=64,
    #     n_head=8
    # )

    best_model.to(DEVICE)
    best_model.eval()

    # Test wytłumaczalności:
    results = captum.compute_captum_analysis(best_model, best_test_loader, DEVICE, sfreq=160.0)

    captum.plot_top_biased(results, top_n=3)
    captum.plot_top_conflicted(results, top_n=2)
    captum.plot_dual_peaks(results, limit=4)
    # captum.analyze_bulk(model, test_loader, device, max_samples=60)

    # Total absolute network attention
    heatmap_all = captum.extract_global_heatmap_data(results, mode='all')
    captum.plot_global_heatmap_and_bars(heatmap_all, results, title_suffix="Global / Total Impact")

    # Attention pointing TOWARDS the correct classification
    heatmap_correct = captum.extract_global_heatmap_data(results, mode='correct_direction')
    captum.plot_global_heatmap_and_bars(heatmap_correct, results, title_suffix="Correct Class Support")

    # Attention pointing AWAY from the correct classification (Conflict/Noise)
    heatmap_wrong = captum.extract_global_heatmap_data(results, mode='incorrect_direction')
    captum.plot_global_heatmap_and_bars(heatmap_wrong, results, title_suffix="Incorrect Class Influence (Noise/Error)")


def load_eeg_dataset(data_dir: str, dataset_name: str = "physionet", segment_type: str = "6s") -> tuple[
    np.ndarray, np.ndarray]:
    """
    Unified function to load and normalize preprocessed MNE epochs from different datasets.

    Parameters:
    - data_dir: Root directory containing subject folders (e.g., 'S001', 'S1').
    - dataset_name: Either 'physionet' or 'bci3a'.
    - segment_type: Only used for physionet ('3s' or '6s').

    Returns:
    - X (np.ndarray): Scaled EEG features (samples, channels, time).
    - y (np.ndarray): Binary labels (0 for left hand, 1 for right hand).
    """

    if dataset_name not in ["physionet", "bci3a"]:
        raise ValueError("Invalid dataset_name. Use 'physionet' or 'bci3a'.")

    all_x, all_y = [], []
    subjects = [f for f in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, f))]

    print(f"Loading {dataset_name.upper()} data from {data_dir}...")

    for subj_folder in subjects:
        subj_path = os.path.join(data_dir, subj_folder)

        # --- DATASET SPECIFIC CONFIGURATION ---
        if dataset_name == "physionet":
            # Folder S001 generates PA001-6s-epo.fif
            subject_id = subj_folder[1:4]
            expected_filename = f"PA{subject_id}-{segment_type}-epo.fif"
            left_id, right_id = 2, 3

        elif dataset_name == "bci3a":
            # Folder S1 generates 1-epo.fif
            subject_id = subj_folder[1:3]
            expected_filename = f"{subject_id}-epo.fif"
            left_id, right_id = 3, 4

        file_path = os.path.join(subj_path, expected_filename)

        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping subject {subj_folder}.")
            continue

        try:
            epochs = mne.read_epochs(file_path, preload=True, verbose=False)

            # Filter out any unintended events keeping only left and right hand
            target_events = [left_id, right_id]
            epochs = epochs[np.isin(epochs.events[:, -1], target_events)]

            X = epochs.get_data(copy=True)
            y = epochs.events[:, -1]

            # Binary classification mapping: Left Hand -> 0, Right Hand -> 1
            y = np.where(y == left_id, 0, 1)

            # Normalization: per-channel z-score across time axis
            mean = np.mean(X, axis=2, keepdims=True)
            std = np.std(X, axis=2, keepdims=True)
            std[std == 0] = 1.0
            X = (X - mean) / std

            all_x.append(X)
            all_y.append(y)

        except Exception as e:
            print(f"Skipped {subj_folder} due to error: {e}")

    if not all_x:
        raise ValueError(f"No valid .fif files loaded for dataset '{dataset_name}'.")

    # Concatenate all subjects and cast to standard neural network types
    final_x = np.concatenate(all_x).astype(np.float32)
    final_y = np.concatenate(all_y).astype(np.int64)

    print(f"Successfully loaded. Total shape: X={final_x.shape}, y={final_y.shape}")
    return final_x, final_y

if __name__ == "__main__":
    main()