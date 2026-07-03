import os
import numpy as np
import torch
import torch.nn as nn
import mne
import copy
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt

from model_classes import TemporalTransformer
import captum_analysis as captum
import data_loader as data
from scripts.model_testing.model_classes import SpatialTransformer


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


def train_cross_individual(subject_data, model_class, device, epochs=30, batch_size=32, lr=1e-3, d_model=128, nhead=8):
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
        num_time_points = X_train.shape[2]

        model_name = model_class.__name__
        if "Temporal" in model_name:
            # Temporal: Time is the sequence, so the model needs to embed Channels (electrodes)
            model_input_size = num_channels
            print(f"[INFO] Configuring {model_name} -> input_size = {model_input_size} (Channels)")

        elif "Spatial" in model_name:
            # Spatial: Channels are the sequence, so the model needs to embed time points
            model_input_size = num_time_points
            print(f"[INFO] Configuring {model_name} -> input_size = {model_input_size} (Time points)")

        model = model_class(
            input_size=model_input_size,
            d_model=d_model,
            nhead=nhead,
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


def get_model_disagreement_ids(results_a, results_b):
    """
    Finds and returns a list of sample IDs where the two models DISAGREE on the outcome
    (i.e., one model predicted correctly and the other failed).

    Args:
        results_a: Dictionary returned by compute_captum_analysis for Model A.
        results_b: Dictionary returned by compute_captum_analysis for Model B.

    Returns:
        list of int: Sample IDs of the disagreements.
    """
    if not results_a or not results_b:
        print("[WARNING] Missing one or both analysis results.")
        return []

    # Create dictionaries indexed by sample ID for O(1) fast matching
    dict_a = {s['id']: s for s in results_a['all_samples_info']}
    dict_b = {s['id']: s for s in results_b['all_samples_info']}

    disagreement_ids = []

    for s_id, sample_a in dict_a.items():
        if s_id in dict_b:
            sample_b = dict_b[s_id]

            is_correct_a = (sample_a['true_class'] == sample_a['pred_class'])
            is_correct_b = (sample_b['true_class'] == sample_b['pred_class'])

            # XOR condition: True if only exactly ONE of them is correct
            if is_correct_a != is_correct_b:
                disagreement_ids.append(s_id)

    if not disagreement_ids:
        print("[INFO] No disagreements found. Both models share identical successes and failures on these samples.")
    else:
        print(f"[INFO] Found {len(disagreement_ids)} sample(s) with conflicting outcomes.")

    return disagreement_ids

def visualise_raw_recordings():
    raw_file_path = "./data/Physionet/S001/S001R04.edf"

    # Loading Raw EDF
    raw = mne.io.read_raw_edf(raw_file_path, preload=True, verbose=False)

    # Extract Text Annotations raw.annotations.description should contain strings: 'T0', 'T1', 'T2'
    raw_text_annotations = raw.annotations.description
    t1_count = np.sum(raw_text_annotations == 'T1')
    t2_count = np.sum(raw_text_annotations == 'T2')

    print(f"\n[STEP 1] Direct EDF Text Annotation Analysis:")
    print(f"-> Found '{t1_count}' occurrences of text 'T1' (Left Hand).")
    print(f"-> Found '{t2_count}' occurrences of text 'T2' (Right Hand).")

    events, event_ids = mne.events_from_annotations(raw, verbose=False)

    # Testing mapping used in preprocessing:
    selected_event_id = {"left_hand": 2, "right_hand": 3}
    epochs = mne.Epochs(raw, events, event_id=selected_event_id, tmin=0, tmax=4, baseline=None, preload=True,
                        verbose=False)

    print(f"MNE Extracted {len(epochs)} target epochs.")
    print(f"-> Does {len(epochs)} equal the sum of T1({t1_count}) and T2({t2_count})? {'YES' if len(epochs) == (t1_count + t2_count) else 'NO'}")

    # Fetch Sampling Frequency
    sfreq = raw.info['sfreq']
    print(f"Sampling frequency: {sfreq} Hz")

    # Select specific motor cortex channels for visual clarity
    target_channels = ['F8..', 'F6..', 'F7..']
    picks = mne.pick_channels(raw.info['ch_names'], include=target_channels)

    # Get exactly the first 3 consecutive events from the dataset
    event_1 = raw.annotations[0]
    event_2 = raw.annotations[1]
    event_3 = raw.annotations[2]

    print(f"\nExtracted First 3 Events from EDF File:")
    print(f"Event 1: {event_1['description']} | Onset: {event_1['onset']}s | Duration: {event_1['duration']}s")
    print(f"Event 2: {event_2['description']} | Onset: {event_2['onset']}s | Duration: {event_2['duration']}s")
    print(f"Event 3: {event_3['description']} | Onset: {event_3['onset']}s | Duration: {event_3['duration']}s")

    def get_signal_slice(onset, duration):
        """Slices raw data based on exact index to prevent floating point rounding issues."""
        start_idx = int(round(onset * sfreq))
        stop_idx = start_idx + int(round(duration * sfreq))
        # raw.get_data returns shape: (n_channels, n_times)
        data, _ = raw[picks, start_idx:stop_idx]
        return data * 1e6  # Convert Volts to microVolts (uV)

    # Reconstructing signal from events
    chunk_1 = get_signal_slice(event_1['onset'], event_1['duration'])
    chunk_2 = get_signal_slice(event_2['onset'], event_2['duration'])
    chunk_3 = get_signal_slice(event_3['onset'], event_3['duration'])

    concatenated_signal = np.concatenate([chunk_1, chunk_2, chunk_3], axis=1)

    # Extract the continuous raw signal (The same length as the concatinated one)
    total_samples = concatenated_signal.shape[1]
    start_index = int(round(event_1['onset'] * sfreq)) #event_1['onset'] is the length of first event
    end_index = start_index + total_samples

    continuous_signal, _ = raw[picks, start_index:end_index]
    continuous_signal = continuous_signal * 1e6  # Convert to uV

    time_axis = np.arange(total_samples) / sfreq

    # Plots:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, sharey=True)
    channel_colors = ['blue', 'green', 'red']

    # Plot 1: Continuous Signal
    for i, (ch_name, color) in enumerate(zip(target_channels, channel_colors)):
        ax1.plot(time_axis, continuous_signal[i] + i * 50, label=ch_name, color=color, linewidth=1.5)

    ax1.set_title("True Continuous Raw Signal (First ~12 seconds)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Amplitude ($\mu$V)")
    ax1.legend(loc='upper right')

    # Add vertical lines to show the cuts
    t_transition_1 = event_1['duration']
    t_transition_2 = event_1['duration'] + event_2['duration']
    ax1.axvline(x=t_transition_1, color='black', linestyle='--', alpha=0.5)
    ax1.axvline(x=t_transition_2, color='black', linestyle='--', alpha=0.5)

    # Plot 2: Concatenated Signal
    for i, (ch_name, color) in enumerate(zip(target_channels, channel_colors)):
        ax2.plot(time_axis, concatenated_signal[i] + i * 50, label=ch_name, color=color, linewidth=1.2)

    ax2.set_title(
        f"Concatenated Signal Created from first 3 events ({event_1['description']} + {event_2['description']} + {event_3['description']} chunks)",
        fontsize=14, fontweight='bold')
    ax2.set_xlabel("Time (seconds)", fontsize=12)
    ax2.set_ylabel("Amplitude ($\mu$V)")
    ax2.legend(loc='upper right')

    ax2.axvline(x=t_transition_1, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(x=t_transition_2, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def main():
    # visualise_raw_recordings()
    # return

    TEST_PATIENT_SET_LEN = 15
    DATA_PATH = "./preprocessed_data/Physionet"
    DATA_PATH_BCI = "./preprocessed_data/BCI_IV_2a"

    # count_bci_samples("./preprocessed_data")

    MODEL_PATH = "./saved_model_states/temporal_transformer.pth"
    SEGMENT_TYPE = "6s"
    MODEL_PATH_TEMPORAL = f"./saved_model_states/temporal_transformer_{SEGMENT_TYPE}.pth"
    MODEL_PATH_SPATIAL = f"./saved_model_states/spatial_transformer_{SEGMENT_TYPE}.pth"

    MODEL_PATH_BCI = f"./saved_model_states/temporal_transformer_bci2a.pth"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {DEVICE}")

    # subject_data_bci,_, _ = data.load_bci_separate_patient_data(DATA_PATH_BCI)
    subject_data,time_array, t0_idx = data.load_physionet_separate_patient_data(DATA_PATH, segment_type=SEGMENT_TYPE)

    allegedly_best_patients = ["S014", "S055", "S059", "S063", "S085"]
    training_subject_data = data.split_dataset_return_training(
        subject_data=subject_data,
        test_patient_count=TEST_PATIENT_SET_LEN,
        batch_size=16,
        test_subjects_list=allegedly_best_patients
    )

    # ----------------------------------------------------------
    best_model, _ = train_cross_individual(training_subject_data, TemporalTransformer, DEVICE, 10, 16)
    # torch.save(best_model, f"./saved_model_states/temporal_transformer_2seconds_end_test_best_2.pth")

    # best_model_low_stats, _ = train_cross_individual(subject_data, TemporalTransformer, DEVICE, 5, 16, 0.0001, 32, 4)
    # torch.save(best_model_low_stats, f"./saved_model_states/temporal_transformer_{SEGMENT_TYPE}_low_stats.pth")

    # # # Train or Load -----------------------------------------
    best_model = torch.load(f"./saved_model_states/temporal_transformer_2seconds_end_test_best_2.pth", map_location=DEVICE)
    best_model.eval()
    # best_model_low_stats = torch.load(f"./saved_model_states/temporal_transformer_{SEGMENT_TYPE}_low_stats.pth", map_location=DEVICE)
    # best_model_low_stats.eval()
    #
    # best_model_low_stats.to(DEVICE)
    # best_model_low_stats.eval()
    #
    # best_model.to(DEVICE)
    # best_model.eval()
    # ----------------------------------------------------------

    # Explainability test:
    test_loader = torch.load("saved_model_states/global_data_loader/test_loader_batch_16.pth")
    results_high = captum.compute_captum_analysis(best_model, test_loader, DEVICE, sfreq=160.0)

    # captum.save_analysis_results(results_high, "./analysis_results2s_analysis_end.pkl")

    # results_low = captum.compute_captum_analysis(best_model_low_stats, test_loader, DEVICE, sfreq=160.0)

    # disputed_ids = get_model_disagreement_ids(results_high, results_low)

    # for id in disputed_ids:


    # captum.plot_top_biased(results, top_n=3)
    # captum.plot_top_conflicted(results, top_n=2)
    # captum.plot_dual_peaks(results, limit=4)

    epochs = mne.read_epochs(f"./preprocessed_data/Physionet/S001/PA001-6s-epo.fif", preload=False)
    ch_names = epochs.ch_names # List of electrode positions

    # Generating plots for EVERY sample in 5 patient test set 225 samples total
    # captum.generate_and_save_samples_in_range(
    #     analysis_results=results_high,
    #     start_id=1,
    #     end_id=230,
    #     fixed_scale=0.03,
    #     dynamic_dir="C:/Moje Pliki/POLITECHNIKA/Magisterka/2sBack/Dynamic",  # Ścieżka A
    #     fixed_dir="C:/Moje Pliki/POLITECHNIKA/Magisterka/2sBack/Fixed"  # Ścieżka B
    # )

    patient_ids = ["S014", "S055", "S059", "S063", "S085"]
    base_output_dir = r"C:\Moje Pliki\POLITECHNIKA\Magisterka\New HmAndBars with electrode names\last2s"

    for patient in patient_ids:
        print(f"\nDrawing heatmaps for patient: {patient}")

        patient_dir = os.path.join(base_output_dir, patient)
        os.makedirs(patient_dir, exist_ok=True)

        # File names:
        left_hand_path = os.path.join(patient_dir, f"{patient} Global Left.png")
        right_hand_path = os.path.join(patient_dir, f"{patient} Global Right.png")
        diff_path = os.path.join(patient_dir, f"{patient} Global Difference.png")

        # Left Hand
        heatmap_left = captum.extract_global_heatmap_data(
            results_high, mode='all', subject_id=patient, target_class=0)

        captum.plot_global_heatmap_and_bars(
            heatmap_left, results_high,
            title_suffix=f"Patient {patient} - Left Hand Only",
            channel_names=ch_names,
            save_path=left_hand_path,
            show_plot=False
        )

        # Right Hand
        heatmap_right = captum.extract_global_heatmap_data(
            results_high, mode='all', subject_id=patient, target_class=1)

        captum.plot_global_heatmap_and_bars(
            heatmap_right, results_high,
            title_suffix=f"Patient {patient} - Right Hand Only",
            channel_names=ch_names,
            save_path=right_hand_path,
            show_plot=False
        )

        # 3. Difference
        captum.plot_difference_heatmap_and_bars(
            heatmap_right - heatmap_left,
            analysis_results=results_high,
            title_suffix=f"Patient {patient} - Right Left Difference",
            channel_names=ch_names,
            save_path=diff_path,
            show_plot=False
        )

    # Total absolute network attention
    heatmap_all = captum.extract_global_heatmap_data(results_high, mode='all')
    captum.plot_global_heatmap_and_bars(
        heatmap_all, results_high,
        title_suffix="Global / Total Impact",
        channel_names=ch_names,
        save_path=base_output_dir + r"\Global.png",
        show_plot=False)

    # Attention pointing TOWARDS the correct classification
    heatmap_correct = captum.extract_global_heatmap_data(results_high, mode='correct_direction')
    captum.plot_global_heatmap_and_bars(
        heatmap_correct, results_high,
        title_suffix="Correct Class Support",
        channel_names=ch_names,
        save_path=base_output_dir + r"\Global Correct.png",
        show_plot=False)

    # Attention pointing AWAY from the correct classification (Conflict/Noise)
    heatmap_wrong = captum.extract_global_heatmap_data(results_high, mode='incorrect_direction')
    captum.plot_global_heatmap_and_bars(
        heatmap_wrong, results_high,
        title_suffix="Incorrect Class Influence (Noise/Error)",
        channel_names=ch_names,
        save_path=base_output_dir + r"\Global Incorrect.png",
        show_plot=False)


if __name__ == "__main__":
    main()