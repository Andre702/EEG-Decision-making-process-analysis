import os
import numpy as np
import torch
import mne
import matplotlib.pyplot as plt
import scipy.io as sio
import re

import tensorboard # needed for Logging?
import edfio # needed for EDF Export

from model_classes import TemporalTransformer
import captum_analysis as captum
import data_loader as data
import training_methods as train

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


def convert_mat_to_annotated_edf(mat_file_path, output_dir):
    """
    Konwertuje pliki sXX.mat z zestawu GigaDB na ustrukturyzowane pliki .edf.
    Skaluje prawidłowo jednostki EEG unikając błędu limitu 8-znakowego w EDF,
    filtruje zepsute próbki oraz zachowuje poprawną oś czasu dla późniejszych operacji MNE.
    """
    try:
        mat = sio.loadmat(mat_file_path, simplify_cells=True)
    except Exception as e:
        raise IOError(f"Nie udało się otworzyć pliku {mat_file_path}: {e}")

    if 'eeg' not in mat:
        raise KeyError(f"Oczekiwano klucza 'eeg' w pliku .mat.")
    eeg = mat['eeg']

    # 1. Pobranie parametrów czasowych
    fs = float(eeg.get('srate', 512.0))
    t_start_ms = float(eeg['frame'][0])  # zazwyczaj -2000.0 ms
    t_end_ms = float(eeg['frame'][1])  # zazwyczaj 5000.0 ms

    duration_sec = (t_end_ms - t_start_ms) / 1000.0  # 7.0s
    n_samples_per_trial = int(round(duration_sec * fs))
    time_to_event_sec = abs(t_start_ms) / 1000.0  # Zazwyczaj zdarzenie jest w 2.0s próbki

    # 2. Pobieramy tylko 64 kanały (bez potencjalnego EXG na końcu)
    raw_left = np.array(eeg['imagery_left'], dtype=np.float32)
    raw_right = np.array(eeg['imagery_right'], dtype=np.float32)

    n_channels = min(64, raw_left.shape[0])
    n_trials_left = raw_left.shape[1] // n_samples_per_trial
    n_trials_right = raw_right.shape[1] // n_samples_per_trial

    # Rozbicie ze sklejonych na wymiary: (kanały, próby, czas), a potem (próby, kanały, czas)
    img_left = raw_left[:n_channels, :].reshape(n_channels, n_trials_left, n_samples_per_trial)
    img_left = np.transpose(img_left, (1, 0, 2))

    img_right = raw_right[:n_channels, :].reshape(n_channels, n_trials_right, n_samples_per_trial)
    img_right = np.transpose(img_right, (1, 0, 2))

    # 3. Obsługa uszkodzonych próbek (Bad Trials) - struktura ma listy array'ów
    bt_dict = eeg.get('bad_trial_indeces', {})

    def extract_bad_indices(class_idx):
        bads = []
        for k in ['bad_trial_idx_mi', 'bad_trial_idx_voltage']:
            if k in bt_dict and isinstance(bt_dict[k], (list, tuple, np.ndarray)) and len(bt_dict[k]) > class_idx:
                arr = bt_dict[k][class_idx]
                # Sprawdzamy czy to nparray z jakimiś danymi (np. jeśli puste, size = 0)
                if isinstance(arr, np.ndarray) and arr.size > 0:
                    bads.extend(arr.tolist())
        # Przeliczamy indeksowanie z MATLAB na Python (od 0) i tworzymy unikalny zbiór
        return list(set([int(x) - 1 for x in bads]))

    # Zakładamy klasyczny podział bad_trials w liście [Left, Right]
    bad_trials_left = extract_bad_indices(0)
    bad_trials_right = extract_bad_indices(1)

    if bad_trials_left:
        img_left = np.delete(img_left, bad_trials_left, axis=0)
    if bad_trials_right:
        img_right = np.delete(img_right, bad_trials_right, axis=0)

    # 4. Łączymy "czyste" bloki (nie stosujemy tutaj maskowania/obcinania długości, dajemy całe 7 sekund!)
    X = np.concatenate([img_left, img_right], axis=0)

    y = np.concatenate([
        np.ones(img_left.shape[0], dtype=int),
        np.ones(img_right.shape[0], dtype=int) * 2
    ])

    # KRYTYCZNE ZABEZPIECZENIE (ROZWIĄZUJE VALUE_ERROR Z ZAKRESEM)
    # Surowe ADC/mikrowolty konwertujemy na Volty używając 1e-6 (format MNE natywny)
    X_volts = X * 1e-6

    # 5. Tworzenie nazw elektrod i struktury do RawArray
    biosemi_64 = [
        'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
        'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
        'FPz', 'FP2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
        'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'
    ]

    concatenated_data = X_volts.transpose(1, 0, 2).reshape(n_channels, -1)
    new_info = mne.create_info(ch_names=biosemi_64[:n_channels], sfreq=fs, ch_types='eeg')
    concatenated_raw = mne.io.RawArray(concatenated_data, new_info, verbose=False)

    # 6. Tworzenie i przypisywanie adnotacji
    annot_onsets = []
    annot_durations = []
    annot_descriptions = []

    epoch_duration = duration_sec  # Skok co równe 7 sekund
    n_epochs = X_volts.shape[0]

    for i in range(n_epochs):
        # Umieszczamy punkt sygnału "Onset" z przesunięciem (najpewniej po 2-sekundowej rozbiegówce epoki)
        # Dzięki temu tmin=1 i tmax=3 ze starego kodu uderzy we właściwy moment sklejonej struktury raw!
        annot_onsets.append(i * epoch_duration + time_to_event_sec)
        annot_durations.append(0.0)  # Wydarzenie chwilowe, z niego MNE odczyta '1 do 3 sekund do przodu'
        desc = 'imagery_left_hand' if y[i] == 1 else 'imagery_right_hand'
        annot_descriptions.append(desc)

    annotations = mne.Annotations(
        onset=annot_onsets,
        duration=annot_durations,
        description=annot_descriptions
    )
    concatenated_raw.set_annotations(annotations)

    # 7. Zapis pliku EDF i obsługa nazwy
    base_name = os.path.basename(mat_file_path)
    match = re.match(r"^[sS](\d+)", base_name)
    subject_id = f"S{int(match.group(1)):03d}" if match else "S000"

    subject_dir = os.path.join(output_dir, subject_id)
    os.makedirs(subject_dir, exist_ok=True)
    edf_file_path = os.path.join(subject_dir, f"{subject_id}.edf")

    mne.export.export_raw(edf_file_path, concatenated_raw, fmt='edf', overwrite=True, verbose=False)
    print(f"Pomyślnie przetworzono i zapisano: {edf_file_path}")


def main():
    # INPUT_FOLDER = "./data/GigaSource"
    # OUTPUT_FOLDER = "./preprocessed_data"
    #
    # all_mat_files = glob.glob(os.path.join(INPUT_FOLDER, "*.mat"))
    #
    # # Filtrujemy pliki przy użyciu wyrażeń regularnych
    # # Akceptujemy tylko nazwy typu sXX.mat (np. s01.mat, s52.mat)
    # # Wykluczamy pliki z tekstem typu s01_trial_sequence_v1.mat
    # eeg_files = []
    # for f in all_mat_files:
    #     name = os.path.basename(f)
    #     if re.match(r"^[sS]\d+\.mat$", name):
    #         eeg_files.append(f)
    #
    # # Sortujemy, aby pliki przetwarzały się po kolei (s01, s02...)
    # eeg_files = sorted(eeg_files)
    # total_files = len(eeg_files)
    #
    # print(f"Znaleziono {total_files} właściwych plików sygnałowych (np. s01.mat).")
    #
    # if total_files == 0:
    #     print(f"Ostrzeżenie: Nie znaleziono plików dopasowanych do wzorca 'sXX.mat' w folderze '{INPUT_FOLDER}'.")
    #     print("Upewnij się, że ścieżka wejściowa jest poprawna oraz pliki nie są rozpakowane do głębszych podfolderów.")
    #
    # successful_conversions = 0
    #
    # for idx, mat_path in enumerate(eeg_files, start=1):
    #     filename = os.path.basename(mat_path)
    #     print(f"\n[{idx}/{total_files}] Przetwarzanie: {filename}")
    #
    #     try:
    #         convert_mat_to_annotated_edf(
    #             mat_file_path=mat_path,
    #             output_dir=OUTPUT_FOLDER
    #         )
    #         successful_conversions += 1
    #     except Exception as e:
    #         print(f"Błąd podczas konwersji pliku {filename}: {e}")
    #
    # print("\n--- Podsumowanie procesu ---")
    # print(f"Pomyślnie przetworzono: {successful_conversions}/{total_files} plików.")
    # print(f"Dane wyjściowe zostały zapisane w strukturze folderów w: {os.path.abspath(OUTPUT_FOLDER)}")
    # return

    # visualise_raw_recordings()
    # return

    # test_file = "./data/Giga/s01.edf"
    #
    # raw = mne.io.read_raw_edf(test_file, preload=False)
    # print("--- KANAŁY ---")
    # print(raw.ch_names[:10], "... Razem kanałów:", len(raw.ch_names))
    #
    # print("\n--- ANOTACJE ---")
    # print(raw.annotations)
    #
    # try:
    #     events, event_id = mne.events_from_annotations(raw)
    #     print("\n--- WYKRYTE ZDARZENIA I ICH ID ---")
    #     print(event_id)
    # except Exception as e:
    #     print("Nie udało się odczytać zdarzeń z anotacji:", e)
    #
    # return

    TEST_PATIENT_SET_LEN = 7
    DATA_PATH = "./preprocessed_data/GigaDB"
    SEGMENT_TYPE = "3s"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {DEVICE}")

    subject_data,time_array, t0_idx = data.load_physionet_separate_patient_data(DATA_PATH, segment_type=SEGMENT_TYPE)

    allegedly_best_patients = ["S014", "S055", "S059", "S063", "S085"]
    training_subject_data = data.split_dataset_return_training(
        subject_data=subject_data,
        test_patient_count=TEST_PATIENT_SET_LEN,
        batch_size=16,
        # test_subjects_list=allegedly_best_patients
    )

    # ----------------------------------------------------------
    best_model, _ = train.train_cross_individual_torch(training_subject_data, TemporalTransformer, 10, 16)
    torch.save(best_model, f"./saved_model_states/temporal_transformer_2seconds_end_test_best_2.pth")

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