import os, mne, shutil
import numpy as np
from eeg_logger import logger

def extract_epochs_giga(data_path: str, save_path_root: str, resample_to: int = None) -> None:
    if not os.path.exists(data_path):
        logger.error(f"No data to preprocess in {data_path}")
        return

    # Tworzenie katalogu
    save_directory: str = __create_save_directory_giga(save_path_root)

    subject_files = sorted([f for f in os.listdir(data_path) if f.endswith(".edf")])

    for file_name in subject_files:
        subject_id = os.path.splitext(file_name)[0]

        try:
            subject_num = int(subject_id[1:])
            standardized_subject = f"S{subject_num:03d}"
        except ValueError:
            standardized_subject = subject_id.upper()

        data_file = os.path.join(data_path, file_name)
        logger.info(f"Reading data from {file_name} (mapped to {standardized_subject})...")

        raw = mne.io.read_raw_edf(data_file, preload=True)

        # Resampling uruchomi się tylko, gdy parametr nie jest None
        if resample_to is not None:
            if raw.info['sfreq'] != resample_to:
                raw.resample(resample_to)
                logger.info(f"Resampled to {resample_to} Hz")
        else:
            logger.info(f"Keeping native sampling rate: {raw.info['sfreq']} Hz")

        epochs_3s, epochs_4s = __extract_giga(raw)

        subject_save_dir = os.path.join(save_directory, standardized_subject)
        os.makedirs(subject_save_dir, exist_ok=True)

        epochs_3s_filename = os.path.join(subject_save_dir, f"PA{standardized_subject[1:4]}-3s-epo.fif")
        epochs_4s_filename = os.path.join(subject_save_dir, f"PA{standardized_subject[1:4]}-4s-epo.fif")

        epochs_3s.save(epochs_3s_filename, overwrite=True)
        epochs_4s.save(epochs_4s_filename, overwrite=True)

        logger.info(f"Preprocessed data for subject {standardized_subject} saved")


def __extract_giga(raw_data: mne.io.BaseRaw) -> tuple[mne.Epochs, mne.Epochs]:
    events, event_ids = mne.events_from_annotations(raw_data)
    logger.info(f"Original event ids: {event_ids}")

    # Mapowanie zdarzeń - szukamy kluczy zwierających "left" / "right" lub "1" / "2"
    selected_event_id = {}

    left_key = [k for k in event_ids.keys() if 'left' in k.lower() or k == '1']
    right_key = [k for k in event_ids.keys() if 'right' in k.lower() or k == '2']

    if left_key and right_key:
        selected_event_id["left_hand"] = event_ids[left_key[0]]
        selected_event_id["right_hand"] = event_ids[right_key[0]]
    else:
        # W razie nietypowych nazw stosujemy bezpieczny fallback
        logger.warning("Could not auto-detect left/right keys. Falling back to default codes.")
        if '1' in event_ids and '2' in event_ids:
            selected_event_id["left_hand"] = event_ids['1']
            selected_event_id["right_hand"] = event_ids['2']
        else:
            raise ValueError(f"Unknown event keys in EDF file: {event_ids.keys()}")

    logger.info(f"Using mapped event IDs: {selected_event_id}")

    # Filtrowanie tylko sygnałów EEG (wykluczamy ewentualne kanały EMG/Stim)
    picks = mne.pick_types(raw_data.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads")

    # Okna czasowe (dla wyobrażeń ruchowych w GigaDB początek ruchu to t=0 po zaprezentowaniu cue)
    tmin_3s, tmax_3s = 1, 3
    tmin_4s, tmax_4s = 0, 3

    epochs_3s = mne.Epochs(
        raw_data,
        events,
        event_id=selected_event_id,
        tmin=tmin_3s,
        tmax=tmax_3s,
        picks=picks,
        baseline=None,
        preload=True,
    )

    epochs_4s = mne.Epochs(
        raw_data,
        events,
        event_id=selected_event_id,
        tmin=tmin_4s,
        tmax=tmax_4s,
        picks=picks,
        baseline=None,
        preload=True,
    )

    epochs_normalised_3s = __normalise(epochs_3s)
    epochs_normalised_4s = __normalise(epochs_4s)

    logger.info(f"Extracted {len(epochs_normalised_3s)} epochs (3s) and {len(epochs_normalised_4s)} epochs (4s)")
    return epochs_normalised_3s, epochs_normalised_4s

def __normalise(epochs: mne.Epochs) -> mne.epochs:
    """
    Applies z-score normalisation according to this formula:
    X* = (X - mean) / std + aN
    """

    data: np.ndarray = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    mean = data.mean(axis=2, keepdims=True)
    std = data.std(axis=2, keepdims=True)
    std[std == 0] = 1.0
    N = np.random.randn(*data.shape)
    a = 0.01

    zscored_data = (data - mean) / std + a * N
    epochs._data = zscored_data

    return epochs


def __create_save_directory_giga(save_path_root: str) -> str:
    path: str = f"{save_path_root}/GigaDB"

    if os.path.exists(path):
        logger.info("Removing old preprocess directory for GigaDB")
        shutil.rmtree(path)

    os.makedirs(path)
    return path

