import os, mne, shutil
import numpy as np
from eeg_logger import logger
from pathlib import Path

def __extract_giga_old(raw_data: mne.io.BaseRaw) -> tuple[mne.Epochs, mne.Epochs]:
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


def extract_epochs(data_path: str, save_path_root: str, resample_to: int = None) -> None:
    if not os.path.exists(data_path):
        logger.error(f"No data to preprocess in {data_path}")
        return

    # Twoja metoda tworząca katalog docelowy (pozostawiona zgodnie z oryginałem)
    save_directory: str = __create_save_directory_giga(save_path_root)

    # Używamy rglob, aby znaleźć pliki .edf bez względu na to, czy są w głównym katalogu, czy w S001/, S002/ itd.
    data_dir_path = Path(data_path)
    subject_files = sorted(list(data_dir_path.rglob("*.edf")))

    for file_path in subject_files:
        subject_id = file_path.stem  # Pobiera samą nazwę pliku, np. "S001" z "S001.edf"

        try:
            subject_num = int(subject_id[1:])
            standardized_subject = f"S{subject_num:03d}"
        except ValueError:
            standardized_subject = subject_id.upper()

        logger.info(f"Reading data from {file_path.name} (mapped to {standardized_subject})...")

        raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose=False)

        # Resampling
        if resample_to is not None:
            if raw.info['sfreq'] != resample_to:
                raw.resample(resample_to)
                logger.info(f"Resampled to {resample_to} Hz")
        else:
            logger.info(f"Keeping native sampling rate: {raw.info['sfreq']} Hz")

        # Ekstrakcja do epok
        epochs_short, epochs_full = __extract_giga(raw)

        # Struktura wynikowa
        subject_save_dir = os.path.join(save_directory, standardized_subject)
        os.makedirs(subject_save_dir, exist_ok=True)

        # Zapis plików FIF z podziałem (np. PA001-2s-epo.fif)
        epochs_short_filename = os.path.join(subject_save_dir, f"PA{standardized_subject[1:4]}-2s-epo.fif")
        epochs_long_filename = os.path.join(subject_save_dir, f"PA{standardized_subject[1:4]}-3s-epo.fif")

        epochs_short.save(epochs_short_filename, overwrite=True)
        epochs_full.save(epochs_long_filename, overwrite=True)

        logger.info(f"Preprocessed data for subject {standardized_subject} saved")


def __extract_giga(raw_data: mne.io.BaseRaw) -> tuple[mne.Epochs, mne.Epochs]:
    # Zabezpieczenie testowe
    if len(raw_data.info['ch_names']) != 64:
        logger.warning(f"Warning: Channel count is {len(raw_data.info['ch_names'])}, expected 64.")

    # Wyodrębnienie zdarzeń wpisanych do struktury przez funkcję konwertującą .mat do .edf
    events, event_ids = mne.events_from_annotations(raw_data, verbose=False)

    # Skoro sami tworzymy strukturę, mamy 100% pewności co do mapowania
    expected_keys = ["imagery_left_hand", "imagery_right_hand"]
    selected_event_id = {k: event_ids[k] for k in expected_keys if k in event_ids}

    if len(selected_event_id) != 2:
        raise ValueError(f"CRITICAL ERROR: Missing expected 'imagery_X_hand' events. Found: {event_ids}")

    # Dobór typów (tylko 64 kanały EEG)
    picks = mne.pick_types(raw_data.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads")

    # W skrypcie konwertującym, adnotacje są umieszczone dokładnie 2 sekundy za startem epoki
    # To odpowiada idealnie chwili zdarzenia bodźca dla pacjenta (Motor Imagery event na t=0)
    # Tniemy więc do przodu wg Twojej starej sprawdzonej metodykli:
    tmin_short, tmax_short = 1, 3  # (ucinamy rozbiegówkę po podaniu komendy o 1s, dając 2 czyste sekundy)
    tmin_full, tmax_full = 0, 3  # (pełne 3 sekundy okna dla Transformera)

    epochs_short = mne.Epochs(
        raw_data,
        events,
        event_id=selected_event_id,
        tmin=tmin_short,
        tmax=tmax_short,
        picks=picks,
        baseline=None,
        preload=True,
        verbose=False
    )

    epochs_full = mne.Epochs(
        raw_data,
        events,
        event_id=selected_event_id,
        tmin=tmin_full,
        tmax=tmax_full,
        picks=picks,
        baseline=None,
        preload=True,
        verbose=False
    )

    # Normalizing samples with (z-score + noise)
    # Zakładam że Twoja funkcja __normalise działa wewnątrz przestrzeni macierzowej poprawnie
    epochs_normalised_2s = __normalise(epochs_short)
    epochs_normalised_3s = __normalise(epochs_full)

    print(
        f"Final 2s epochs size: {len(epochs_normalised_2s)} (Expected length: {len(epochs_short.times)} samples/epoch)")
    print(
        f"Final 3s epochs size: {len(epochs_normalised_3s)} (Expected length: {len(epochs_full.times)} samples/epoch)")

    return epochs_normalised_2s, epochs_normalised_3s

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

