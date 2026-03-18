import os, mne, shutil
from eeg_logger import logger

"""
# Event | Type  | Description
_________________________________________
# 768   | 0x0300 | Start of a trial
# 769   | 0x0301 | Left hand (class 1)
# 770   | 0x0302 | Right hand (class 2)
# 771   | 0x0303 | Foot (class 3)
# 772   | 0x0304 | Tongue (class 4)
# 783   | 0x030F | Cue unknown
# 1023  | 0x03FF | Rejected trial
# 1072  | 0x0430 | Eye movements
# 32766 | 0x7FFE | Start of a new run
"""


def extract_epochs(data_path: str, save_path_root: str) -> None:
    if not os.path.exists(data_path):
        logger.error(f"No data to preprocess in {data_path}")
        return

    save_directory: str = __create_save_directory(save_path_root)
    subject_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]

    for subject in subject_folders:
        data_file = os.path.join(data_path, subject, f"{subject[1:3]}.gdf")

        if not os.path.exists(data_file):
            logger.warning(f"File {data_file} not found. Skipping subject {subject}")
            continue

        logger.info(f"Reading data from {subject}...")
        raw = mne.io.read_raw_gdf(data_file, preload=True)
        epochs = __extract(raw)

        os.makedirs(os.path.join(save_directory, subject), exist_ok=True)

        filename = os.path.join(save_directory, subject, f"{subject[1:3]}-epo.fif")
        epochs.save(filename)
        logger.info(f"Preprocessed data for subject {subject[1:3]} saved as {filename}")


def __extract(raw_data: mne.io.BaseRaw) -> mne.Epochs:
    events, event_ids = mne.events_from_annotations(raw_data)  # EXTRACT EVENTS
    logger.info(f"Event ids: {event_ids}")  # Log event IDs to verify they're correct

    # Only keep events for left hand (3) and right hand (4) as specified in the dataset
    selected_event_id = {"left_hand": 3, "right_hand": 4}  # Corrected event IDs based on dataset description

    # PICK ONLY EEG
    picks = mne.pick_types(raw_data.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads")

    # Adjusted time window to match the trial timing in the dataset (0s to 7s)
    tmin, tmax = 0.0, 7.0

    # Create epochs for the selected events (left hand and right hand)
    epochs = mne.Epochs(
        raw_data,
        events,
        event_id=selected_event_id,
        tmin=tmin,
        tmax=tmax,
        picks=picks,
        baseline=None,
        preload=True,
        event_repeated="merge",  # Handle repeated events by merging them
    )

    logger.info(f"Number of epochs: {len(epochs)}")
    logger.info(f"Epochs: {epochs}")
    return epochs


def __create_save_directory(save_path_root: str) -> str:
    path: str = f"{save_path_root}/BCI_III_3a"

    if os.path.exists(path):
        logger.info("Removing old preprocess directory for BCI_III_3a")
        shutil.rmtree(path)

    os.makedirs(path)

    return path
