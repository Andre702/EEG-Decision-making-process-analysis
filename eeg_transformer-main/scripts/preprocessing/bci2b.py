import os, mne, shutil
from eeg_logger import logger

"""
# Event |  Type  | Description
_________________________________________
# 276   | 0x0114 | Idling EEG (eyes open)
# 277   | 0x0115 | Idling EEG (eyes closed)
# 768   | 0x0300 | Start of a trial
# 769   | 0x0301 | Cue onset left (class 1)
# 770   | 0x0302 | Cue onset right (class 2)
# 781   | 0x030D | BCI feedback (continuous)
# 783   | 0x030F | Cue unknown
# 1023  | 0x03FF | Rejected trial
# 1077  | 0x0435 | Horizontal eye movement
# 1078  | 0x0436 | Vertical eye movement
# 1079  | 0x0437 | Eye rotation
# 1081  | 0x0439 | Eye blinks
# 32766 | 0x7FFE | Start of a new run
"""


def extract_epochs(data_path: str, save_path_root: str) -> None:

    if not os.path.exists(data_path):
        logger.error(f"No data to preprocess in {data_path}")
        return

    save_directory: str = __create_save_directory(save_path_root)
    subject_folders = [f for f in os.listdir(data_path)]

    for subject in subject_folders:
        subject_dir = os.path.join(data_path, subject)

        subject_save_dir = os.path.join(save_directory, subject)
        os.makedirs(subject_save_dir, exist_ok=True)

        train_data_files = [f for f in os.listdir(subject_dir) if f.endswith("T.gdf")]
        logger.info(f"Reading data from {subject}...")

        for idx, train_file in enumerate(train_data_files[:2], start=1):
            train_file_path = os.path.join(subject_dir, train_file)

            raw_train = mne.io.read_raw_gdf(train_file_path, eog=["EOG-left", "EOG-central", "EOG-right"], preload=True)
            train_epochs = __extract(raw_train)

            train_filename = os.path.join(subject_save_dir, f"PB{subject[1:3]}0{idx}T-epo.fif")
            train_epochs.save(train_filename)
            logger.info(f"Training data for subject {subject[1:3]} saved as {train_filename}")


def __extract(raw_data: mne.io.BaseRaw) -> mne.Epochs:

    events, event_ids = mne.events_from_annotations(raw_data)  # EXTRACT EVENTS
    logger.info(f"Event ids: {event_ids}")  # THIS IS IMPORTANT BECAUSE IT PROVIDES MAPPING TO EVENT IDS
    selected_event_id = None

    if event_ids["769"] == 10 and event_ids["770"] == 11:
        logger.info("Using event IDs 769 and 770, mapping to 10 and 11 for left and right hand.")
        selected_event_id = {"left_hand": 10, "right_hand": 11}
    else:
        logger.warning("Event values for IDs 769, 770 not matching 10 and 11, falling back to 4, 5.")
        selected_event_id = {"left_hand": 4, "right_hand": 5}

    # PICK ONLY EEG
    picks = mne.pick_types(raw_data.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads")

    epochs = mne.Epochs(
        raw_data,
        events,
        event_id=selected_event_id,
        picks=picks,
        baseline=None,
        preload=True,
    )
    logger.info(f"Epochs: {epochs}")
    return epochs


def __create_save_directory(save_path_root: str) -> str:

    path: str = f"{save_path_root}/BCI_IV_2b"

    if os.path.exists(path):
        logger.info("Removing old preprocess directory for BCI_IV_2b")
        shutil.rmtree(path)

    os.makedirs(path)

    return path
