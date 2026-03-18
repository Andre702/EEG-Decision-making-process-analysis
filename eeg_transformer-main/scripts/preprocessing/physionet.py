import os, mne, shutil
import numpy as np
from eeg_logger import logger

"""
In Physionet dataset, runs regarding motor imagery are [4, 8, 12].
Each annotation includes one of three codes (T0, T1, or T2):

- T0 corresponds to rest
- T1 corresponds to onset of motion (real or imagined) of the left fist (in runs 3, 4, 7, 8, 11, and 12)
- T2 corresponds to onset of motion (real or imagined) of the right fist (in runs 3, 4, 7, 8, 11, and 12)

The preprocessing is done as specified in article:

For each subject, 21 trials were selected per class. Each trial lasted 8 seconds, with the
first 2s for rest, the following 4s for motor imagery, and the last 2s for rest.
3s (480 samples) and 6s (960 samples) segments of EEGdata were used to train and test our models.
We used both 3s and 6s data for the classification. 3s data included the first 3s data from the motor imagery period,
and 6s data included the entire motor imagery period as well as one second before
and one second after the motor imagery period.

We applied the Z-score normalization to preprocess the EEG data, and
added the random noise to prevent over-fitting, as shown in the following formula:

X* = (X - mean) / std + aN

where:
X* - normalised epochs data
X - raw epochs data
mean - mean value of data
std - standard deviation of data
N - random noise, used np.random.randn
a - percent of random noise, set to 0.01

"""


def extract_epochs(data_path: str, save_path_root: str) -> None:

    if not os.path.exists(data_path):
        logger.error(f"No data to preprocess in {data_path}")
        return

    save_directory: str = __create_save_directory(save_path_root)
    subject_folders = [f for f in os.listdir(data_path)]
    bad_subjects = [88, 92, 100, 104]  # THESE SUBJECTS HAVE INCOMPLETE ANNOTATIONS

    for idx, subject in enumerate(subject_folders, start=1):

        if idx in bad_subjects:
            continue

        data_file_run_4 = os.path.join(data_path, subject, f"S{subject[1:4]}R04.edf")
        data_file_run_8 = os.path.join(data_path, subject, f"S{subject[1:4]}R08.edf")
        data_file_run_12 = os.path.join(data_path, subject, f"S{subject[1:4]}R12.edf")

        logger.info(f"Reading data from {subject}...")

        raw_run_4 = mne.io.read_raw_edf(data_file_run_4, preload=True)
        raw_run_8 = mne.io.read_raw_edf(data_file_run_8, preload=True)
        raw_run_12 = mne.io.read_raw_edf(data_file_run_12, preload=True)

        raws = mne.concatenate_raws([raw_run_4, raw_run_8, raw_run_12])
        epochs_3s, epochs_6s = __extract(raws)

        os.makedirs(os.path.join(save_directory, subject))

        epochs_3s_filename = os.path.join(save_directory, subject, f"PA{subject[1:4]}-3s-epo.fif")
        epochs_6s_filename = os.path.join(save_directory, subject, f"PA{subject[1:4]}-6s-epo.fif")

        epochs_3s.save(epochs_3s_filename)
        epochs_6s.save(epochs_6s_filename)

        logger.info(f"Preprocessed data for subject {subject[1:4]} saved")


def __extract(raw_data: mne.io.BaseRaw) -> tuple[mne.Epochs, mne.Epochs]:

    events, event_ids = mne.events_from_annotations(raw_data)  # EXTRACT EVENTS
    logger.info(f"Event ids: {event_ids}")  # THIS IS IMPORTANT BECAUSE IT PROVIDES MAPPING TO EVENT IDS
    selected_event_id = {"left_hand": 2, "right_hand": 3}  # BASED ON EVENT_IDS

    # PICK ONLY EEG
    picks = mne.pick_types(raw_data.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads")
    tmin_3s, tmax_3s = 2.0, 5.0
    tmin_6s, tmax_6s = 1.0, 7.0

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

    epochs_6s = mne.Epochs(
        raw_data,
        events,
        event_id=selected_event_id,
        tmin=tmin_6s,
        tmax=tmax_6s,
        picks=picks,
        baseline=None,
        preload=True,
    )

    epochs_normalised_3s = __normalise(epochs_3s)
    epochs_normalised_6s = __normalise(epochs_6s)

    logger.info(f"Extracted {len(epochs_normalised_3s)} epochs (3s) and {len(epochs_normalised_6s)} epochs (6s)")
    return epochs_normalised_3s, epochs_normalised_6s


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


def __create_save_directory(save_path_root: str) -> str:

    path: str = f"{save_path_root}/Physionet"

    if os.path.exists(path):
        logger.info("Removing old preprocess directory for Physionet")
        shutil.rmtree(path)

    os.makedirs(path)

    return path
