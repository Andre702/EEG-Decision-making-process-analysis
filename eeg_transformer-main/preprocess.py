import os, sys
import scripts.preprocessing.bci2a as bci2a
import scripts.preprocessing.bci2b as bci2b
import scripts.preprocessing.bci3a as bci3a
import scripts.preprocessing.physionet as physionet
from eeg_logger import logger
from download import DATA_BASE_DIR

PREPROCESSED_DATA_BASE_DIR: str = "./preprocessed_data"


def main() -> None:
    dataset_name: str = sys.argv[1] if len(sys.argv) > 1 else ""

    if not os.path.exists(PREPROCESSED_DATA_BASE_DIR):
        os.makedirs(PREPROCESSED_DATA_BASE_DIR)

    match dataset_name:
        case "bci3a":
            bci3a.extract_epochs(data_path=f"{DATA_BASE_DIR}/BCI_III_3a", save_path_root=PREPROCESSED_DATA_BASE_DIR)
        case "bci2a":
            bci2a.extract_epochs(data_path=f"{DATA_BASE_DIR}/BCI_IV_2a", save_path_root=PREPROCESSED_DATA_BASE_DIR)
        case "bci2b":
            bci2b.extract_epochs(data_path=f"{DATA_BASE_DIR}/BCI_IV_2b", save_path_root=PREPROCESSED_DATA_BASE_DIR)
        case "physionet":
            physionet.extract_epochs(data_path=f"{DATA_BASE_DIR}/Physionet", save_path_root=PREPROCESSED_DATA_BASE_DIR)
        case _:
            logger.warning("No dataset to preprocess provided")


if __name__ == "__main__":
    main()
    # input("Press Enter to exit...")