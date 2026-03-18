import logging


class StreamFormatter(logging.Formatter):

    grey = "\x1b[38;21m"
    blue = "\x1b[34m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class FileFormatter(logging.Formatter):

    base_format = "%(asctime)s - %(levelname)s - %(message)s"

    def format(self, record) -> str:
        formatter = logging.Formatter(self.base_format)
        return formatter.format(record)


logger = logging.getLogger("eeg_logger")
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(StreamFormatter())

file_handler = logging.FileHandler(filename="./logs.log")
file_handler.setFormatter(FileFormatter())

logger.addHandler(stream_handler)
logger.addHandler(file_handler)
