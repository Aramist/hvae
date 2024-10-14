import logging
import sys
from pathlib import Path


class LoggerWrapper:
    """A wrapper around our logger to ensure only one process logs to file"""

    def __init__(self, rank: int, log_path: Path):
        self.rank = rank
        if rank != 0:
            self.logger = None
            self.log_path = None
        else:
            self.logger = logging.getLogger("train_logger")
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(message)s")
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(logging.INFO)
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)
            self.log_path = log_path

    def info(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.info(*args, **kwargs)

    def warn(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.warn(*args, **kwargs)

    def error(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.error(*args, **kwargs)

    def debug(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.debug(*args, **kwargs)

    def critical(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.critical(*args, **kwargs)

    def exception(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.exception(*args, **kwargs)

    def __del__(self):
        if self.logger is not None:
            self.logger.handlers.clear()
            self.logger = None
