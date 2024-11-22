import time
from pathlib import Path
from typing import Dict

from mbmbm import logger


def setup_file_logger(file_path, level="INFO"):
    """init rotating file logging"""
    file = Path(file_path).absolute()
    file.parent.mkdir(exist_ok=True, parents=True)
    handler_id = logger.add(file, retention=3, rotation="20MB", level=level)
    return handler_id


def close_file_logger(handler_id):
    """Remove the file logger handler."""
    logger.remove(handler_id)


class LogTime:
    """Class to measure the execution time in a with block

    Usage:
        with MeasureTime() as t:
            ...
        print(t.duration)
    """

    def __init__(self, caption="Passed time:", time_dict: Dict = None, time_id=None):
        """starts time measure"""
        self.start = 0
        self.end = 0
        self.duration = -1
        self.caption = caption
        self.time_dict = time_dict
        # self.time_value_dict = dict()
        self.time_id = time_id
        if time_dict is not None:
            assert time_id is not None

    def __enter__(self):
        """(re)-starts time measure"""
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """stops time measure and save value"""
        self.end = time.time()
        self.duration = self.end - self.start
        time_value, time_unit = get_proper_time(self.duration)
        logger.info(f"{self.caption} {time_value:2.2f} {time_unit}")
        if self.time_dict is not None:
            if self.time_id in self.time_dict:
                logger.warning(f"Overwriting {self.time_id} in time dict ")
            self.time_dict.update({self.time_id: [f"{time_value:4.2f} {time_unit}", self.duration, self.end]})
            # self.time_value_dict.update({self.time_id: self.duration})

    def duration_till_now(self):
        """returns the time until now but does not save in duration.

        if measure was already stopped, return the real duration (from start to end, NOT from start to now)
        """
        if self.duration != -1:
            return self.duration  # already measured
        return time.time() - self.start


def get_proper_time(sec: float):
    value = sec
    unit = "s"
    if value <= 1:
        value *= 1000
        unit = "mils"
        if value <= 1:
            value *= 1000
            unit = "myks"
        return value, unit
    if value >= 60:
        value /= 60
        unit = "min"

        if value >= 60:
            value /= 60
            unit = "h"
    return value, unit


def get_formatted_time(sec: float, with_linebreak=False):
    value, unit = get_proper_time(sec)
    sep = "\n" if with_linebreak else " "
    return f"{value:4.2f}{sep}{unit}"
