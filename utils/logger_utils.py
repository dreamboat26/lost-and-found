"""
 # @ Author: Meet Patel
 # @ Create Time: 2024-07-21 08:24:18
 # @ Modified by: Meet Patel
 # @ Modified time: 2024-07-21 08:24:20
 # @ Description:
 """

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - Line %(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def configure_logging(log_file: str) -> None:
    """
    Configure the logger to dump the logs in given file.

    Args:
        log_file (str): Log file to store the logs.
    """
    for handler in logger.handlers:
        logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
