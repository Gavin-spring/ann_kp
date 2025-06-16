# logger_config.py
import logging
import sys

def setup_logger():
    """Sets up the root logger for the project."""
    logging.basicConfig(
        level=logging.INFO, # logging levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
        stream=sys.stdout,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


# TODO: replace with a more sophisticated logger setup if needed