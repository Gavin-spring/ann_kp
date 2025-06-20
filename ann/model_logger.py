# # File: ann/logger_config.py
# -*- coding: utf-8 -*-

import logging
import sys
import os
from datetime import datetime
import dnn_config as cfg

LOG_DIR = cfg.MODEL_LOGS_DIR
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(log_type: str):
    """
    Configures a logger that outputs to both the console and a file.
    This function creates a unique log file for each run, named with a timestamp.
    """
    # 1. Get the root logger.
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) # Set the lowest level for the logger.

    # 2. Prevent duplicate handlers by clearing any existing ones.
    # This is important to ensure a new file is created for each run.
    if logger.hasHandlers():
        logger.handlers.clear()

    # 3. Create a unique log file name with a timestamp.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_type}_{timestamp}.log"
    log_filepath = os.path.join(LOG_DIR, log_filename)

    # 4. Create file handler to log even debug messages.
    file_handler = logging.FileHandler(log_filepath, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # 5. Create console handler to show info and above.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    # 6. Add handlers to the logger.
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Add a log entry to confirm the file is created.
    logger.info(f"Logger initialized. Log file: {log_filepath}")


