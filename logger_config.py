# logger_config.py

import logging
import sys
import os
import config
from datetime import datetime

LOG_DIR = config.LOG_DIR
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(log_type: str):
    """
    configures a logger that outputs to both the console and a file.
    This function creates a unique log file for each run, named with a timestamp.
    The log file is stored in the LOG_DIR directory, and the logger can be used
    for different types of logs, such as 'generation' or 'benchmark'.    

    Args:
        log_type (str): The type of log being created, e.g., 'generation', 'benchmark'.
    """
    # 1. create a unique log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_type}_{timestamp}.log"
    log_filepath = os.path.join(LOG_DIR, log_filename)

    # 2. add a logger for the specific type
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # 3. prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # 4. create file handler (FileHandler)
    file_handler = logging.FileHandler(log_filepath, mode='w')
    file_handler.setLevel(logging.DEBUG) # file will log DEBUG and above
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # 5. create console handler (StreamHandler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) # console will show INFO and above
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    # 6. add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

