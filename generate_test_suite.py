# generate_test_suite.py
# -*- coding: utf-8 -*-


'''
This script generates a suite of knapsack problem instances for benchmarking algorithms.
It creates a series of CSV files in the specified directory, each containing a unique instance.
'''

import os
import generator as gen
import config as cfg
import logging
from logger_config import setup_logger


# --- Test suite Configuration ---
TEST_SUITE_DIR = cfg.TEST_SUITE_DIR
CORRELATION_TYPE = 'uncorrelated'
N_RANGE = cfg.N_RANGE
# Item properties
MAX_WEIGHT = cfg.MAX_WEIGHT
MAX_VALUE = cfg.MAX_VALUE
CAPACITY_RATIO = cfg.CAPACITY_RATIO

# --- Logger Setup ---
setup_logger('generation')
logger = logging.getLogger(__name__)

# --- Function to Create the Test Suite ---
# This function generates a suite of 0-1 knapsack problem instances of varying sizes.
def create_suite():
    """Generates and saves a suite of knapsack problem instances."""
    logger.info(f"--- Generating Test Suite in '{TEST_SUITE_DIR}' Directory ---")
    
    # Ensure the target directory exists
    os.makedirs(TEST_SUITE_DIR, exist_ok=True)

    for n_items in range(N_RANGE[0], N_RANGE[1], N_RANGE[2]):
        logger.info(f"\nGenerating instance for n = {n_items}...")
        
        # Generate the instance
        items, capacity = gen.generate_knapsack_instance(
            n=n_items,
            correlation=CORRELATION_TYPE,
            max_weight=MAX_WEIGHT,
            max_value=MAX_VALUE,
            capacity_ratio=CAPACITY_RATIO
        )
        
        # Define a descriptive filename
        filename = os.path.join(TEST_SUITE_DIR, f"instance_n{n_items}_{CORRELATION_TYPE}.csv")
        
        # Save the instance to the file
        gen.save_instance_to_file(items, capacity, filename)
        
    logger.info("\n--- Test Suite Generation Complete ---")

# create 0-1 instances of fixed size n
def create_fixed_size_suite(n: int, num_instances: int):
    """
    Generates and saves a suite of knapsack problem instances of a fixed size.
    Each instance will be saved as a separate file.

    Args:
        n (int): The fixed number of items for each instance.
        num_instances (int): The total number of instances to generate.
    """
    logger.info(f"--- Generating {num_instances} instances of size n={n} in '{cfg.TEST_SUITE_DIR}' ---") #

    # Ensure the target directory exists
    os.makedirs(cfg.TEST_SUITE_DIR, exist_ok=True)

    for i in range(num_instances):
        instance_number = i + 1
        logger.info(f"\nGenerating instance {instance_number}/{num_instances} for n = {n}...")

        # Generate the instance using the core generator and config parameters
        items, capacity = gen.generate_knapsack_instance(
            n=n,
            correlation=cfg.CORRELATION_TYPE,
            max_weight=cfg.MAX_WEIGHT,
            max_value=cfg.MAX_VALUE,
            capacity_ratio=cfg.CAPACITY_RATIO
        )

        # Define a descriptive, unique filename for each instance.
        # The instance number (e.g., _1, _2) is added to prevent overwriting files.
        filename = os.path.join(
            cfg.TEST_SUITE_DIR,
            f"instance_n{n}_{cfg.CORRELATION_TYPE}_{instance_number}.csv"
        )

        # Save the instance to the file
        gen.save_instance_to_file(items, capacity, filename)

    logger.info(f"\n--- Generation of {num_instances} fixed-size instances complete ---")

if __name__ == "__main__":
    # create_suite()
    
    fixed_item_count = 10
    number_of_instances_to_create = 10    
    create_fixed_size_suite(n=fixed_item_count, num_instances=number_of_instances_to_create)
