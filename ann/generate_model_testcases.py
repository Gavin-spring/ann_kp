# generate_model_testcases.py
# -*- coding: utf-8 -*-
'''
This script generates a suite of knapsack problem instances ONLY for model training and testing!
'''

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import generator as gen
import model_config as cfg
import logging
from logger_config import setup_logger

# Configurations for logging
setup_logger('model_testcase')
logger = logging.getLogger(__name__)

# Generate a suite of fixed-size knapsack instances for model training or testing
def model_create_fixed_size_suite(n: int, num_instances: int, mode: str):
    """
    Generates a suite of knapsack instances and saves them to the appropriate directory.

    Args:
        n (int): The fixed number of items for each instance.
        num_instances (int): The total number of instances to generate.
        mode (str): The purpose of the generation. Must be 'train' or 'test'.
    """
    if mode == 'train':
        target_dir = cfg.MODEL_TRAIN_CASES #
        purpose_str = "training"
    elif mode == 'test':
        target_dir = cfg.MODEL_TEST_CASES #
        purpose_str = "testing"
    else:
        raise ValueError("Mode must be either 'train' or 'test'")

    logger.info(f"--- Generating {num_instances} instances for {purpose_str} of size n={n} in '{target_dir}' ---")
    os.makedirs(target_dir, exist_ok=True)

    for i in range(num_instances):
        instance_number = i + 1
        logger.info(f"\nGenerating instance {instance_number}/{num_instances} for n = {n}...")

        # Generate the instance using the core generator and config parameters
        items, capacity = gen.generate_knapsack_instance(
            n=n,
            correlation=cfg.CORRELATION_TYPE, #
            max_weight=cfg.MAX_WEIGHT, #
            max_value=cfg.MAX_VALUE, #
            capacity_ratio=cfg.CAPACITY_RATIO #
        )

        # Define a descriptive, unique filename for each instance.
        filename = os.path.join(
            target_dir,
            f"instance_n{n}_{cfg.CORRELATION_TYPE}_{instance_number}.csv"
        )

        # Save the instance to the file
        gen.save_instance_to_file(items, capacity, filename)

    logger.info(f"\n--- Generation of {num_instances} fixed-size instances complete ---")

if __name__ == "__main__":
    # Generate 1000 instances (n=10) for model training and validation
    logger.info("Starting model training case generation...")
    model_create_fixed_size_suite(n=10, num_instances=1000, mode='train')
    logger.info("Model training case generation completed successfully.")
    
    # Generate 200 instances (n=10) for final model testing
    logger.info("Starting model testing case generation...")
    model_create_fixed_size_suite(n=10, num_instances=200, mode='test')
    logger.info("Model test case generation completed successfully.")
    


