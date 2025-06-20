# generate_model_testcases.py
# -*- coding: utf-8 -*-
'''
This script generates a suite of knapsack problem instances ONLY for model training and testing!
'''

import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import generator as gen
import dnn_config as cfg
import logging
from logger_config import setup_logger

setup_logger('instance_generation')
logger = logging.getLogger(__name__)


def model_create_fixed_size_suite(n: int, num_instances: int, mode: str):
    """
    Generates a suite of fixed-size knapsack instances and saves them to the appropriate directory.

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


def generate_varied_size_suite(start_n: int, end_n: int, step_n: int, instances_per_n: int, mode: str):
    """
    Generates enough instances for a range of n values.

    Args:
        start_n (int): The starting number of items.
        end_n (int): The ending number of items.
        step_n (int): The step to increment n by.
        instances_per_n (int): How many instances to generate for each n.
        mode (str): 'train' or 'test'.
    """
    logger.info(f"--- Starting generation for varied sizes from n={start_n} to n={end_n} ---")
    for n in range(start_n, end_n + step_n, step_n):
        model_create_fixed_size_suite(n=n, num_instances=instances_per_n, mode=mode)
    logger.info("--- All generation tasks complete ---")


if __name__ == "__main__":
    # Generate model training instances for n=5 to n=100, with 200 instances per n
    generate_varied_size_suite(
        start_n=5, 
        end_n=100, 
        step_n=5, 
        instances_per_n=200, 
        mode='train'
    )

    # Generate model testing instances for n=5 to n=100, with 50 instances per n
    generate_varied_size_suite(
        start_n=5, 
        end_n=100, 
        step_n=5, 
        instances_per_n=50, 
        mode='test'
    )
    logger.info("Model training and testing instances generation complete.")   


