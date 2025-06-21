# generate_data.py
# -*- coding: utf-8 -*-

"""
This is the single entry point for generating all datasets for the project.
It uses the configuration from 'configs/config.yaml' and the core functions
from 'src/utils/generator.py'.
"""

import os
import logging
from tqdm import tqdm # Using tqdm for a nice progress bar

# --- Import project modules ---
from src.utils.config_loader import cfg
from src.utils.logger import setup_logger
import src.utils.generator as gen # Import our new generator library

def create_dataset(
    dataset_name: str,
    output_dir: str,
    instance_params: dict,
    n_range: tuple = None,
    n_fixed: int = None,
    num_instances: int = 1
):
    """
    A generic function to create a dataset of knapsack instances.

    Args:
        dataset_name (str): A name for the generation task (e.g., 'DNN-Training').
        output_dir (str): The directory to save the instance files.
        instance_params (dict): Parameters for the instance generator.
        n_range (tuple): A tuple for varied sizes (start, stop, step).
        n_fixed (int): A fixed size for all instances.
        num_instances (int): The number of instances to generate for each size 'n'.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"--- Starting dataset generation: '{dataset_name}' ---")
    os.makedirs(output_dir, exist_ok=True)
    
    if n_range:
        range_of_n = range(n_range[0], n_range[1], n_range[2])
        total_tasks = len(range_of_n) * num_instances
    elif n_fixed:
        range_of_n = [n_fixed]
        total_tasks = num_instances
    else:
        logger.error("Either n_range or n_fixed must be provided.")
        return

    with tqdm(total=total_tasks, desc=f"Generating {dataset_name}") as pbar:
        for n in range_of_n:
            for i in range(num_instances):
                items, capacity = gen.generate_knapsack_instance(
                    n=n,
                    correlation=instance_params['correlation'],
                    max_weight=instance_params['max_weight'],
                    max_value=instance_params['max_value'],
                    capacity_ratio=instance_params['capacity_ratio']
                )

                filename = os.path.join(output_dir, f"instance_n{n}_{instance_params['correlation']}_{i+1}.csv")
                gen.save_instance_to_file(items, capacity, filename)
                pbar.update(1)

    logger.info(f"--- Dataset generation '{dataset_name}' complete. Files saved in '{output_dir}'. ---")

if __name__ == '__main__':
    # --- Configure logger ONCE for this script run ---
    setup_logger(run_name="data_generation", log_dir=cfg.paths.logs)

    # --- Control Panel for Generating Datasets ---
    # You can uncomment the section you want to run.
    
    # === Task 1: Generate the test suite for classic algorithms ===
    # print("Running Task 1: Generate test suite for classic algorithms...")
    # classic_params = {
    #     'correlation': cfg.data_gen.correlation_type,
    #     'max_weight': cfg.data_gen.max_weight,
    #     'max_value': cfg.data_gen.max_value,
    #     'capacity_ratio': cfg.data_gen.capacity_ratio,
    # }
    # create_dataset(
    #     dataset_name="Classic-Benchmark-Suite",
    #     output_dir=cfg.paths.data_testing,
    #     instance_params=classic_params,
    #     n_range=cfg.data_gen.n_range,
    #     num_instances=1 # One instance per size 'n' for the classic suite
    # )

    # === Task 2: Generate the training set for the DNN model ===
    print("Running Task 2: Generate training set for DNN model...")
    dnn_params = {
        'correlation': cfg.data_gen.correlation_type, # Using shared generation params
        'max_weight': cfg.data_gen.max_weight,
        'max_value': cfg.data_gen.max_value,
        'capacity_ratio': cfg.data_gen.capacity_ratio,
    }
    dnn_n_range = (cfg.ml.dnn.generation.start_n, cfg.ml.dnn.generation.end_n + 1, cfg.ml.dnn.generation.step_n)
    create_dataset(
        dataset_name="DNN-Training-Set",
        output_dir=cfg.paths.data_training,
        instance_params=dnn_params,
        n_range=dnn_n_range,
        num_instances=200 # Example: 200 instances per size 'n'
    )

    # === Task 3: Generate the validation set for the DNN model ===
    # print("Running Task 3: Generate validation set for DNN model...")
    # dnn_n_range_val = (cfg.ml.dnn.generation.start_n, cfg.ml.dnn.generation.end_n + 1, cfg.ml.dnn.generation.step_n)
    # create_dataset(
    #     dataset_name="DNN-Validation-Set",
    #     output_dir=cfg.paths.data_validation,
    #     instance_params=dnn_params,
    #     n_range=dnn_n_range_val,
    _instances=50 # Example: 50 instances per size 'n'
    # )

    print("\nAll selected data generation tasks are complete.")