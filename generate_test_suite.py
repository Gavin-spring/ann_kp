# generate_test_suite.py
# -*- coding: utf-8 -*-


'''
This script generates a suite of knapsack problem instances for benchmarking algorithms.
It creates a series of CSV files in the specified directory, each containing a unique instance.
'''

import os
import generator as gen
import config as cfg


# --- Test suite Configuration ---
TEST_SUITE_DIR = cfg.TEST_SUITE_DIR
CORRELATION_TYPE = 'uncorrelated'
N_RANGE = cfg.N_RANGE
# Item properties
MAX_WEIGHT = cfg.MAX_WEIGHT
MAX_VALUE = cfg.MAX_VALUE
CAPACITY_RATIO = cfg.CAPACITY_RATIO


def create_suite():
    """Generates and saves a suite of knapsack problem instances."""
    print(f"--- Generating Test Suite in '{TEST_SUITE_DIR}' Directory ---")
    
    # Ensure the target directory exists
    os.makedirs(TEST_SUITE_DIR, exist_ok=True)

    for n_items in range(N_RANGE[0], N_RANGE[1], N_RANGE[2]):
        print(f"\nGenerating instance for n = {n_items}...")
        
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
        
    print("\n--- Test Suite Generation Complete ---")

if __name__ == "__main__":
    create_suite()