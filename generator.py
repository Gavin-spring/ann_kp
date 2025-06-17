# generator.py
# -*- coding: utf-8 -*-


'''
This module provides functions to generate different variants of instances of the knapsack problem.
'''

import random
from typing import List, Tuple
import os
import csv
import logging
logger = logging.getLogger(__name__)


# Function to generate a knapsack instance with one constraint
def generate_knapsack_instance(
    n: int,
    correlation: str = 'uncorrelated',
    max_weight: int = 1000,
    max_value: int = 1000,
    capacity_ratio: float = 0.5
) -> Tuple[List[Tuple[int, int]], int]:

    """
    Generate an instance of the knapsack problem with one constraintion.

    Args:
        n (int): Number of items to generate.
        correlation (str): Type of correlation between item values and weights.
            Options: 'uncorrelated', 'weakly_correlated',
                    'strongly_correlated', 'subset_sum'.
        max_weight (int): Maximum weight for a single item.
        max_value (int): Maximum value for a single item (used when uncorrelated).
        capacity_ratio (float): Ratio of knapsack capacity to the total weight of all items (between 0.0 and 1.0).

    Returns:
        Tuple[List[Tuple[int, int]], int]:
            - A list of items, each represented as a tuple (value, weight).
            - The computed knapsack capacity.
    """

    if correlation not in ['uncorrelated', 'weakly_correlated', 'strongly_correlated', 'subset_sum']:
        raise ValueError("Correlation type must be one of 'uncorrelated', 'weakly_correlated', 'strongly_correlated', or 'subset_sum'")

    items = []
    total_weight = 0

    for _ in range(n):
        weight = random.randint(1, max_weight)
        value = 0

        if correlation == 'uncorrelated':
            value = random.randint(1, max_value)
        elif correlation == 'weakly_correlated':
            # Value and weight are weakly correlated, with significant noise
            # The noise range can be around 25% of the maximum value
            noise = int(max_value / 4)
            value = max(1, weight + random.randint(-noise, noise))
        elif correlation == 'strongly_correlated':
            # Value and weight are strongly correlated, with very little noise  
            # The noise range can be around 10% of the maximum value
            noise = int(max_value / 10)
            value = max(1, weight + random.randint(-noise, noise))
        elif correlation == 'subset_sum':
            # Value equals weight
            # this is the subset sum problem
            value = weight
        
        items.append((value, weight))
        total_weight += weight

    # Calculate knapsack capacity based on total weight and ratio
    if not (0.0 < capacity_ratio <= 1.0):
        raise ValueError("Capacity ratio must be between 0.0 and 1.0")
    capacity = int(total_weight * capacity_ratio)

    return items, capacity



# Function to generate a multi-dimensional knapsack problem instance
def generate_mckp_instance(
    n: int,
    num_constraints: int,
    max_weights: List[int],
    max_value: int = 1000,
    capacity_ratios: List[float] = None
) -> Tuple[List[Tuple[int, List[int]]], List[int]]:
    """
    generate a multi-dimensional knapsack problem instance.
    Args:
        n (int): Number of items to generate.
        num_constraints (int): Number of constraints (dimensions).
        max_weights (List[int]): Maximum weight for each constraint.
        max_value (int): Maximum value for a single item.
        capacity_ratios (List[float]): Ratios of knapsack capacities to the total weights of all items.

    Returns:
        - items: [(value, [weight1, weight2, ...]), ...]
        - capacities: [capacity1, capacity2, ...]
    """
    if len(max_weights) != num_constraints or len(capacity_ratios) != num_constraints:
        # Ensure that max_weights and capacity_ratios match the number of constraints
        raise ValueError("max_weights and capacity_ratios must have the same length as num_constraints")

    items = []
    total_weights = [0] * num_constraints

    for _ in range(n):
        value = random.randint(1, max_value)
        weights = [random.randint(1, max_w) for max_w in max_weights]
        items.append((value, weights))
        for i in range(num_constraints):
            total_weights[i] += weights[i]

    capacities = [int(total_weights[i] * capacity_ratios[i]) for i in range(num_constraints)]

    return items, capacities

# TODO: refactor save_instance_to_file and load_instance_from_file to handle multi-dimensional instances


def save_instance_to_file(items: List[Tuple[int, int]], capacity: int, filename: str):
    """Saves the generated instance to a csv file."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', newline='') as f:
        # Write the first line with number of total items and capacity
        f.write(f"{len(items)} {capacity}\n")
        writer = csv.writer(f)
        writer.writerow(['value', 'weight'])
        # Write each item as a row in the csv file
        for value, weight in items:
            writer.writerow([value, weight])
            
    logger.info(f"Instance successfully saved to {filename}")



def load_instance_from_file(filename: str) -> Tuple[List[int], List[int], int]:
    """
    Loads a knapsack instance from a csv file.
    Assumes first line is 'num_items capacity', and subsequent lines are 'weight value'.
    Includes validation to check if the number of items matches the header.

    Returns:
        Tuple[List[int], List[int], int]: (weights_list, values_list, capacity)
    """
    weights = []
    values = []
    
    with open(filename, 'r', newline='') as f:
        # 1. Read and use meta-data from the first line
        meta_line = f.readline().strip()
        num_items_str, capacity_str = meta_line.split()
        capacity = int(capacity_str)
        # Convert num_items_str to an integer for later validation
        expected_num_items = int(num_items_str)

        # 2. Use csv.reader to process the rest of the file
        reader = csv.reader(f)
        
        # 3. Skip the header row
        try:
            next(reader)
        except StopIteration:
            # This handles the case where the file has a header but no data rows
            logger.warning(f"Warning: File '{filename}' contains no data rows.")
            # We can decide to return early or continue
        
        # 4. Read each data row
        for row in reader:
            values.append(int(row[0]))
            weights.append(int(row[1]))

    # 5. VALIDATION STEP: Check if the actual number of items matches the expected number
    actual_num_items = len(values)
    if actual_num_items != expected_num_items:
        logger.warning(f"Inconsistent data in '{filename}'. \
                       Header specified {expected_num_items} items, but file contained {actual_num_items} items.")

    logger.info(f"Instance successfully loaded from {filename} ({actual_num_items} items).")
    return weights, values, capacity


if __name__ == '__main__':
    logger.info("This script provides functions to generate and handle knapsack instances.")
    logger.info("To generate a full test suite, run 'generate_test_suite.py'.")

# Example usage of the generator
# if __name__ == "__main__":
#     print("--- Example 1.1: Small, Uncorrelated Instance ---")
#     small_items, small_capacity = generate_knapsack_instance(
#         n=15,
#         correlation='uncorrelated',
#         max_weight=10,
#         max_value=1000,
#         capacity_ratio=0.6
#     )
#     print(f"Generated {len(small_items)} items with capacity {small_capacity}")
#     save_instance_to_file(small_items, small_capacity, "small_uncorrelated_2.txt")
#     print("-" * 40)
