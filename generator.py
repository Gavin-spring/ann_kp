# generator.py
# -*- coding: utf-8 -*-

import random
from typing import List, Tuple
import os
import csv


def generate_knapsack_instance(
    n: int,
    correlation: str = 'uncorrelated',
    max_weight: int = 1000,
    max_value: int = 1000,
    capacity_ratio: float = 0.5
) -> Tuple[List[Tuple[int, int]], int]:

    """
    Generate an instance of the knapsack problem.

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
            
    print(f"Instance successfully saved to {filename}")



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
            print(f"Warning: File '{filename}' contains no data rows.")
            # We can decide to return early or continue
        
        # 4. Read each data row
        for row in reader:
            values.append(int(row[0]))
            weights.append(int(row[1]))

    # 5. VALIDATION STEP: Check if the actual number of items matches the expected number
    actual_num_items = len(values)
    if actual_num_items != expected_num_items:
        print(f"Warning: Inconsistent data in '{filename}'.")
        print(f"  Header specified {expected_num_items} items, but file contained {actual_num_items} items.")

    print(f"Instance successfully loaded from {filename} ({actual_num_items} items).")
    return weights, values, capacity


if __name__ == '__main__':
    print("This script provides functions to generate and handle knapsack instances.")
    print("To generate a full test suite, run 'generate_test_suite.py'.")

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
