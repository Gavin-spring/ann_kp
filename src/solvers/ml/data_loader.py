# File: data_loader.py
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
# Local imports
import generator as gen
import algorithms as alg
import configs.dnn_config as cfg

def load_knapsack_dataset_from_files(n_items: int, data_dir: str, max_n: int):
    """
    Loads instance files, computes the optimal solution, and returns a
    FULLY NORMALIZED and padded feature vector and the solution.
    """
    dataset = []
    if n_items > max_n:
        print(f"Error: n_items ({n_items}) exceeds the maximum allowed ({max_n}).")
        return None
    
    filename_pattern = f"instance_n{n_items}_*.csv"
    test_suite_path = os.path.join(data_dir, filename_pattern)
    instance_files = glob.glob(test_suite_path)

    if not instance_files:
        print(f"Warning: No .csv files found for n={n_items} in '{data_dir}'.")
        return None
        
    target_feature_len = max_n * 4 + 1 
        
    for filepath in instance_files:
        weights, values, capacity = gen.load_instance_from_file(filepath)
        
        weights_np = np.array(weights, dtype=np.float32)
        values_np = np.array(values, dtype=np.float32)

        # --- START OF NORMALIZATION ---
        
        # 1. Normalize weights and values to be between [0, 1]
        #    Assumes cfg.MAX_WEIGHT and cfg.MAX_VALUE are the upper bounds from generation.
        weights_norm = weights_np / cfg.MAX_WEIGHT
        values_norm = values_np / cfg.MAX_VALUE

        # 2. Calculate Value Density using the NORMALIZED values
        #    This keeps the density feature on a more controlled scale.
        value_densities = values_norm / (weights_norm + 1e-6)

        # 3. Calculate Weight-to-Capacity Ratio (this is already a ratio, so it's fine)
        weight_to_capacity_ratios = weights_np / capacity

        # 4. Normalize optimal value
        optimal_value = alg.knapsack_gurobi(weights=weights, values=values, capacity=capacity)
        normalized_optimal_value = np.float32(optimal_value / cfg.TARGET_SCALE_FACTOR)
        
        # --- END OF NORMALIZATION ---
        
        # Combine all NORMALIZED features into one vector
        feature_vector = np.concatenate([
            weights_norm, 
            values_norm,
            value_densities,
            weight_to_capacity_ratios,
            [capacity / cfg.MAX_WEIGHT] # Normalized capacity
        ]).astype(np.float32)

        padding_size = target_feature_len - len(feature_vector)
        padded_feature_vector = np.pad(feature_vector, (0, padding_size), 'constant')
                
        dataset.append((padded_feature_vector, normalized_optimal_value))
        
    return dataset

class KnapsackDataset(Dataset):
    """PyTorch Dataset for the knapsack problem."""
    def __init__(self, data):
        self.features = [torch.tensor(item[0]) for item in data]
        self.labels = [torch.tensor(item[1]).unsqueeze(0) for item in data]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]