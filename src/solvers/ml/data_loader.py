# src/solvers/ml/data_loader.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

from src.utils.generator import load_instance_from_file

logger = logging.getLogger(__name__)

def get_dataset_for_n(n_items: int, data_dir: str):
    """
    Loads all instances for a given n, normalizes them, and prepares them for the model.
    """
    # --- THIS IS THE FIX ---
    # Import cfg here, inside the function, only when it's needed.
    from src.utils.config_loader import cfg
    
    dataset = []
    hyperparams = cfg.ml.dnn.hyperparams
    max_n = hyperparams.max_n
    
    if n_items != -1 and n_items > max_n: # Allow n_items=-1 as a special case for single file loading
        logger.error(f"n_items ({n_items}) exceeds the maximum allowed ({max_n}).")
        return None
    
    # If n_items is -1, it implies we are loading a single file, so we glob the whole directory.
    # Otherwise, we use the specific pattern. This is a small hack for the 'solve' method.
    filename_pattern = f"instance_n{n_items}_*.csv" if n_items != -1 else "*.csv"
    search_path = os.path.join(data_dir, filename_pattern)
    instance_files = glob.glob(search_path)

    if not instance_files:
        logger.warning(f"No .csv files found with pattern '{filename_pattern}' in '{data_dir}'.")
        return None
    
    for filepath in instance_files:
        weights, values, capacity = load_instance_from_file(filepath)
        
        # --- START OF NORMALIZATION ---
        weights_norm = np.array(weights, dtype=np.float32) / hyperparams.max_weight_norm
        values_norm = np.array(values, dtype=np.float32) / hyperparams.max_value_norm
        
        # Handle potential division by zero if weight is zero
        value_densities = values_norm / (weights_norm + 1e-6)
        
        # Handle potential division by zero if capacity is zero
        weight_to_capacity_ratios = np.array(weights, dtype=np.float32) / (capacity + 1e-6)
        
        # Simplified capacity normalization
        normalized_capacity = capacity / (hyperparams.max_n * cfg.data_gen.max_weight)

        feature_vector = np.concatenate([
            weights_norm, 
            values_norm,
            value_densities,
            weight_to_capacity_ratios,
            [normalized_capacity]
        ]).astype(np.float32)

        padding_size = hyperparams.input_size - len(feature_vector)
        if padding_size < 0:
            logger.warning(f"Feature vector for {filepath} is too long ({len(feature_vector)} > {hyperparams.input_size}). Truncating.")
            feature_vector = feature_vector[:hyperparams.input_size]
            padding_size = 0
            
        padded_feature_vector = np.pad(feature_vector, (0, padding_size), 'constant')
        
        # For training, we need the optimal value as a label
        # We use a placeholder here; the 'train' method will compute it.
        dataset.append({
            "features": padded_feature_vector, 
            "label": -1, # Will be calculated during training
            "raw_instance": (weights, values, capacity)
        })
        
    return dataset

class KnapsackDataset(Dataset):
    """PyTorch Dataset for the knapsack problem."""
    def __init__(self, data):
        self.features = [torch.tensor(item["features"]) for item in data]
        self.labels = [torch.tensor([item["label"]]).float() for item in data]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]