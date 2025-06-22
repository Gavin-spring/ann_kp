# src/solvers/ml/feature_extractor.py
import numpy as np
import torch
import os
import logging

from src.utils.generator import load_instance_from_file

logger = logging.getLogger(__name__)

def extract_features_from_instance(instance_path: str) -> torch.Tensor | None:
    """
    Loads a single instance file, normalizes its data, pads it, and returns
    a feature tensor ready for model inference.

    Args:
        instance_path (str): The path to the .csv instance file.

    Returns:
        torch.Tensor | None: A single feature tensor for the model, or None if an error occurs.
    """
    from src.utils.config_loader import cfg
    try:
        weights, values, capacity = load_instance_from_file(instance_path)
    except Exception as e:
        logger.error(f"Failed to load instance file {instance_path}: {e}")
        return None

    hyperparams = cfg.ml.dnn.hyperparams
    
    # --- START OF NORMALIZATION (Identical to the logic in preprocess_data.py) ---
    weights_norm = np.array(weights, dtype=np.float32) / hyperparams.max_weight_norm
    values_norm = np.array(values, dtype=np.float32) / hyperparams.max_value_norm
    
    value_densities = values_norm / (weights_norm + 1e-6)
    weight_to_capacity_ratios = np.array(weights, dtype=np.float32) / (capacity + 1e-6)
    normalized_capacity = capacity / (hyperparams.max_n * cfg.data_gen.max_weight)

    feature_vector = np.concatenate([
        weights_norm, 
        values_norm,
        value_densities,
        weight_to_capacity_ratios,
        [normalized_capacity]
    ]).astype(np.float32)

    # --- Padding ---
    padding_size = hyperparams.input_size - len(feature_vector)
    if padding_size < 0:
        logger.warning(f"Feature vector for {instance_path} is too long ({len(feature_vector)} > {hyperparams.input_size}). Truncating.")
        feature_vector = feature_vector[:hyperparams.input_size]
        padding_size = 0
        
    padded_feature_vector = np.pad(feature_vector, (0, padding_size), 'constant')
    
    return torch.tensor(padded_feature_vector)