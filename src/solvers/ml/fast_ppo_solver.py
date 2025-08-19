# In src/solvers/ml/fast_ppo_solver.py
# REPLACE the entire file with this DEFINITIVE FINAL version.

import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List

from src.solvers.ml.ppo_solver import PPOSolver

class FastPPOSolver(PPOSolver):
    def __init__(self, model_run_dir: str, **kwargs):
        from src.utils.config_loader import cfg
        super().__init__(model_run_dir)
        self.is_deterministic = cfg.ml.testing.is_deterministic
        self.name = "PPO"

    def solve(self, instance_path: str) -> Dict[str, Any]:
        raise NotImplementedError("FastPPOSolver is designed for batch evaluation via solve_batch.")

    def solve_batch(self, batch_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        from src.utils.config_loader import cfg
        # --- 0. Sort items by value-to-weight ratio ---
        raw_weights_unsorted = batch_data['weights'].to(self.model.device)
        raw_values_unsorted = batch_data['values'].to(self.model.device)
        raw_capacity = batch_data['capacity'].to(self.model.device)
        mask_unsorted = batch_data['attention_mask'].to(self.model.device)

        ratio = raw_values_unsorted / (raw_weights_unsorted + 1e-9)
        ratio[~mask_unsorted] = -1.0 
        sorted_indices = torch.argsort(ratio, dim=1, descending=True)
        
        raw_weights = torch.gather(raw_weights_unsorted, 1, sorted_indices)
        raw_values = torch.gather(raw_values_unsorted, 1, sorted_indices)
        
        # --- 1. Pre-processing (Masking, Scaling, Padding, Normalization) ---
        mask_tensor = torch.gather(mask_unsorted, 1, sorted_indices)
        initial_capacity_mask = raw_weights <= raw_capacity.unsqueeze(1)
        mask_tensor = mask_tensor & initial_capacity_mask

        # --- 1b. CRITICAL FIX: Manually scale the data before normalization ---
        # This replicates the behavior of KnapsackEnv._get_observation
        scaled_weights = raw_weights / cfg.ml.generation.max_weight
        scaled_values = raw_values / cfg.ml.generation.max_value
        scaled_capacity = raw_capacity / cfg.ml.generation.max_weight

        # 1c. Padding
        items_tensor = torch.stack((scaled_weights, scaled_values), dim=2)
        target_len = self.env.observation_space['items'].shape[0]
        current_len = items_tensor.shape[1]
        pad_len = 0
        if current_len < target_len:
            pad_len = target_len - current_len
            items_tensor = F.pad(items_tensor, (0, 0, 0, pad_len), 'constant', 0)
            mask_tensor = F.pad(mask_tensor, (0, pad_len), 'constant', 0)

        # 1d. Normalization (now on correctly scaled data)
        # Note: We pass the scaled_capacity to the normalizer, NOT the raw one.
        obs_to_normalize = { 'items': items_tensor.cpu().numpy(), 'capacity': scaled_capacity.unsqueeze(1).cpu().numpy() }
        normalized_obs_tensors = self.env.normalize_obs(obs_to_normalize)
        
        obs_for_features = {
            'items': torch.from_numpy(normalized_obs_tensors['items']).to(self.model.device),
            'capacity': torch.from_numpy(normalized_obs_tensors['capacity']).to(self.model.device),
            'mask': mask_tensor
        }
        
        padded_raw_weights = F.pad(raw_weights, (0, pad_len), 'constant', 0) if pad_len > 0 else raw_weights

        # --- 2. Model Inference ---
        start_time = time.perf_counter()
        with torch.no_grad():
            action_indices_tensor = self.model.policy.decode_batch(
                obs_for_features, padded_raw_weights, raw_capacity, deterministic=self.is_deterministic
            )
        end_time = time.perf_counter()
        avg_time_per_instance = (end_time - start_time) / raw_weights.shape[0]

        # --- 3. Post-processing ---
        batch_results = []
        for i in range(raw_weights_unsorted.shape[0]):
            n = batch_data['n'][i].item()
            solution_indices_sorted = action_indices_tensor[i].tolist()
            
            final_value, final_weight = 0, 0
            instance_capacity = raw_capacity[i].item()
            instance_weights_sorted = raw_weights[i]
            instance_values_sorted = raw_values[i]
            
            packed_items = set()
            for idx in solution_indices_sorted:
                if idx < n and idx not in packed_items:
                    if final_weight + instance_weights_sorted[idx] <= instance_capacity:
                        final_weight += instance_weights_sorted[idx]
                        final_value += instance_values_sorted[idx]
                        packed_items.add(idx)

            batch_results.append({"value": final_value, "time": avg_time_per_instance})
        
        return batch_results