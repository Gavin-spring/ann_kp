# src/solvers/ml/rl_solver.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import os
import time
import numpy as np

from src.solvers.interface import SolverInterface
from src.utils.generator import load_instance_from_file
from .data_loader import PreprocessedKnapsackDataset

try:
    from .rl_model import PointerNetwork
except ImportError:
    print("Error: Please ensure the file 'rl_model.py' exists under 'src/solvers/ml/',")
    print("and it contains the 'PointerNetwork' class.")
    exit()

logger = logging.getLogger(__name__)

class RLSolver(SolverInterface):
    """
    A reinforcement learning-based solver for the knapsack problem using a pointer network.
    Implements the existing project interface.
    """
    def __init__(self, config, device, model_path=None):
        super().__init__(config)
        self.name = "PointerNet RL"
        self.device = device

        # 1. Initialize the model (Actor)
        self.model = PointerNetwork(
            embedding_dim=self.config.hyperparams.embedding_dim,
            hidden_dim=self.config.hyperparams.hidden_dim,
            max_decoding_len=self.config.hyperparams.max_n, # Max solution length is number of items
            terminating_symbol=None, # Not needed for the knapsack problem
            n_glimpses=self.config.hyperparams.n_glimpses,
            tanh_exploration=self.config.hyperparams.tanh_exploration,
            use_tanh=self.config.hyperparams.use_tanh,
            beam_size=1, # Use greedy decoding during inference
            use_cuda=True if self.device == 'cuda' else False
        ).to(self.device)
        
        # Load pretrained model if provided
        if model_path:
            if os.path.exists(model_path):
                logger.info(f"Loading pre-trained RL model for evaluation: {model_path}")
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                raise FileNotFoundError(f"Model file not found for RLSolver: {model_path}")
        
        self.model.eval() # Set to evaluation mode by default
        logger.info(f"{self.name} solver initialized on device {self.device}")

    def train(self, model_save_path: str, plot_save_path: str):
        """
        Full training pipeline for the RL model.
        """
        logger.info(f"--- Starting Reinforcement Learning Training for {self.name} ---")
        
        # 1. Setup optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.training.learning_rate)
        # You can also add a learning rate scheduler here

        # 2. Prepare data loader
        # RL does not require precomputed labels, but we can reuse preprocessed features for speed
        # For simplicity, we assume raw data is loaded directly
        from src.utils.config_loader import cfg # Local import
        train_dir = cfg.paths.data_training
        val_dir = cfg.paths.data_validation
        
        # Define dataset class inline for simplicity
        class RawKnapsackDataset(torch.utils.data.Dataset):
            def __init__(self, data_dir):
                self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
            def __len__(self):
                return len(self.files)
            def __getitem__(self, idx):
                instance_path = self.files[idx]
                weights, values, capacity = load_instance_from_file(instance_path)
                # Return as dictionary for easier handling
                return {'weights': torch.tensor(weights, dtype=torch.float32),
                        'values': torch.tensor(values, dtype=torch.float32),
                        'capacity': torch.tensor(capacity, dtype=torch.float32)}

        train_dataset = RawKnapsackDataset(train_dir)
        train_loader = DataLoader(train_dataset, batch_size=self.config.training.batch_size, shuffle=True)

        # 3. Training loop (adapted from pemami4911/trainer.py)
        best_val_reward = -float('inf')
        baseline = torch.zeros(1, device=self.device) # Exponential moving average baseline
        beta = 0.8 # Baseline smoothing factor

        for epoch in range(self.config.training.total_epochs):
            self.model.train()
            total_reward = 0.0
            
            for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.training.total_epochs}"):
                # Move batch data to target device
                weights = batch_data['weights'].to(self.device)
                values = batch_data['values'].to(self.device)
                # capacity = batch_data['capacity'].to(self.device) # Used in reward calculation

                # Reshape input into shape expected by model: (batch, input_dim, seq_len)
                # Here input_dim = 2, representing (weight, value)
                inputs = torch.stack([weights, values], dim=1)

                # Forward pass
                probs, action_idxs = self.model(inputs) # PointerNetwork returns log probabilities and action indices

                # Calculate rewards
                rewards = self._calculate_reward(action_idxs, batch_data)
                rewards = rewards.to(self.device)

                # Update baseline
                if baseline.sum() == 0: # First iteration
                    baseline = rewards.mean()
                else:
                    baseline = baseline * beta + (1. - beta) * rewards.mean()

                # Compute advantage
                advantage = rewards - baseline.detach() # detach is crucial

                # Compute REINFORCE loss
                log_probs = 0
                for prob in probs:
                    log_probs += torch.log(prob)
                
                loss = -(log_probs * advantage).mean()

                # Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_reward += rewards.mean().item()

            avg_epoch_reward = total_reward / len(train_loader)
            logger.info(f"Epoch {epoch+1}, Avg Reward: {avg_epoch_reward:.4f}")

            # Simple validation and saving logic
            if avg_epoch_reward > best_val_reward:
                best_val_reward = avg_epoch_reward
                torch.save(self.model.state_dict(), model_save_path)
                logger.info(f"  -> New best model saved to {model_save_path} (Avg Reward: {best_val_reward:.4f})")

    def _calculate_reward(self, selected_indices, instance_batch):
        """
        Given the item selection sequence output by the model, compute the total value as the reward.
        This function handles batched input.
        """
        batch_size = selected_indices[0].size(0)
        batch_rewards = []

        for i in range(batch_size):
            capacity = instance_batch['capacity'][i].item()
            weights = instance_batch['weights'][i]
            values = instance_batch['values'][i]

            current_weight = 0.0
            current_value = 0.0
            
            # Convert output indices to Python list
            item_priority_list = [idx[i].item() for idx in selected_indices]
            
            packed_items = set() # Prevent duplicate selections
            
            for item_idx in item_priority_list:
                if item_idx in packed_items:
                    continue # Skip already added items
                
                if current_weight + weights[item_idx] <= capacity:
                    current_weight += weights[item_idx]
                    current_value += values[item_idx]
                    packed_items.add(item_idx)
            
            batch_rewards.append(current_value)

        return torch.tensor(batch_rewards, dtype=torch.float32)

    def solve(self, instance_path: str):
        """
        Solve a single knapsack problem instance using the trained RL model.
        """
        self.model.eval() # Ensure model is in evaluation mode
        
        # 1. Load and prepare data
        weights, values, capacity = load_instance_from_file(instance_path)
        weights_t = torch.tensor(weights, dtype=torch.float32).unsqueeze(0) # Add batch dimension
        values_t = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
        
        input_tensor = torch.stack([weights_t, values_t], dim=1).to(self.device)

        # 2. Model inference (using greedy decoding)
        start_time = time.perf_counter()
        with torch.no_grad():
            # Assume model returns both log probabilities and action indices
            _, action_idxs = self.model(input_tensor) 
        end_time = time.perf_counter()
        
        # 3. Compute final solution
        # `_calculate_reward` can be reused, it returns a tensor of rewards
        instance_data = {'weights': weights_t, 'values': values_t, 'capacity': torch.tensor([capacity])}
        final_value = self._calculate_reward(action_idxs, instance_data).item()
        
        # Extract the list of item indices
        item_indices = [idx[0].item() for idx in action_idxs]
        solution_mask = [0] * len(weights)
        
        # Determine final packing based on selection order
        final_weight = 0
        final_packed_indices = set()
        for idx in item_indices:
            if idx in final_packed_indices:
                continue
            if final_weight + weights[idx] <= capacity:
                final_weight += weights[idx]
                final_packed_indices.add(idx)
        
        for idx in final_packed_indices:
            solution_mask[idx] = 1

        return {
            "value": final_value,
            "time": end_time - start_time,
            "solution": solution_mask
        }