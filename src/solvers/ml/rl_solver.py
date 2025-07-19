# src/solvers/ml/rl_solver.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import logging
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Tuple

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

def knapsack_collate_fn(batch):
    """
    Self defined batch collate function for knapsack problem.
    This function handles padding of weights and values, and creates attention masks.
    """
    # 1. extract weights, values, capacity, and n from the batch
    weights_list = [item['weights'] for item in batch]
    values_list = [item['values'] for item in batch]
    capacity_list = [item['capacity'] for item in batch]
    n_list = [item['n'] for item in batch]
    filenames = [item['filename'] for item in batch]

    # 2. get the maximum length of weights in the batch
    max_len = max(len(w) for w in weights_list)

    padded_weights = []
    padded_values = []
    attention_masks = []

    # 3. pad and create attention masks in batch
    for i in range(len(batch)):
        w = weights_list[i]
        v = values_list[i]
        current_len = len(w)
        
        # Calculate padding length
        padding_len = max_len - current_len

        # use F.pad to pad(right side) the weights and values``
        # format: F.pad(input, (pad_left, pad_right), mode, value)
        padded_w = F.pad(w, (0, padding_len), 'constant', 0)
        padded_v = F.pad(v, (0, padding_len), 'constant', 0)
        padded_weights.append(padded_w)
        padded_values.append(padded_v)

        # create attention mask(1s for actual data, 0s for padding)
        mask = torch.cat([torch.ones(current_len), torch.zeros(padding_len)])
        attention_masks.append(mask)

    # 4. convert lists to tensors
    return {
        'weights': torch.stack(padded_weights),
        'values': torch.stack(padded_values),
        'capacity': torch.stack(capacity_list),
        'n': torch.stack(n_list),
        'attention_mask': torch.stack(attention_masks).bool(), # Convert to boolean tensor
        'filenames': filenames
    }

class RawKnapsackDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        instance_path = self.files[idx]
        weights, values, capacity = load_instance_from_file(instance_path)
        
        # return raw data without padding
        return {
            'weights': torch.tensor(weights, dtype=torch.float32),
            'values': torch.tensor(values, dtype=torch.float32),
            'capacity': torch.tensor(capacity, dtype=torch.float32),
            'n': torch.tensor(len(weights), dtype=torch.int32),
            'filename': os.path.basename(instance_path)
        }

class RLSolver(SolverInterface):
    """
    A reinforcement learning-based solver for the knapsack problem using a pointer network.
    Implements the existing project interface.
    """
    def __init__(self, config, device, model_path=None, compile_model=True):
        super().__init__(config)
        self.name = "PointerNet RL"
        self.device = device

        # 1. Initialize the model (Actor)
        self.model = PointerNetwork(
            embedding_dim=self.config.hyperparams.embedding_dim,
            hidden_dim=self.config.hyperparams.hidden_dim,
            max_decoding_len=self.config.hyperparams.max_n, # Max solution length is number of items
            n_glimpses=self.config.hyperparams.n_glimpses,
            tanh_exploration=self.config.hyperparams.tanh_exploration,
            use_tanh=self.config.hyperparams.use_tanh,
            use_cuda=True if self.device == 'cuda' else False,
            input_dim=2 # Two inputs: weights and values
        ).to(self.device)

        # Compile the model for potential speed-up
        if compile_model:
            try:
                self.model = torch.compile(self.model)
                logger.info("Successfully compiled the model with torch.compile() for potential speed-up.")
            except Exception as e:
                logger.warning(f"Model compilation failed, proceeding without it. Reason: {e}")

        # Load pretrained model if provided
        if model_path:
            if os.path.exists(model_path):
                logger.info(f"Loading pre-trained RL model for evaluation: {model_path}")
                
                # Check if the state_dict contains '_orig_mod.' prefix
                # This is a common prefix when using torch.compile
                state_dict = torch.load(model_path, map_location=self.device)
                if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
                    logger.info("Found compiled model state_dict, stripping '_orig_mod.' prefix...")
                    new_state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
                    self.model.load_state_dict(new_state_dict)
                else:
                    self.model.load_state_dict(state_dict)
            else:
                raise FileNotFoundError(f"Model file not found for RLSolver: {model_path}")
        
        self.model.eval() # Set to evaluation mode by default
        logger.info(f"{self.name} solver initialized on device {self.device}")

    # --- Main Orchestration Method ---
    def train(self, model_save_path: str, plot_save_path: str):
        """
        Orchestrates the entire training process, now passing the scheduler
        to the epoch training function.
        """
        logger.info(f"--- Starting RL Training for {self.name} ---")
        
        # 1. Setup training components
        optimizer, scheduler = self._setup_training() # Now receives the scheduler

        # 2. Prepare data loaders
        train_loader, val_loader = self._prepare_dataloaders()
        if not train_loader or not val_loader:
            return

        # 3. Main training loop
        best_val_reward = -float('inf')
        history = []
        total_epochs = self.config.training.total_epochs
        
        logger.info(f"Starting training for {total_epochs} epochs...")
        for epoch in range(total_epochs):
            # Pass the scheduler to the training function
            train_reward, final_baseline = self._train_one_epoch(
                train_loader, optimizer, scheduler
            )
            
            val_reward = self._validate_one_epoch(val_loader)
            
            history.append({
                'epoch': epoch + 1, 
                'train_reward': train_reward, 
                'val_reward': val_reward,
                'baseline': final_baseline.item()
            })
            
            # Note: scheduler.step() is now called inside _train_one_epoch after each batch,
            # so we don't call it here. If using ReduceLROnPlateau, you would call it here:
            # scheduler.step(val_reward)
            
            logger.info(f"Epoch {epoch+1}/{total_epochs}, Train Reward: {train_reward:.4f}, Val Reward: {val_reward:.4f}, Baseline: {final_baseline.item():.4f}")

            if val_reward > best_val_reward:
                best_val_reward = val_reward
                torch.save(self.model.state_dict(), model_save_path)
                logger.info(f"  -> New best model saved to {model_save_path} (Val Reward: {best_val_reward:.4f})")
        
        # 4. Finalize and plot results
        self._plot_reward_curve(pd.DataFrame(history), plot_save_path)
        logger.info(f"--- Finished Training. Best validation reward: {best_val_reward:.4f} ---")

    # --- Helper Methods ---
    def _setup_training(self):
        """
        Initializes and returns the optimizer and the learning rate scheduler
        by reading parameters from the config.
        """
        train_cfg = self.config.training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=train_cfg.learning_rate)
        
        # Create a learning rate scheduler based on the config
        # This scheduler decreases the LR by a factor of 'lr_decay_rate'
        # every 'lr_decay_step' steps.
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(range(
                train_cfg.lr_decay_step,
                train_cfg.lr_decay_step * 1000, # A large upper bound for milestones
                train_cfg.lr_decay_step
            )),
            gamma=train_cfg.lr_decay_rate
        )
        
        logger.info("Optimizer and LR Scheduler have been set up.")
        return optimizer, scheduler

    def _prepare_dataloaders(self):
        """Loads raw data and prepares PyTorch DataLoaders using a collate_fn."""
        from src.utils.config_loader import cfg
        from src.utils.generator import load_instance_from_file

        logger.info("Preparing data loaders for RL training...")
        try:
            train_dir = cfg.paths.data_training
            val_dir = cfg.paths.data_validation

            train_dataset = RawKnapsackDataset(train_dir)
            val_dataset = RawKnapsackDataset(val_dir)
            
            # --- Use the knapsack_collate_fn defined above ---
            train_loader = DataLoader(train_dataset, batch_size=self.config.training.batch_size, shuffle=True, collate_fn=knapsack_collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=self.config.training.batch_size, collate_fn=knapsack_collate_fn)
            
            return train_loader, val_loader
        except Exception as e:
            logger.error(f"Failed to prepare dataloaders for RL training. Error: {e}", exc_info=True)
            return None, None

    def _train_one_epoch(self, data_loader: DataLoader, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler) -> Tuple[float, torch.Tensor]:
        """
        Executes a single training epoch for the RL model,
        now including gradient clipping and scheduler stepping.
        """
        self.model.train() # Set model to training mode
        self.model.decoder.decode_type = 'stochastic'

        total_epoch_reward = 0.0
        # Use baseline_beta from the config
        beta = self.config.training.baseline_beta
        # Initialize baseline on the correct device
        baseline = torch.zeros(1, device=self.device)

        # 1. Use a GradScaler for mixed precision training
        # AMP (Automatic Mixed Precision) taking adventages of Tensor Cores is a feature in PyTorch that allows you to use lower precision (float16) for training,
        # For NVIDIA GPU（RTX 20XX），Tensor Cores dealing with float16 as multiple times faster than float32, and can train larger models.
        # For Pointer Networks and Transformer models, Computationally intensive operations (such as matrix multiplication nn.Linear, torch.bmm) 
        # are typically stable and efficient under float16, and they account for more than 95% of the model's computation time. 
        # Numerically sensitive operations  (such as softmax, log in loss functions, or activation functions with large input ranges) are only a minority . 

        scaler = torch.amp.GradScaler()
        for i, batch_data in enumerate(tqdm(data_loader, desc="Training", leave=False)):
            
            weights = batch_data['weights'].to(self.device)
            values = batch_data['values'].to(self.device)
            capacity = batch_data['capacity'].to(self.device)
            attention_mask = batch_data['attention_mask'].to(self.device)
            
            inputs = torch.stack([weights, values], dim=1)            
            inputs_for_model = inputs.permute(0, 2, 1) # Shape: (batch_size, feature_num, max_n)
            
            optimizer.zero_grad()
            
            # 2. Perform forward pass with mixed precision
            with torch.amp.autocast(device_type='cuda'):
                probs_list, action_idxs = self.model(inputs_for_model, capacity, attention_mask)
                rewards = self._calculate_reward(action_idxs, batch_data).to(self.device)

                # 3. Calculate the baseline using an exponential moving average
                if i == 0 and baseline.sum() == 0:
                    baseline = rewards.mean()
                else:
                    baseline = baseline * beta + (1. - beta) * rewards.mean()
                    
                # 4. Calculate the advantage
                advantage = rewards - baseline.detach()

                # Calculate the loss
                log_probs_of_actions = 0
                for prob_dist, action_idx in zip(probs_list, action_idxs):
                    log_prob = torch.log(prob_dist.gather(1, action_idx.unsqueeze(1)).squeeze(1))
                    log_prob[log_prob < -1000] = 0.0
                    log_probs_of_actions += log_prob

                loss = -(log_probs_of_actions * advantage).mean()

            # 5. Scale the loss for mixed precision
            scaler.scale(loss).backward()

            # optimizer.zero_grad() # used when not using AMP
            # loss.backward() # used when not using AMP

            # --- APPLY GRADIENT CLIPPING ---
            # Use the max_grad_norm value from the config
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm
            )

            # optimizer.step() # used when not using AMP

            # --- STEP THE LEARNING RATE SCHEDULER ---
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            total_epoch_reward += rewards.mean().item()
        
        return total_epoch_reward / len(data_loader), baseline

    def _validate_one_epoch(self, data_loader: DataLoader) -> float:
        """Executes a single validation epoch."""
        self.model.eval() # Set model to evaluation mode
        self.model.decoder.decode_type = 'greedy' # Use greedy decoding for validation

        total_val_reward = 0.0
        with torch.no_grad():
            for batch_data in tqdm(data_loader, desc="Validating", leave=False): # I've added tqdm here for a progress bar
                weights = batch_data['weights'].to(self.device)
                values = batch_data['values'].to(self.device)
                capacity = batch_data['capacity'].to(self.device)
                attention_mask = batch_data['attention_mask'].to(self.device)
                
                inputs = torch.stack([weights, values], dim=1)
                # This permute should be consistent with the one in training
                inputs_for_model = inputs.permute(0, 2, 1) 
                
                _ , action_idxs = self.model(inputs_for_model, capacity, attention_mask)                

                rewards = self._calculate_reward(action_idxs, batch_data)
                total_val_reward += rewards.mean().item()
                
        return total_val_reward / len(data_loader)
        
    def _plot_reward_curve(self, history_df: pd.DataFrame, save_path: str):
        """Helper function to plot and save the reward curve."""
        if history_df.empty:
            logger.warning("No training history to plot.")
            return

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(15, 7))
        
        sns.lineplot(data=history_df, x='epoch', y='train_reward', label='Training Average Reward')
        sns.lineplot(data=history_df, x='epoch', y='val_reward', label='Validation Average Reward')
        # Add a line for the baseline value
        if 'baseline' in history_df.columns:
            sns.lineplot(data=history_df, x='epoch', y='baseline', label='Reward Baseline (EMA)', linestyle='--', color='gray')
        
        plt.title('Training & Validation Reward Curve', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Average Reward (Total Value)', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Reward curve plot saved to {save_path}")

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
        n_items = len(weights)

        # Create tensors with a batch dimension of 1
        weights_t = torch.tensor(weights, dtype=torch.float32).unsqueeze(0) # Add batch dimension
        values_t = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
        capacity_t = torch.tensor([capacity], dtype=torch.float32).to(self.device)

        # Prepare input tensor consistent with training (shape: [1, n, 2])
        input_tensor_original_shape = torch.stack([weights_t, values_t], dim=1).to(self.device)
        input_tensor = input_tensor_original_shape.permute(0, 2, 1) # Shape: (1, feature_num, max_n)

        # 2. create attention_mask
        # For a single instance, there is no padding, so the mask is all True.
        # Shape should be [batch_size, seq_len], which is [1, n_items] here.
        attention_mask = torch.ones(1, n_items, dtype=torch.bool, device=self.device)

        # 3. Model inference (using greedy decoding)
        start_time = time.perf_counter()
        with torch.no_grad():
            # Assume model returns both log probabilities and action indices
            _, action_idxs = self.model(input_tensor, capacity_t, attention_mask) 
        end_time = time.perf_counter()
        
        # 4. Compute final solution
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

    def solve_batch(self, batch_data):
        """
        Solves a batch of knapsack instances at once.
        """
        self.model.eval()
        self.model.decoder.decode_type = 'greedy'

        # 1. move batch data to device
        weights = batch_data['weights'].to(self.device)
        values = batch_data['values'].to(self.device)
        capacity = batch_data['capacity'].to(self.device)
        attention_mask = batch_data['attention_mask'].to(self.device)
        
        batch_size = weights.size(0)

        # prepare inputs for the model
        inputs_stacked = torch.stack([weights, values], dim=1)
        inputs_for_model = inputs_stacked.permute(0, 2, 1)

        # 2. perform model inference for the entire batch
        start_time = time.perf_counter()
        with torch.no_grad():
            _, action_idxs = self.model(inputs_for_model, capacity, attention_mask)
        end_time = time.perf_counter()

        # Calculate average time per instance
        avg_time_per_instance = (end_time - start_time) / batch_size

        # 3. Calculate rewards for the batch
        rewards = self._calculate_reward(action_idxs, batch_data)

        # 4. Prepare the final results for each instance in the batch
        batch_results = []
        for i in range(batch_size):
            # get raw instance data for the i-th instance
            original_n = batch_data['n'][i].item()
            instance_weights = batch_data['weights'][i][:original_n].tolist()
            
            # extract solution indices from action_idxs
            item_indices = [idx[i].item() for idx in action_idxs]
            solution_mask = [0] * original_n
            
            final_weight = 0
            final_packed_indices = set()
            for idx in item_indices:
                # check if idx is within the original n
                if idx >= original_n:
                    continue
                if idx in final_packed_indices:
                    continue
                if final_weight + instance_weights[idx] <= batch_data['capacity'][i].item():
                    final_weight += instance_weights[idx]
                    final_packed_indices.add(idx)
            
            for idx in final_packed_indices:
                solution_mask[idx] = 1

            batch_results.append({
                "value": rewards[i].item(),
                "time": avg_time_per_instance,
                "solution": solution_mask
            })
        
        return batch_results