# src/solvers/ml/dnn_solver.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import time
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.solvers.interface import SolverInterface 
from .dnn_model import KnapsackDNN # Import the model architecture
from .data_loader import get_dataset_for_n, KnapsackDataset

logger = logging.getLogger(__name__)

class DNNSolver(SolverInterface):
    """
    A full-featured solver for the Knapsack Problem using a Deep Neural Network.
    This class handles training, evaluation, and solving.
    """
    def __init__(self, config=None):
        # Import cfg here, inside the method, instead of at the top of the file.
        from src.utils.config_loader import cfg

        # We pass the relevant part of the global config to the solver
        super().__init__(config=cfg.ml.dnn)
        self.name = "DNN Solver"
        self.device = cfg.ml.device
        
        # Initialize the model using hyperparameters from the config
        self.model = KnapsackDNN(
            input_size=self.config.hyperparams.input_size,
            config=self.config
        ).to(self.device)
        logger.info(f"{self.name} initialized on device: {self.device}")

    def train(self):
        """
        Handles the complete training and validation loop for the DNN model
        based on the settings in the config file.
        """
        logger.info(f"--- Starting Training for {self.name} ---")
        
        # Get configs
        train_cfg = self.config.training
        gen_cfg = self.config.generation
        paths = self.config.get_paths() # Use the path generation method
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(paths['model_file']), exist_ok=True)
        os.makedirs(os.path.dirname(paths['plot_file_loss']), exist_ok=True)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        best_val_loss = float('inf')
        train_losses, val_losses = [], []
        
        # Use the baseline algorithm from config to calculate labels
        # We need to import cfg here too if the __init__ is not called, but it is.
        from src.utils.config_loader import cfg
        baseline_solver = cfg.classic_solvers.baseline_algorithm

        logger.info("--- Starting Curriculum Training ---")
        for n in range(gen_cfg.start_n, gen_cfg.end_n + gen_cfg.step_n, gen_cfg.step_n):
            logger.info(f"----- Preparing data for n={n} -----")
            # This returns a list of dictionaries
            dataset_n = get_dataset_for_n(n_items=n, data_dir=cfg.paths.data_training)
            if not dataset_n:
                continue

            # Calculate the optimal solution (label) for each instance
            for item in tqdm(dataset_n, desc=f"Solving n={n} for labels"):
                weights, values, capacity = item["raw_instance"]
                optimal_value = baseline_solver(weights=weights, values=values, capacity=capacity)
                item["label"] = optimal_value / self.config.hyperparams.target_scale_factor
            
            # Create PyTorch dataset and loaders
            train_size = int(0.8 * len(dataset_n))
            val_size = len(dataset_n) - train_size
            train_data, val_data = random_split(dataset_n, [train_size, val_size])
            
            train_loader = DataLoader(KnapsackDataset(train_data), batch_size=train_cfg.batch_size, shuffle=True)
            val_loader = DataLoader(KnapsackDataset(val_data), batch_size=train_cfg.batch_size)
            
            # Training loop for this 'n'
            for epoch in range(train_cfg.epochs_per_n):
                self.model.train()
                running_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        val_loss += criterion(outputs, labels).item()
                
                avg_train_loss = running_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                scheduler.step(avg_val_loss)
                
                logger.info(f"Overall Epoch {len(val_losses)}, n={n}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(self.model.state_dict(), paths['model_file'])
                    logger.info(f"  -> New best model saved to {paths['model_file']}")

        # --- Plotting ---
        self._plot_loss_curve(train_losses, val_losses, paths['plot_file_loss'])
        logger.info("--- Finished Training ---")

    def solve(self, instance_path: str) -> dict[str, any]:
        """
        Uses the trained neural network to predict the optimal value for a given instance file.
        """
        logger.debug(f"Solving instance {instance_path} with {self.name}...")
        
        # 1. Load the model if not already loaded
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        model_path = self.config.get_paths()["model_file"]
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # 2. Prepare the input data
        # To solve a single instance, we need a dedicated function.
        # This part needs further refinement to process a single file efficiently.
        # For now, this is a placeholder to show the logic.
        dataset = get_dataset_for_n(n_items=-1, data_dir=os.path.dirname(instance_path))
        raw_instance_data = None
        for item in dataset:
            # This is not a robust way to find the instance, but serves as an example
            raw_instance_data = item
            break
        
        if raw_instance_data is None:
            return {"value": -1, "time": 0, "solution": []}
            
        features_tensor = torch.tensor(raw_instance_data["features"]).unsqueeze(0).to(self.device)

        # 3. Get model prediction and measure time
        start_time = time.perf_counter()
        with torch.no_grad():
            predicted_normalized_value = self.model(features_tensor)
        end_time = time.perf_counter()

        # 4. De-normalize the prediction
        predicted_value = predicted_normalized_value.item() * self.config.hyperparams.target_scale_factor
        
        return {
            "value": predicted_value,
            "time": end_time - start_time,
            "solution": [] # Note: This model predicts value, not the item set.
        }
        
    def _plot_loss_curve(self, train_losses, val_losses, save_path):
        """Helper function to plot and save the loss curve."""
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        
        gen_cfg = self.config.generation
        train_cfg = self.config.training
        num_n_steps = (gen_cfg.end_n - gen_cfg.start_n) // gen_cfg.step_n + 1
        for i in range(1, num_n_steps):
            epoch_marker = i * train_cfg.epochs_per_n
            plt.axvline(x=epoch_marker, color='r', linestyle='--', linewidth=0.8)
            new_n = gen_cfg.start_n + i * gen_cfg.step_n
            plt.text(epoch_marker + 1, max(train_losses) * 0.9, f'n={new_n}', color='r', rotation=90)

        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Loss curve plot saved to {save_path}")
