# src/solvers/ml/dnn_solver.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import time
import logging
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from typing import Dict, Any, Type
from types import SimpleNamespace
from .feature_extractor import extract_features_from_instance

from src.solvers.interface import SolverInterface
from .dnn_model import KnapsackDNN
from .data_loader import PreprocessedKnapsackDataset

logger = logging.getLogger(__name__)

class DNNSolver(SolverInterface):
    """
    A full-featured solver for the Knapsack Problem using a Deep Neural Network.
    """
    def __init__(self, config: SimpleNamespace, device: str):
        """
        Initializes the DNNSolver using a dedicated config object and device string.
        """
        super().__init__(config)
        self.name = "DNN"
        self.device = device
        
        self.model = KnapsackDNN(
            input_size=self.config.hyperparams.input_size,
            config=self.config
        ).to(self.device)
        logger.info(f"{self.name} Solver initialized on device: {self.device}")

    def train(self):
        """
        Handles the training loop using pre-processed data for maximum speed.
        """
        logger.info(f"--- Starting Training for {self.name} using pre-processed data ---")
        
        # --- 1. Setup ---
        train_cfg = self.config.training
        paths = self._get_run_paths()
        
        os.makedirs(os.path.dirname(paths['model_file']), exist_ok=True)
        os.makedirs(os.path.dirname(paths['plot_file_loss']), exist_ok=True)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        
        # --- 2. Data Loading ---
        from src.utils.config_loader import cfg # Local import for paths
        train_dataset = PreprocessedKnapsackDataset(os.path.join(cfg.paths.data, "processed_training.pt"))
        val_dataset = PreprocessedKnapsackDataset(os.path.join(cfg.paths.data, "processed_validation.pt"))

        if not train_dataset or not val_dataset:
            logger.error("Could not load pre-processed datasets or datasets are empty. Aborting training.")
            logger.error("Please run 'preprocess_data.py' successfully first.")
            return

        train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size)

        # --- 3. Training Loop ---
        best_val_loss = float('inf')
        history = []
        total_epochs = train_cfg.total_epochs
        
        logger.info(f"Starting training for {total_epochs} epochs...")
        for epoch in range(total_epochs):
            self.model.train()
            running_train_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} Training", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
            
            self.model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    running_val_loss += criterion(outputs, labels).item()
            
            avg_train_loss = running_train_loss / len(train_loader)
            avg_val_loss = running_val_loss / len(val_loader)
            history.append({'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})
            scheduler.step(avg_val_loss)
            
            logger.info(f"Epoch {epoch+1}/{total_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), paths['model_file'])
                logger.info(f"  -> New best model saved to {paths['model_file']} (Val Loss: {best_val_loss:.6f})")

        # --- 4. Finalizing ---
        self._plot_loss_curve(pd.DataFrame(history), paths['plot_file_loss'])
        logger.info(f"--- Finished Training. Best validation loss: {best_val_loss:.6f} ---")
    
    def solve(self, instance_path: str) -> Dict[str, Any]:
        """
        Uses the trained neural network to predict the optimal value for a given instance file.
        This method now has its own independent data processing pipeline.
        """
        model_path = self._get_run_paths()["model_file"]
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        
        # Load model state only once if not already in eval mode
        if self.model.training:
            logger.info(f"Loading model state from {model_path} for evaluation.")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

        # 1. Prepare the input data using the dedicated feature extractor
        features_tensor = extract_features_from_instance(instance_path)
        
        if features_tensor is None:
            logger.error(f"Could not process features for {instance_path}. Skipping.")
            return {"value": -1, "time": 0, "solution": []}
            
        # Add batch dimension and send to device
        features_tensor = features_tensor.unsqueeze(0).to(self.device)

        # 2. Get model prediction and measure time
        start_time = time.perf_counter()
        with torch.no_grad():
            predicted_normalized_value = self.model(features_tensor)
        end_time = time.perf_counter()

        # 3. De-normalize the prediction
        predicted_value = predicted_normalized_value.item() * self.config.hyperparams.target_scale_factor
        
        return {
            "value": predicted_value,
            "time": end_time - start_time,
            "solution": [] # Note: This model predicts value, not the item set.
        }

    def _get_run_paths(self) -> dict:
        """A helper method to generate paths for the current run based on its config."""
        from src.utils.config_loader import cfg # Local import for path roots is fine here
        
        gen_cfg = self.config.generation
        train_cfg = self.config.training
        # Use the single total_epochs value from config for a consistent run name
        run_name = f"dnn_n{gen_cfg.start_n}-{gen_cfg.end_n}_epochs{train_cfg.total_epochs}"
        
        return {
            "model_file": os.path.join(cfg.paths.models, f"{run_name}.pth"),
            "plot_file_loss": os.path.join(cfg.paths.plots, f"{run_name}_loss_curve.png"),
        }
        
    def _plot_loss_curve(self, history_df: pd.DataFrame, save_path: str):
        """Helper function to plot and save the loss curve using seaborn."""
        if history_df.empty:
            logger.warning("No training history to plot.")
            return

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(15, 7))
        
        sns.lineplot(data=history_df, x='epoch', y='train_loss', label='Training Loss')
        sns.lineplot(data=history_df, x='epoch', y='val_loss', label='Validation Loss')
        
        gen_cfg = self.config.generation
        train_cfg = self.config.training
        n_range = range(gen_cfg.start_n, gen_cfg.end_n + 1, gen_cfg.step_n)

        # Draw vertical lines indicating change in 'n'
        epochs_per_n_step = train_cfg.epochs_per_n * len(n_range)
        # This curriculum logic seems off, let's simplify based on total epochs
        # A simpler plot would not have the n markers unless training is truly sequential per n
        
        plt.title('Training & Validation Loss Curve', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Loss curve plot saved to {save_path}")
