# File: train.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import math
import logging

# Local imports
import dnn_config as cfg
from model import KnapsackPredictor
from data_loader import load_knapsack_dataset_from_files, KnapsackDataset
from model_logger import setup_logger

def main():
    """Main function to run the curriculum training process."""
    # --- Logger Setup ---
    # Call this at the very beginning.
    setup_logger('model_training')
    logger = logging.getLogger(__name__)

    logger.info(f"Using device: {cfg.DEVICE}")

    model = KnapsackPredictor(cfg.INPUT_SIZE).to(cfg.DEVICE)
    logger.info("DNN Model defined:")
    # The model structure is multi-line, so we log it line-by-line or as a single block.
    for line in str(model).split('\n'):
        logger.info(line)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    # Add a scheduler to adjust the learning rate based on validation loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    MODEL_SAVE_PATH, PLOT_SAVE_PATH = cfg.get_model_and_plot_names()

    best_val_loss = math.inf
    train_losses, val_losses = [], []

    logger.info(f"Model will be saved to: {MODEL_SAVE_PATH}")
    logger.info("--- Starting Curriculum Training ---")

    for n in range(cfg.START_N, cfg.END_N + cfg.STEP_N, cfg.STEP_N):
        logger.info(f"----- Training on problems of size n={n} -----")
        
        knapsack_data = load_knapsack_dataset_from_files(
            n_items=n, data_dir=cfg.MODEL_TRAIN_CASES, max_n=cfg.MAX_N
        )
        if not knapsack_data:
            logger.warning(f"No training data found for n={n}. Skipping.")
            continue

        train_size = int(0.8 * len(knapsack_data))
        val_size = len(knapsack_data) - train_size
        train_data, val_data = random_split(knapsack_data, [train_size, val_size])
        
        train_loader = DataLoader(KnapsackDataset(train_data), batch_size=cfg.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(KnapsackDataset(val_data), batch_size=cfg.BATCH_SIZE)

        for epoch in range(cfg.EPOCHS_PER_N):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item()
            
            avg_train_loss = running_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            scheduler.step(avg_val_loss) # Adjust learning rate based on validation loss
            
            total_epoch_num = len(val_losses)
            # This is detailed info, so logger.info or logger.debug are both fine.
            logger.info(f"Overall Epoch {total_epoch_num}, n={n}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                logger.info(f"  -> New best model saved!")

    # --- Plotting ---
    logger.info("Generating and saving loss curve plot...")
    plt.figure(figsize=(12, 6)) # Increased figure size for better readability
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')

    # Add vertical lines and annotations to mark when 'n' changes
    num_n_steps = (cfg.END_N - cfg.START_N) // cfg.STEP_N + 1
    for i in range(1, num_n_steps):
        epoch_marker = i * cfg.EPOCHS_PER_N
        plt.axvline(x=epoch_marker, color='r', linestyle='--', linewidth=0.8)
        # Add text annotation for 'n' value
        new_n = cfg.START_N + i * cfg.STEP_N
        plt.text(epoch_marker + 1, max(train_losses) * 0.9, f'n={new_n}', color='r', rotation=90)

    plt.title('Training and Validation Loss Over Epochs (with n Transitions)') # Updated title
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout() # Adjust plot to ensure everything fits
    plt.savefig(PLOT_SAVE_PATH)
    plt.close()

    logger.info(f"Loss curve plot saved as {PLOT_SAVE_PATH}")
    logger.info("--- Finished Training ---")
if __name__ == '__main__':
    main()