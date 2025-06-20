# File: train.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import math
# Local imports
import dnn_config as cfg
from model import KnapsackPredictor
from data_loader import load_knapsack_dataset_from_files, KnapsackDataset

def main():
    """Main function to run the curriculum training process."""
    print(f"Using device: {cfg.DEVICE}")

    model = KnapsackPredictor(cfg.INPUT_SIZE).to(cfg.DEVICE)
    print("\nDNN Model defined:")
    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

    MODEL_SAVE_PATH, PLOT_SAVE_PATH = cfg.get_model_and_plot_names()

    best_val_loss = math.inf
    train_losses, val_losses = [], []

    print(f"Model will be saved to: {MODEL_SAVE_PATH}")
    print("\n--- Starting Curriculum Training ---")

    for n in range(cfg.START_N, cfg.END_N + cfg.STEP_N, cfg.STEP_N):
        print(f"\n----- Training on problems of size n={n} -----")
        
        knapsack_data = load_knapsack_dataset_from_files(
            n_items=n, data_dir=cfg.MODEL_TRAIN_CASES, max_n=cfg.MAX_N
        )
        if not knapsack_data:
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
            
            total_epoch_num = len(val_losses)
            print(f"Overall Epoch {total_epoch_num}, n={n}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"  -> New best model saved!")

    # --- Plotting ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_SAVE_PATH)
    plt.close()

    print(f"\nLoss curve plot saved as {PLOT_SAVE_PATH}")
    print("--- Finished Training ---")

if __name__ == '__main__':
    main()