# File: ann/model_dnn_gpu.py
# -*- coding: utf-8 -*-

import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import numpy as np
import glob
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
# Local imports from the 'ann' package and root
import generator as gen
import algorithms as alg
import model_config as cfg

# --- Step 0: Define device (GPU or CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Centralized Hyperparameters for Training and Evaluation
START_N = 5
END_N = 100
STEP_N = 5
EPOCHS_PER_N = 10 # Train for 10 epochs on each 'n' before moving to the next
MAX_N = END_N # The largest 'n' determines the model's input size
INPUT_SIZE = MAX_N * 2 + 1

# --- Step 1 Load dataset from files for a specific n ---
def load_knapsack_dataset_from_files(n_items: int, data_dir: str, max_n: int):
    dataset = []
    if n_items > max_n:
        print(f"Error: n_items ({n_items}) exceeds the maximum allowed ({max_n}).")
        return None
    
    filename_pattern = f"instance_n{n_items}_*.csv"
    print(f"Loading files: '{filename_pattern}' in '{data_dir}'...")
    test_suite_path = os.path.join(data_dir, filename_pattern) 
    instance_files = glob.glob(test_suite_path)
    if not instance_files:
        print(f"Error: No .csv files found for n={n_items} in '{data_dir}'.")
        return None
    print(f"Found {len(instance_files)} instance files for n={n_items}. Processing...")
    
    # Calculate the target feature vector length based on max_n
    target_feature_len = max_n * 2 + 1
    for filepath in instance_files:
        weights, values, capacity = gen.load_instance_from_file(filepath)
        optimal_value = alg.knapsack_gurobi(weights=weights, values=values, capacity=capacity)
        # NOTE: This is the input to the DNN model
        feature_vector = np.array(weights + values + [capacity / cfg.MAX_WEIGHT], dtype=np.float32)
        # Pad the feature vector with zeros to match the target length
        padding_size = target_feature_len - len(feature_vector)
        padded_feature_vector = np.pad(feature_vector, (0, padding_size), 'constant')
        dataset.append((padded_feature_vector, np.float32(optimal_value)))
    print("Dataset loading and processing complete.")
    return dataset



# --- Step 2: Create PyTorch Dataset and DataLoader ---
class KnapsackDataset(Dataset):
    def __init__(self, data):
        self.features = [torch.tensor(item[0]) for item in data]
        self.labels = [torch.tensor(item[1]).unsqueeze(0) for item in data]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]



# --- Step 3: Define the DNN Model ---
class KnapsackPredictor(nn.Module):
    def __init__(self, input_size):
        super(KnapsackPredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x



# --- Step 4: Training Loops ---
print("\n--- Step 4: Training the DNN Model with Curriculum Learning ---")
model = KnapsackPredictor(INPUT_SIZE)
model.to(device)
print("\nDNN Model defined with fixed input size for n_max=100:")
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # regularization

total_epochs_trained = (END_N - START_N) // STEP_N * EPOCHS_PER_N + EPOCHS_PER_N
model_filename = f"model_curriculum_n{START_N}-{END_N}_totalepochs{total_epochs_trained}.pth"
os.makedirs(cfg.SAVED_MODELS_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(cfg.SAVED_MODELS_DIR, model_filename)

plot_filename = f"plot_curriculum_n{START_N}-{END_N}_totalepochs{total_epochs_trained}_losscurve.png"
os.makedirs(cfg.MODEL_PLOTS_DIR, exist_ok=True)
plot_save_path = os.path.join(cfg.MODEL_PLOTS_DIR, plot_filename)

best_val_loss = math.inf
train_losses = []
val_losses = []

print(f"Model will be saved to: {MODEL_SAVE_PATH}")
print("\n--- Starting Curriculum Training ---")

# --- The new outer loop for curriculum learning ---
for n in range(START_N, END_N + STEP_N, STEP_N):
    print(f"\n----- Training on problems of size n={n} -----")
    
    # 1. Load data for the current 'n' with padding
    knapsack_dataset = load_knapsack_dataset_from_files(n_items=n, data_dir=cfg.MODEL_TRAIN_CASES, max_n=MAX_N)
    if not knapsack_dataset:
        print(f"Warning: No data found for n={n}. Skipping.")
        continue
    
    # 2. Create new DataLoaders for this 'n'
    train_size = int(0.8 * len(knapsack_dataset))
    val_size = len(knapsack_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(knapsack_dataset, [train_size, val_size])
    train_loader = DataLoader(KnapsackDataset(train_data), batch_size=32, shuffle=True)
    val_loader = DataLoader(KnapsackDataset(val_data), batch_size=32)

    # 3. Inner loop: Train for a few epochs on this data
    for epoch in range(EPOCHS_PER_N):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        current_val_loss = val_loss / len(val_loader)
        train_losses.append(running_loss / len(train_loader))
        val_losses.append(current_val_loss)
        
        # The print statement now shows the overall progress
        total_epoch_num = len(val_losses)
        print(f"Overall Epoch {total_epoch_num}, n={n}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {current_val_loss:.4f}")

        # Check and save the best model based on validation loss
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> Found new best model! Saved to {MODEL_SAVE_PATH}")

# --- Plotting Section ---
print("--- Plotting training history ---")
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
# Save the plot to the specified path
plt.savefig(plot_save_path)
plt.close()

print(f"Loss curve plot saved as {plot_save_path}")
print("--- Finished Training ---")



# --- STEP 5: Final Evaluation on Varied-Size Test Set ---
print("\n--- Evaluating BEST Saved Model on the ENTIRE Test Set ---")

# 1. Create a new instance of the model and move it to the device
final_model = KnapsackPredictor(INPUT_SIZE)
final_model.to(device)

# 2. Load the state dictionary of the best trained model
total_epochs_trained = (END_N - START_N) // STEP_N * EPOCHS_PER_N + EPOCHS_PER_N
model_filename = f"model_curriculum_n{START_N}-{END_N}_totalepochs{total_epochs_trained}.pth"
MODEL_SAVE_PATH = os.path.join(cfg.SAVED_MODELS_DIR, model_filename)

if not os.path.exists(MODEL_SAVE_PATH):
    sys.exit(f"Model file not found at {MODEL_SAVE_PATH}. Please run training first.")
    
print(f"Loading best model from: {MODEL_SAVE_PATH}")
final_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
final_model.eval()

# 3. Loop through all test sizes, evaluate, and collect results
evaluation_results = []

for n in range(START_N, END_N + STEP_N, STEP_N):
    print(f"\n--- Evaluating for n={n} ---")
    test_dataset = load_knapsack_dataset_from_files(n_items=n, data_dir=cfg.MODEL_TEST_CASES, max_n=MAX_N)
    
    if not test_dataset:
        print(f"No test data found for n={n}. Skipping.")
        continue

    dnn_predictions = []
    gurobi_solutions = []
    dnn_times = []
    gurobi_times = []
    
    with torch.no_grad():
        for features, gurobi_solution in test_dataset:
            # DNN performance
            features_tensor = torch.tensor(features).to(device)            
            start_time = time.perf_counter()
            predicted_value = final_model(features_tensor)
            end_time = time.perf_counter()            
            dnn_predictions.append(predicted_value.item())
            dnn_times.append(end_time - start_time)

            # --- Gurobi Performance (for time comparison) ---
            gurobi_solutions.append(gurobi_solution.item())
            weights = features[:n].tolist() # Use the current 'n' for slicing
            values = features[n:n*2].tolist()
            capacity = int(features[n*2] * cfg.MAX_WEIGHT) # De-normalize using the correct index
            start_time = time.perf_counter()
            alg.knapsack_gurobi(weights=weights, values=values, capacity=capacity)
            end_time = time.perf_counter()
            gurobi_times.append(end_time - start_time)

    # Calculate metrics for the current 'n'
    dnn_predictions = np.array(dnn_predictions)
    gurobi_solutions = np.array(gurobi_solutions)
    
    non_zero_mask = gurobi_solutions != 0
    relative_errors = np.abs(gurobi_solutions[non_zero_mask] - dnn_predictions[non_zero_mask]) / gurobi_solutions[non_zero_mask]
    mre = np.mean(relative_errors) * 100 # As a percentage
    mae = np.mean(np.abs(gurobi_solutions - dnn_predictions))
    rmse = np.sqrt(np.mean((gurobi_solutions - dnn_predictions)**2))

    avg_dnn_time_ms = np.mean(dnn_times) * 1000
    avg_gurobi_time_ms = np.mean(gurobi_times) * 1000
    
    evaluation_results.append({
        'n': n,
        'mae': mae,
        'mre': mre,
        'rmse': rmse,
        'avg_dnn_time_ms': avg_dnn_time_ms,
        'avg_gurobi_time_ms': avg_gurobi_time_ms
    })
    print(f"n={n}: MAE = {mae:.2f}, MRE = {mre:.2f}%, RMSE = {rmse:.2f}%, DNN Time = {avg_dnn_time_ms:.4f}ms, \
          Gurobi Time = {avg_gurobi_time_ms:.4f}ms")


# 4. Print the final summary table
print("\n\n--- Final Performance Comparison Across All Test Sizes ---")
print("-" * 100)
print(f"{'N_Items':<10} | {'MAE':<15} | {'MRE (%)':<15} | {'RMSE':<15} | {'Avg DNN Time (ms)':<20} | {'Avg Gurobi Time (ms)':<20}")
print("-" * 100)
for result in evaluation_results:
    print(f"{result['n']:<10} | {result['mae']:<15.2f} | {result['mre']:<15.2f} | {result['rmse']:<15.2f} \
          | {result['avg_dnn_time_ms']:<20.4f} | {result['avg_gurobi_time_ms']:<20.4f}")
print("-" * 100)

# 5. Plot the performance curves
# Extract data for plotting
n_values = [r['n'] for r in evaluation_results]
mae_values = [r['mae'] for r in evaluation_results]
mre_values = [r['mre'] for r in evaluation_results]
rmse_values = [r['rmse'] for r in evaluation_results]
time_values_dnn = [r['avg_dnn_time_ms'] for r in evaluation_results]
time_values_gurobi = [r['avg_gurobi_time_ms'] for r in evaluation_results]

# Create plot for accuracy vs. problem size
plt.figure(figsize=(12, 6))
plt.plot(n_values, mae_values, marker='o', linestyle='-', label='Mean Absolute Error (MAE)')
plt.plot(n_values, rmse_values, marker='s', linestyle='--', label='Root Mean Squared Error (RMSE)')
plt.title('Model Error vs. Problem Size (n)')
plt.xlabel('Number of Items (n)')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plot_acc_path = os.path.join(cfg.MODEL_PLOTS_DIR, "evaluation_accuracy_vs_n.png")
plt.savefig(plot_acc_path)
plt.close()
print(f"\nAccuracy plot saved to {plot_acc_path}")

# Create plot for time vs. problem size
plt.figure(figsize=(12, 6))
plt.plot(n_values, time_values_dnn, marker='o', linestyle='-', label='DNN Prediction Time (ms)')
plt.plot(n_values, time_values_gurobi, marker='s', linestyle='--', label='Gurobi Solve Time (ms)')
plt.title('Prediction Time vs. Problem Size (n)')
plt.xlabel('Number of Items (n)')
plt.ylabel('Average Time per Instance (ms)')
plt.legend()
plt.grid(True)
plot_time_path = os.path.join(cfg.MODEL_PLOTS_DIR, "evaluation_time_vs_n.png")
plt.savefig(plot_time_path)
plt.close()
print(f"Time comparison plot saved to {plot_time_path}")

