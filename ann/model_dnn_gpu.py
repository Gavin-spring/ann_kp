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


# --- Step 1 Load dataset from files for a specific n ---
def load_knapsack_dataset_from_files(n_items: int, data_dir: str):
    dataset = []
    filename_pattern = f"instance_n{n_items}_*.csv"
    test_suite_path = os.path.join(data_dir, filename_pattern)
    print(f"Searching for files with pattern: '{filename_pattern}' in '{data_dir}'...")
    instance_files = glob.glob(test_suite_path)
    
    if not instance_files:
        print(f"Error: No .csv files found for n={n_items} in '{data_dir}'.")
        return None
    print(f"Found {len(instance_files)} instance files for n={n_items}. Processing...")
    for filepath in instance_files:
        weights, values, capacity = gen.load_instance_from_file(filepath)
        optimal_value = alg.knapsack_gurobi(weights=weights, values=values, capacity=capacity)
        feature_vector = np.array(weights + values + [capacity / cfg.MAX_WEIGHT], dtype=np.float32)
        dataset.append((feature_vector, np.float32(optimal_value)))
    print("Dataset loading and processing complete.")
    return dataset

N_ITEMS = 10
knapsack_dataset = load_knapsack_dataset_from_files(n_items=N_ITEMS, data_dir=cfg.MODEL_TRAIN_CASES)

if not knapsack_dataset:
    sys.exit("Dataset could not be loaded. Exiting.") # Exit if no data
print(f"\nSuccessfully loaded {len(knapsack_dataset)} samples for n={N_ITEMS}.")

# --- Step 2: Create PyTorch Dataset and DataLoader ---
class KnapsackDataset(Dataset):
    def __init__(self, data):
        self.features = [torch.tensor(item[0]) for item in data]
        self.labels = [torch.tensor(item[1]).unsqueeze(0) for item in data]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_size = int(0.8 * len(knapsack_dataset))
val_size = len(knapsack_dataset) - train_size
train_data, val_data = torch.utils.data.random_split(knapsack_dataset, [train_size, val_size])
train_loader = DataLoader(KnapsackDataset(train_data), batch_size=32, shuffle=True)
val_loader = DataLoader(KnapsackDataset(val_data), batch_size=32)
print(f"\nCreated DataLoader with {len(train_loader)} training batches.")



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

INPUT_SIZE = N_ITEMS * 2 + 1
model = KnapsackPredictor(INPUT_SIZE)
model.to(device) # Move the model to the selected device (GPU/CPU)
print("\nDNN Model defined:")
print(model)



# --- Step 4: Training Loops ---
# Training configurations
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # regularization
EPOCHS = 300 # NOTE: training epochs
num_instances = len(knapsack_dataset)
best_val_loss = math.inf

# Create a base filename from parameters
base_filename = f"model_n{N_ITEMS}_epochs{EPOCHS}_instances{num_instances}"

# Create full path for the model file using config
os.makedirs(cfg.SAVED_MODELS_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(cfg.SAVED_MODELS_DIR, base_filename + ".pth")

# Create full path for the plot file using config
os.makedirs(cfg.MODEL_PLOTS_DIR, exist_ok=True)
plot_save_path = os.path.join(cfg.MODEL_PLOTS_DIR, base_filename + "_losscurve.png")

# Start training
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        # Move input and label tensors to the selected device
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data
            # Move input and label tensors to the selected device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    current_val_loss = val_loss / len(val_loader)
    # Log the losses
    train_losses.append(running_loss / len(train_loader))
    val_losses.append(current_val_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {running_loss / len(train_loader):.4f}, Val Loss: {current_val_loss:.4f}")
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



# --- Step 5: Evaluate the trained model ---
print("\n--- Evaluating BEST Saved Model on the Test Set ---")

# 1. Create a new instance of the model and move it to the device
final_model = KnapsackPredictor(INPUT_SIZE)
final_model.to(device)

# 2. Load the state dictionary of the best trained model
#    Construct the model path dynamically to ensure we load the correct model
EPOCHS = 300 # NOTE: this should match the training epochs
num_instances = len(knapsack_dataset)
base_filename = f"model_n{N_ITEMS}_epochs{EPOCHS}_instances{num_instances}"
MODEL_SAVE_PATH = os.path.join(cfg.SAVED_MODELS_DIR, base_filename + ".pth")

if not os.path.exists(MODEL_SAVE_PATH):
    sys.exit(f"Model file not found at {MODEL_SAVE_PATH}. Please run training first.")
    
final_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
final_model.eval()

# 3. Load the dedicated test dataset
test_dataset = load_knapsack_dataset_from_files(n_items=N_ITEMS, data_dir=cfg.MODEL_TEST_CASES)
if not test_dataset:
    sys.exit("Test dataset could not be loaded. Exiting.")

# 4. Loop through the test set and collect performance metrics
dnn_predictions = []
gurobi_solutions = []
dnn_times = []
gurobi_times = []

with torch.no_grad():
    for features, gurobi_solution in test_dataset:
        gurobi_solutions.append(gurobi_solution.item())
        
        # --- DNN Performance ---
        features_tensor = torch.tensor(features).to(device)

        start_time = time.perf_counter()
        predicted_value = final_model(features_tensor)
        end_time = time.perf_counter()
        dnn_predictions.append(predicted_value.item())
        dnn_times.append(end_time - start_time)
        
        # --- Gurobi Performance (for time comparison) ---
        # Decode features back to weights, values, capacity to run Gurobi
        weights = features[:N_ITEMS].tolist()
        values = features[N_ITEMS:N_ITEMS*2].tolist()
        capacity = int(features[-1] * cfg.MAX_WEIGHT) # De-normalize
        
        start_time = time.perf_counter()
        alg.knapsack_gurobi(weights=weights, values=values, capacity=capacity)
        end_time = time.perf_counter()
        gurobi_times.append(end_time - start_time)

# 5. Calculate and print the final performance summary
dnn_predictions = np.array(dnn_predictions)
gurobi_solutions = np.array(gurobi_solutions)

mse = np.mean((gurobi_solutions - dnn_predictions)**2)
mae = np.mean(np.abs(gurobi_solutions - dnn_predictions))
avg_dnn_time = np.mean(dnn_times)
avg_gurobi_time = np.mean(gurobi_times)

print("\n--- Final Performance Comparison on Test Set ---")
print(f"Number of test instances: {len(test_dataset)}")
print(f"\nAccuracy Metrics:")
print(f"  - Mean Squared Error (MSE): {mse:.4f}")
print(f"  - Root Mean Squared Error (RMSE): {np.sqrt(mse):.4f}")
print(f"  - Mean Absolute Error (MAE): {mae:.4f}")
print(f"\nSpeed Metrics (per instance):")
print(f"  - Average DNN Prediction Time: {avg_dnn_time*1000:.6f} ms")
print(f"  - Average Gurobi Solve Time: {avg_gurobi_time*1000:.6f} ms")
print("-" * 50)
# print("\n--- Evaluating BEST Saved Model on a New Instance ---")
# final_model = KnapsackPredictor(INPUT_SIZE)
# # Move the final model structure to the device before loading state_dict
# final_model.to(device)
# final_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
# final_model.eval()

# # Generate a new test instance
# test_instance, capacity = gen.generate_knapsack_instance(n=N_ITEMS, correlation='uncorrelated', max_weight=100, max_value=100, capacity_ratio=0.5)
# test_weights = [item[1] for item in test_instance]
# test_values = [item[0] for item in test_instance]
# true_value = alg.knapsack_gurobi(weights=test_weights, values=test_values, capacity=capacity)
# test_features_np = np.array(test_weights + test_values + [capacity / 100], dtype=np.float32)

# # Move the final test tensor to the device
# test_features_tensor = torch.from_numpy(test_features_np).to(device)

# with torch.no_grad():
#     predicted_value = final_model(test_features_tensor)

# print(f"Problem: weights={test_weights}, values={test_values}, capacity={capacity}")
# print(f"Gurobi's True Optimal Value: {true_value}")
# print(f"DNN's Predicted Value (from loaded best model): {predicted_value.item():.2f}")