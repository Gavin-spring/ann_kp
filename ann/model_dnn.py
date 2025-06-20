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

import generator as gen
import algorithms as alg
import model_config as cfg


# --- Step 1 Load dataset from files for a specific n ---
def load_knapsack_dataset_from_files(n_items: int):
    """
    Loads a dataset for a specific size 'n' from .csv files in a directory.
    
    Args:
        n_items (int): The specific number of items (n) of the knapsack instances to load.
    """
    dataset = []
    
    # Create a specific filename pattern to match only the desired n_items
    filename_pattern = f"instance_n{n_items}_*.csv"
    test_suite_path = os.path.join(cfg.TEST_SUITE_DIR, filename_pattern)
    
    print(f"Searching for files with pattern: '{filename_pattern}'...")
    
    # Find all .csv files in the directory that match the pattern
    instance_files = glob.glob(test_suite_path)

    if not instance_files:
        print(f"Error: No .csv files found for n={n_items} in '{cfg.TEST_SUITE_DIR}'.")
        print(f"Please run 'create_fixed_size_suite(n={n_items}, ...)' or 'create_suite()' first.")
        return None

    print(f"Found {len(instance_files)} instance files for n={n_items}. Processing...")

    for filepath in instance_files:
        weights, values, capacity = gen.load_instance_from_file(filepath)
        # Get the optimal value using the Gurobi solver
        optimal_value = alg.knapsack_gurobi(weights=weights, values=values, capacity=capacity)
        feature_vector = np.array(weights + values + [capacity / cfg.MAX_WEIGHT], dtype=np.float32)
        dataset.append((feature_vector, np.float32(optimal_value)))

    print("Dataset loading and processing complete.")
    return dataset

N_ITEMS = 10 # size of the knapsack problem instances
knapsack_dataset = load_knapsack_dataset_from_files(n_items=N_ITEMS)

# Check if dataset was loaded successfully before proceeding
if knapsack_dataset:
    print(f"\nSuccessfully loaded {len(knapsack_dataset)} samples for n={N_ITEMS}.")



# --- Step 2: Create PyTorch Dataset and DataLoader ---
class KnapsackDataset(Dataset):
    # QUESTION
    """Custom PyTorch Dataset for our knapsack problems."""
    def __init__(self, data):
        self.features = [torch.tensor(item[0]) for item in data]
        self.labels = [torch.tensor(item[1]).unsqueeze(0) for item in data]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Split data into training and validation sets
train_size = int(0.8 * len(knapsack_dataset))
val_size = len(knapsack_dataset) - train_size
train_data, val_data = torch.utils.data.random_split(knapsack_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(KnapsackDataset(train_data), batch_size=32, shuffle=True)
val_loader = DataLoader(KnapsackDataset(val_data), batch_size=32)

print(f"\nCreated DataLoader with {len(train_loader)} training batches.")



# --- Step 3: Define the DNN Model ---
class KnapsackPredictor(nn.Module):
    # QUESTION：what is .module?
    def __init__(self, input_size):
        super(KnapsackPredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, 128) # QUESTION: why 128?
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1) # Output a single value
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x) # No activation on the final output for regression
        return x

# The input size is n_items * 2 (for weights and values) + 1 (for capacity)
INPUT_SIZE = N_ITEMS * 2 + 1
model = KnapsackPredictor(INPUT_SIZE)
print("\nDNN Model defined:")
print(model)



# --- Step 4: Training Loops ---
# Training configuration
criterion = nn.MSELoss() # Mean Squared Error is good for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) # NOTE: learning rate of 0.001

best_val_loss = math.inf
EPOCHS = 200 # NOTE: Number of epochs to train
num_instances = len(knapsack_dataset) #

model_filename = f"model_n{N_ITEMS}_epochs{EPOCHS}_instances{num_instances}.pth" # TODO: change to .ckpt afterwards
MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, model_filename)

print(f"Model will be saved to: {MODEL_SAVE_PATH}")

# Operation in training loop
print("\n--- Starting Training ---")
train_losses = []
val_losses = []
for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    
    model.train() # Set the model to training mode
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        # Get inputs and labels from the dataloader
        inputs, labels = data
        # Zero the parameter gradients
        optimizer.zero_grad() # QUESTION: why do we need to zero the gradients?
        # Forward pass
        outputs = model(inputs)
        # Calculate loss
        loss = criterion(outputs, labels)
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    train_end_time = time.time()
    
    # --- Validation Stage ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad(): #QUESTION: why do we need to use torch.no_grad()?
        for i, data in enumerate(val_loader):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    current_val_loss = val_loss / len(val_loader)
    
    val_end_time = time.time()
    
    # Store and print statistics
    train_losses.append(running_loss / len(train_loader))
    val_losses.append(current_val_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, "
          f"Train Loss: {running_loss / len(train_loader):.4f}, "
          f"Val Loss: {current_val_loss:.4f}")

    # Check and save the best model
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  -> Found new best model! Saved to {MODEL_SAVE_PATH}")

    # Estimate total time based on the first epoch
    if epoch == 0:
        train_duration = train_end_time - epoch_start_time
        val_duration = val_end_time - train_end_time
        total_epoch_duration = val_end_time - epoch_start_time
        estimated_total_time = total_epoch_duration * EPOCHS
        print(f"  -> Time for 1st Epoch: {total_epoch_duration:.2f}s "
              f"(Train: {train_duration:.2f}s, Val: {val_duration:.2f}s)")
        print(f"  -> Estimated Total Training Time for {EPOCHS} epochs: {estimated_total_time:.2f}s "
              f"({estimated_total_time/60:.2f} minutes)")
        
# plot the training and validation loss
print("--- Plotting training history ---")
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")
plt.close()

print("Loss curve plot saved as loss_curve.png")
print("--- Finished Training ---")



# --- Step 5: Evaluate the trained model ---
print("\n--- Evaluating BEST Saved Model on a New Instance ---")

# 1. Create a new instance of the model architecture
final_model = KnapsackPredictor(INPUT_SIZE)

# 2. Load the saved state dictionary
final_model.load_state_dict(torch.load(MODEL_SAVE_PATH))

# 3. Set the model to evaluation mode
final_model.eval()

# Generate a single new test instance
test_instance, capacity = gen.generate_knapsack_instance(n=N_ITEMS, correlation='uncorrelated', max_weight=100, max_value=100, capacity_ratio=0.5)
test_weights = [item[1] for item in test_instance]
test_values = [item[0] for item in test_instance]
true_value = alg.knapsack_gurobi(weights=test_weights, values=test_values, capacity=capacity)
test_features_np = np.array(test_weights + test_values + [capacity / 100], dtype=np.float32)
test_features_tensor = torch.from_numpy(test_features_np)

# Get the model's prediction using the loaded model
with torch.no_grad():
    predicted_value = final_model(test_features_tensor)

print(f"Problem: weights={test_weights}, values={test_values}, capacity={capacity}")
print(f"Gurobi's True Optimal Value: {true_value}")
print(f"DNN's Predicted Value (from loaded best model): {predicted_value.item():.2f}")

