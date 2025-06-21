# File: ann/dnn_config.py
# -*- coding: utf-8 -*-

'''
# Centralized configuration for model case-generating, training and testing.
'''
import os
import sys

# --- Path Definitions ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

ANN_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_TRAIN_CASES = os.path.join(ANN_DIR, "model_train_cases")
MODEL_TEST_CASES = os.path.join(ANN_DIR, "model_test_cases")
SAVED_MODELS_DIR = os.path.join(ANN_DIR, "saved_models")
MODEL_PLOTS_DIR = os.path.join(ANN_DIR, "model_plots")
MODEL_LOGS_DIR = os.path.join(ANN_DIR, "model_logs")
LOG_DIR = MODEL_LOGS_DIR

# --- Test cases Generation Settings ---
CORRELATION_TYPE = 'uncorrelated'

# --- Hardware Configuration ---
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset and Model Hyperparameters ---
START_N = 5
END_N = 100
STEP_N = 5
MAX_N = END_N  # The largest 'n' determines the model's input size
INPUT_SIZE = MAX_N * 4 + 1

MAX_WEIGHT = 100 # Assuming this was the max weight used for normalization
MAX_VALUE = 100 # Assuming this was the max value used for normalization
CAPACITY_RATIO = 0.5  # Ratio of capacity to the sum of weights
TARGET_SCALE_FACTOR = float(MAX_N * MAX_VALUE) # A plausible upper bound for the optimal value in any instance.

# --- Training Hyperparameters ---
EPOCHS_PER_N = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

# Baseline Algorithm
import algorithms as alg
BASELINE_ALGORITHM = alg.knapsack_gurobi

# --- Utility Function for Naming ---
def get_model_and_plot_names():
    total_epochs = (END_N - START_N) // STEP_N * EPOCHS_PER_N + EPOCHS_PER_N
    base_filename = f"model_curriculum_n{START_N}-{END_N}_totalepochs{total_epochs}"
    model_filename = f"{base_filename}.pth"
    plot_filename = f"{base_filename}_losscurve.png"
    
    model_path = os.path.join(SAVED_MODELS_DIR, model_filename)
    plot_path = os.path.join(MODEL_PLOTS_DIR, plot_filename)
    
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(MODEL_PLOTS_DIR, exist_ok=True)
    
    return model_path, plot_path