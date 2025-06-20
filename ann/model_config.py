# model_config.py
# Centralized configuration for model training and testing.

import os
import algorithms as alg

# --- Directory Settings ---
# Base directory for the 'ann' module, which is the directory of this config file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directories for data, saved models, and plots, relative to the 'ann' folder.
MODEL_TRAIN_CASES = os.path.join(BASE_DIR, "model_train_cases")
MODEL_TEST_CASES = os.path.join(BASE_DIR, "model_test_cases")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
MODEL_PLOTS_DIR = os.path.join(BASE_DIR, "model_plots")
MODEL_LOGS_DIR = os.path.join(BASE_DIR, "model_logs")

# --- Test cases Generation Settings ---
CORRELATION_TYPE = 'uncorrelated'

# Item properties
MAX_WEIGHT = 100
MAX_VALUE = 100
CAPACITY_RATIO = 0.5

BASELINE_ALGORITHM = alg.knapsack_gurobi