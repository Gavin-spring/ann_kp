# model_config.py
# Centralized configuration for model training and testing.

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import algorithms as alg

# --- Directory Settings ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_TRAIN_CASES = os.path.join(BASE_DIR, "model_train_cases")
MODEL_TEST_CASES = os.path.join(BASE_DIR, "model_test_cases")

# --- Test cases Generation Settings ---
CORRELATION_TYPE = 'uncorrelated'

# Item properties
MAX_WEIGHT = 100
MAX_VALUE = 100
CAPACITY_RATIO = 0.5

BASELINE_ALGORITHM = alg.knapsack_gurobi
