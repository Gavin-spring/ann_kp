# config.py
# Centralized configuration for the knapsack benchmark project.

import os
import algorithms as alg

# --- Directory Settings ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_SUITE_DIR = os.path.join(BASE_DIR, "test_cases")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# --- Test Suite Generation Settings ---
# See generator.py for correlation options:
# 'uncorrelated', 'weakly_correlated', 'strongly_correlated', 'subset_sum'
CORRELATION_TYPE = 'uncorrelated'

# Defines problem sizes to generate: (start_n, stop_n, step)
N_RANGE = (10, 501, 10)

# Item properties
MAX_WEIGHT = 100
MAX_VALUE = 100
CAPACITY_RATIO = 0.5 # Knapsack capacity as a ratio of total item weight

# --- Benchmark Runner Settings ---
# Algorithms to test. The key is the name shown in plots,
# and the value is the function from algorithms.py.
ALGORITHMS_TO_TEST = {
    "2D DP": alg.knapsack_01_2d,
    "1D DP (Optimized)": alg.knapsack_01_1d,
    "Branch and Bound": alg.knapsack_branch_and_bound,
    "Gurobi": alg.knapsack_gurobi
}

BASELINE_ALGORITHM = ALGORITHMS_TO_TEST["Gurobi"]  # Default baseline algorithm for comparison

# TODO: check outputs of every algorithm with gurobi
