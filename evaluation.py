# File: evaluate.py
# -*- coding: utf-8 -*-

import torch
import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt
import logging

# Local imports
import configs.dnn_config as cfg
from model import KnapsackPredictor
from data_loader import load_knapsack_dataset_from_files
import algorithms as alg
from src.utils.logger_config import setup_logger

def main():
    """Main function to evaluate the trained model."""
    # --- Logger Setup ---
    setup_logger('model_evaluation')
    logger = logging.getLogger(__name__)

    logger.info(f"Using device: {cfg.DEVICE}")
    logger.info("--- Evaluating BEST Saved Model on the Test Set ---")

    final_model = KnapsackPredictor(cfg.INPUT_SIZE).to(cfg.DEVICE)
    MODEL_SAVE_PATH, _ = cfg.get_model_and_plot_names()

    if not os.path.exists(MODEL_SAVE_PATH):
        logger.error(f"Model file not found at {MODEL_SAVE_PATH}. Please run train.py first.")
        sys.exit(f"Model file not found at {MODEL_SAVE_PATH}. Please run train.py first.")
        
    logger.info(f"Loading best model from: {MODEL_SAVE_PATH}")
    final_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    final_model.eval()

    evaluation_results = []
    for n in range(cfg.START_N, cfg.END_N + cfg.STEP_N, cfg.STEP_N):
        logger.info(f"--- Evaluating for n={n} ---")
        test_dataset = load_knapsack_dataset_from_files(
            n_items=n, data_dir=cfg.MODEL_TEST_CASES, max_n=cfg.MAX_N
        )
        if not test_dataset:
            logger.warning(f"No test data found for n={n}. Skipping.")
            continue

        dnn_predictions, gurobi_solutions = [], []
        dnn_times, gurobi_times = [], []

        with torch.no_grad():
            for features, normalized_solution in test_dataset:
                
                # 1. PREPARE INPUT TENSOR
                features_tensor = torch.tensor(features).unsqueeze(0).to(cfg.DEVICE)

                # 2. RUN DNN MODEL (ONCE) AND TIME IT
                start_time = time.perf_counter()
                predicted_normalized_value = final_model(features_tensor)
                dnn_times.append(time.perf_counter() - start_time)
                
                # 3. DE-NORMALIZE THE PREDICTION and append the REAL value
                predicted_value = predicted_normalized_value.item() * cfg.TARGET_SCALE_FACTOR
                dnn_predictions.append(predicted_value)

                # 4. DE-NORMALIZE THE GROUND TRUTH and append the REAL value
                gurobi_solution = normalized_solution.item() * cfg.TARGET_SCALE_FACTOR
                gurobi_solutions.append(gurobi_solution)

                # 5. RUN GUROBI SEPARATELY (for time comparison only)
                weights = features[:cfg.MAX_N].tolist()
                values = features[cfg.MAX_N:cfg.MAX_N*2].tolist()
                # De-normalize capacity from the feature vector to get the real capacity
                capacity = int(features[cfg.MAX_N * 4] * cfg.MAX_WEIGHT)
                
                start_time = time.perf_counter()
                alg.knapsack_gurobi(weights=weights, values=values, capacity=capacity)
                gurobi_times.append(time.perf_counter() - start_time)

        # Calculate metrics
        dnn_predictions = np.array(dnn_predictions)
        gurobi_solutions = np.array(gurobi_solutions)
        
        non_zero_mask = gurobi_solutions != 0
        relative_errors = np.abs(gurobi_solutions[non_zero_mask] - dnn_predictions[non_zero_mask]) / gurobi_solutions[non_zero_mask]
        mre = np.mean(relative_errors) * 100
        mae = np.mean(np.abs(gurobi_solutions - dnn_predictions))
        rmse = np.sqrt(np.mean((gurobi_solutions - dnn_predictions)**2))
        avg_dnn_time_ms = np.mean(dnn_times) * 1000
        avg_gurobi_time_ms = np.mean(gurobi_times) * 1000
        
        evaluation_results.append({'n': n, 'mae': mae, 'mre': mre, 'rmse': rmse,
                                   'avg_dnn_time_ms': avg_dnn_time_ms, 'avg_gurobi_time_ms': avg_gurobi_time_ms})
        logger.info(f"n={n}: MAE={mae:.2f}, MRE={mre:.2f}%, RMSE={rmse:.2f}, DNN Time={avg_dnn_time_ms:.4f}ms, Gurobi Time={avg_gurobi_time_ms:.4f}ms")

    # Print summary table and plots
    print_summary_and_plots(evaluation_results)

def print_summary_and_plots(results):
    """Prints the final results table and generates performance plots."""
    logger = logging.getLogger(__name__)

    logger.info("--- Final Performance Comparison ---")
    header = f"{'N_Items':<10} | {'MAE':<15} | {'MRE (%)':<15} | {'RMSE':<15} | {'Avg DNN Time (ms)':<20} | {'Avg Gurobi Time (ms)':<20}"
    separator = "-" * len(header)
    logger.info(separator)
    logger.info(header)
    logger.info(separator)
    for res in results:
        log_line = f"{res['n']:<10} | {res['mae']:<15.2f} | {res['mre']:<15.2f} | {res['rmse']:<15.2f} | {res['avg_dnn_time_ms']:<20.4f} | {res['avg_gurobi_time_ms']:<20.4f}"
        logger.info(log_line)
    logger.info(separator)
    
    # Plotting logic remains the same
    n_values = [r['n'] for r in results]
    mae_values = [r['mae'] for r in results]
    mre_values = [r['mre'] for r in results]
    rmse_values = [r['rmse'] for r in results]
    time_values_dnn = [r['avg_dnn_time_ms'] for r in results]
    time_values_gurobi = [r['avg_gurobi_time_ms'] for r in results]

    # Create and save accuracy plot
    plt.figure(figsize=(12, 6))
    plt.plot(n_values, mae_values, marker='o', linestyle='-', label='Mean Absolute Error (MAE)')
    plt.plot(n_values, mre_values, marker='^', linestyle='--', label='Mean Relative Error (MRE)')
    plt.plot(n_values, rmse_values, marker='s', linestyle='-', label='Root Mean Squared Error (RMSE)')
    plt.title('Model Error vs. Problem Size (n)')
    plt.xlabel('Number of Items (n)')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plot_acc_path = os.path.join(cfg.MODEL_PLOTS_DIR, "evaluation_accuracy_vs_n.png")
    plt.savefig(plot_acc_path)
    plt.close()
    logger.info(f"Accuracy plot saved to {plot_acc_path}")

    # Create and save time comparison plot
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
    logger.info(f"Time comparison plot saved to {plot_time_path}")

if __name__ == '__main__':
    main()