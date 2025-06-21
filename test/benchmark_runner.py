# benchmark_runner.py
# -*- coding: utf-8 -*-

import os
import time
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import generator as gen
import configs.config as cfg
import argparse
from src.utils.logger_config import setup_logger
import logging

# --- Logger Setup ---
setup_logger('benchmark')
logger = logging.getLogger(__name__)

# --- Benchmark Configuration ---
TEST_SUITE_DIR = cfg.TEST_SUITE_DIR
PLOT_DIR = cfg.PLOT_DIR
RESULTS_DIR = cfg.RESULTS_DIR
ALGORITHMS_TO_TEST = cfg.ALGORITHMS_TO_TEST
BASELINE_ALGORITHM_FUNCTION = cfg.BASELINE_ALGORITHM


def run_benchmarks(limit=None):
    """
    Loads instances, runs all algorithms, and collects performance data.
    Args:
        limit (int, optional): The maximum number of instances to run. Defaults to None (run all).
    Returns:
        - A DataFrame for execution times.
        - A DataFrame for solution quality (MSE).
        - The string key of the baseline algorithm.
    """
    test_files = glob.glob(os.path.join(TEST_SUITE_DIR, "*.csv"))
    if not test_files:
        logger.error(f"No test cases found in '{TEST_SUITE_DIR}'.")
        logger.warning("Please run 'generate_test_suite.py' first.")
        return None, None, None

    # Sort files by the number of items 'n' to ensure we test the smallest instances first
    sorted_files = sorted(test_files, key=lambda p: int(p.split('_n')[1].split('_')[0]))

    # If a limit is provided, slice the list of files
    if limit is not None and limit > 0:
        logger.info(f"--- Running in limited mode. Processing only the first {limit} instances. ---")
        sorted_files = sorted_files[:limit]

    # --- Find the string key for the baseline algorithm ---
    baseline_key = None
    for name, func in ALGORITHMS_TO_TEST.items():
        if func == BASELINE_ALGORITHM_FUNCTION:
            baseline_key = name
            break
    
    if baseline_key is None:
        logger.error("The baseline algorithm function could not be found in the ALGORITHMS_TO_TEST dictionary.")
        return None, None, None
    
    logger.info(f"Using '{baseline_key}' as the baseline for quality comparison.")
    logger.info(f"--- Running benchmarks on {len(sorted_files)} instances ---")

    time_results = []
    quality_results = []

    for filepath in sorted_files:
        n_items = int(os.path.basename(filepath).split('_n')[1].split('_')[0])
        logger.info(f"Processing instance: {os.path.basename(filepath)} (n={n_items})")
        weights, values, capacity = gen.load_instance_from_file(filepath)

        instance_solution_values = {}
        for name, func in ALGORITHMS_TO_TEST.items():
            start_time = time.perf_counter()
            solution_value = func(weights=weights, values=values, capacity=capacity)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            logger.debug(f"  -> {name}: Time {execution_time:.6f}s, Value {solution_value}")
            
            time_results.append({
                "n_items": n_items,
                "algorithm": name,
                "time": execution_time
            })
            instance_solution_values[name] = solution_value

        optimal_value = instance_solution_values.get(baseline_key)
        if optimal_value is None:
            logger.warning(f"Could not get baseline solution for instance n={n_items}. Skipping quality calculation.")
            continue

        for name, value in instance_solution_values.items():
            squared_error = (optimal_value - value) ** 2
            quality_results.append({
                "n_items": n_items,
                "algorithm": name,
                "mse": squared_error
            })

    time_df = pd.DataFrame(time_results)
    quality_df = pd.DataFrame(quality_results)
    
    quality_df = quality_df[quality_df['algorithm'] != baseline_key].copy()

    return time_df, quality_df, baseline_key


def plot_time_results(df):
    """Plots algorithm execution time vs. problem size."""
    if df is None or df.empty:
        logger.warning("No time data available to plot.")
        return

    logger.info("--- Plotting time performance ---")
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    plot = sns.lineplot(
        data=df,
        x="n_items",
        y="time",
        hue="algorithm",
        marker='o',
        linestyle='-'
    )
    
    plot.set_title('Algorithm Execution Time vs. Problem Size (n)', fontsize=16)
    plot.set_xlabel('Number of Items (n)', fontsize=12)
    plot.set_ylabel('Execution Time (seconds)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Algorithm')
    plt.tight_layout()
    
    output_path = os.path.join(PLOT_DIR, "benchmark_time_comparison.png")
    plt.savefig(output_path)
    logger.info(f"Time performance plot saved to {output_path}")
    plt.close()


def plot_quality_results(df, baseline_key):
    """Plots algorithm solution MSE vs. problem size."""
    if df is None or df.empty:
        logger.warning("No quality data available to plot.")
        return

    logger.info("--- Plotting solution quality (MSE) ---")
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    plot = sns.lineplot(
        data=df,
        x="n_items",
        y="mse",
        hue="algorithm",
        marker='o',
        linestyle='-'
    )
    
    plot.set_title('Solution Quality (MSE) vs. Problem Size (n)', fontsize=16)
    plot.set_xlabel('Number of Items (n)', fontsize=12)
    plot.set_ylabel(f'Squared Error (vs. {baseline_key})', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Algorithm')
    plt.tight_layout()
    
    output_path = os.path.join(PLOT_DIR, "benchmark_quality_comparison.png")
    plt.savefig(output_path)
    logger.info(f"Solution quality plot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    # --- Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run benchmark tests for knapsack algorithms.")
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of test instances to run (for quick testing).')
    args = parser.parse_args()

    # --- Directory and Benchmark Execution ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    time_df, quality_df, baseline_key = run_benchmarks(limit=args.limit)

    if time_df is not None and not time_df.empty:
        time_filepath = os.path.join(RESULTS_DIR, "benchmark_time_results.csv")
        time_df.to_csv(time_filepath, index=False)
        logger.info(f"--- Time benchmark results saved to '{time_filepath}' ---")
        plot_time_results(time_df)
    else:
        logger.warning("No time benchmark data was generated.")

    if quality_df is not None and not quality_df.empty:
        quality_filepath = os.path.join(RESULTS_DIR, "benchmark_quality_results.csv")
        quality_df.to_csv(quality_filepath, index=False)
        logger.info(f"--- Solution quality benchmark results saved to '{quality_filepath}' ---")
        plot_quality_results(quality_df, baseline_key)
    else:
        logger.warning("No solution quality benchmark data was generated.")

    logger.info(f"\n--- Benchmark complete. Plots are in '{PLOT_DIR}', results are in '{RESULTS_DIR}'. ---")