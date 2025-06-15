# benchmark_runner.py

import os
import time
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import algorithms as alg
import generator as gen

# --- Benchmark Configuration ---
TEST_SUITE_DIR = "test_cases"
PLOT_DIR = "plots"

# Algorithms to test
ALGORITHMS_TO_TEST = {
    "2D DP": alg.knapsack_01_2d,
    "1D DP (Optimized)": alg.knapsack_01_1d
}



def run_benchmarks():
    """Loads instances, runs algorithms, and collects performance data."""
    
    # Find all test case files in the directory, looking for .csv files.
    test_files = glob.glob(os.path.join(TEST_SUITE_DIR, "*.csv"))
    if not test_files:
        print(f"Error: No test cases found in '{TEST_SUITE_DIR}'.")
        print("Please run 'generate_test_suite.py' first.")
        return None

    print(f"--- Running Benchmarks on {len(test_files)} Instances ---")
    
    results = []

    # Loop through each test file
    for filepath in sorted(test_files, key=lambda p: int(p.split('_n')[1].split('_')[0])):
        # Extract number of items from filename for plotting
        n_items = int(os.path.basename(filepath).split('_n')[1].split('_')[0])
        
        print(f"\nProcessing instance: {os.path.basename(filepath)} (n={n_items})")
        weights, values, capacity = gen.load_instance_from_file(filepath)

        # Test each algorithm
        for name, func in ALGORITHMS_TO_TEST.items():
            start_time = time.perf_counter()
            # Execute the algorithm using keyword arguments
            _ = func(weights=weights, values=values, capacity=capacity)
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            print(f"  -> {name}: {execution_time:.6f} seconds")
            
            # Store the result
            results.append({
                "n_items": n_items,
                "algorithm": name,
                "time": execution_time
            })

    return pd.DataFrame(results)



def plot_results(df):

    if df is None or df.empty:
        print("No data to plot.")
        return

    print("\n--- Plotting Results ---")
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    sns.lineplot(
        data=df,
        x="n_items",
        y="time",
        hue="algorithm",
        marker='o',
        linestyle='-'
    )
    
    plt.title('Algorithm Performance vs. Problem Size (n)', fontsize=16)
    plt.xlabel('Number of Items (n)', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Algorithm')
    plt.tight_layout()
    
    output_path = os.path.join(PLOT_DIR, "benchmark_comparison.png")
    
    plt.savefig(output_path)
    print(f"Plot successfully saved to {output_path}")
    
    plt.show()


if __name__ == "__main__":
    results_df = run_benchmarks()
    if results_df is not None:
        plot_results(results_df)
        print("\n--- Benchmark Complete. Plot saved to 'benchmark_comparison.png' ---")