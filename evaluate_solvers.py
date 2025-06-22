# evaluate_solvers.py
import logging
import os
import sys
import pandas as pd
from tqdm import tqdm
import numpy as np

from src.utils.config_loader import cfg, ALGORITHM_REGISTRY
from src.utils.logger import setup_logger
from src.evaluation.plotting import plot_evaluation_errors, plot_evaluation_times
from src.evaluation.reporting import save_results_to_csv

def main():
    """
    This script evaluates all solvers, strictly ensures error metrics can be
    calculated, and then generates reports and plots.
    """
    setup_logger(run_name="evaluation_session", log_dir=cfg.paths.logs)
    logger = logging.getLogger(__name__)

    # --- 1. Setup Solvers ---
    # We will test all solvers currently active in the registry.
    solvers_to_evaluate = ALGORITHM_REGISTRY
    if not solvers_to_evaluate:
        logger.critical("No solvers are active in ALGORITHM_REGISTRY. Exiting.")
        sys.exit(1)
    logger.info(f"Solvers to be evaluated: {list(solvers_to_evaluate.keys())}")
        
    # --- 2. Data Loading ---
    test_data_dir = cfg.paths.data_testing
    if not os.path.exists(test_data_dir) or not os.listdir(test_data_dir):
        logger.error(f"Test data directory is empty or does not exist: {test_data_dir}")
        logger.error("Please run 'generate_data.py' to create test instances first.")
        return
    instance_files = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir) if f.endswith('.csv')]
    raw_results = []

    # --- 3. Run Evaluation Loop ---
    for name, SolverClass in solvers_to_evaluate.items():
        logger.info(f"--- Evaluating Solver: {name} ---")
        try:
            if name == "DNN":
                solver_instance = SolverClass(config=cfg.ml.dnn, device=cfg.ml.device)
            else:
                solver_instance = SolverClass(config={})
            
            for instance_file in tqdm(instance_files, desc=f"Solving with {name}"):
                result = solver_instance.solve(instance_file)
                # Add instance filename as a unique identifier
                raw_results.append({
                    "solver": name,
                    "instance": os.path.basename(instance_file),
                    "n": int(os.path.basename(instance_file).split('_n')[1].split('_')[0]),
                    "value": result.get("value", -1),
                    "time_seconds": result.get("time", -1),
                })
        except Exception as e:
            logger.error(f"Solver '{name}' failed during evaluation. Error: {e}", exc_info=True)

    # --- 4. Process Results and Calculate All Metrics ---
    if not raw_results:
        logger.critical("CRITICAL: No results were generated from any solver. Exiting.")
        sys.exit(1)
        
    results_df = pd.DataFrame(raw_results)
    
    # a. General aggregation for average time and value
    agg_df = results_df.groupby(['solver', 'n']).agg(
        avg_value=('value', 'mean'),
        avg_time_ms=('time_seconds', lambda x: x.mean() * 1000)
    ).reset_index()

    # b. Strict check and detailed error calculation
    baseline_class = cfg.ml.baseline_algorithm
    baseline_name = None
    for name, solver_class in ALGORITHM_REGISTRY.items():
        if solver_class == baseline_class:
            baseline_name = name
            break
            
    if baseline_name and "DNN" in results_df['solver'].unique() and baseline_name in results_df['solver'].unique():
        logger.info(f"Calculating DNN error metrics against baseline '{baseline_name}'...")
        
        # Pivot the raw results to align DNN and baseline values for each instance
        error_pivot_df = results_df.pivot_table(
            index=['n', 'instance'], 
            columns='solver', 
            values='value'
        ).reset_index()
        
        if "DNN" in error_pivot_df.columns and baseline_name in error_pivot_df.columns:
            error_pivot_df.dropna(subset=["DNN", baseline_name], inplace=True)
            
            # Calculate per-instance errors first
            error_pivot_df['absolute_error'] = (error_pivot_df[baseline_name] - error_pivot_df['DNN']).abs()
            error_pivot_df['relative_error'] = (error_pivot_df['absolute_error'] / error_pivot_df[baseline_name].abs().replace(0, 1e-9)).fillna(0)
            error_pivot_df['squared_error'] = error_pivot_df['absolute_error'] ** 2
            
            # Then, group by 'n' to get the final MAE, MRE, and RMSE metrics
            error_summary_df = error_pivot_df.groupby('n').agg(
                mae=('absolute_error', 'mean'),
                mre=('relative_error', lambda x: x.mean() * 100),
                rmse=('squared_error', lambda x: np.sqrt(x.mean()))
            ).reset_index()

            # Merge the calculated error metrics back into the main aggregated DataFrame
            agg_df = pd.merge(agg_df, error_summary_df, on='n', how='left')
        else:
            logger.warning("Could not calculate all error metrics due to incomplete data after pivoting.")
    else:
        logger.warning("Skipping error metric calculation because DNN or baseline results are missing.")

    # --- 5. Save Reports and Generate Plots ---
    logger.info("--- Finalizing Results and Plots ---")
    
    csv_path = os.path.join(cfg.paths.results, "evaluation_full_summary.csv")
    save_results_to_csv(agg_df, csv_path)

    if 'rmse' in agg_df.columns:
        plot_errors_path = os.path.join(cfg.paths.plots, "evaluation_errors_vs_n.png")
        plot_evaluation_errors(agg_df, plot_errors_path)
    else:
        logger.info("Skipping error plot generation as error metrics were not calculated.")
        
    plot_times_path = os.path.join(cfg.paths.plots, "evaluation_times_vs_n.png")
    plot_evaluation_times(agg_df, plot_times_path)

    logger.info("--- Evaluation script finished successfully! ---")

if __name__ == '__main__':
    main()