# evaluate_solvers.py
import logging
import os
import sys
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
import torch
from types import SimpleNamespace
import copy

from src.utils.config_loader import cfg, ALGORITHM_REGISTRY
from src.utils.logger import setup_logger
from src.evaluation.plotting import plot_evaluation_errors, plot_evaluation_times
from src.evaluation.reporting import save_results_to_csv
from src.utils.run_utils import create_run_name

def main():
    """
    This script evaluates all solvers, strictly ensures error metrics can be
    calculated, and then generates reports and plots.
    """
    # --- Setup Argument Parser ---
    parser = argparse.ArgumentParser(description="Evaluate knapsack problem solvers.")
    parser.add_argument(
        "--dnn-model-path",
        type=str,
        default=None,
        help="Path to a pre-trained .pth model file for the DNNSolver."
    )
    parser.add_argument(
        "--rl-model-path",
        type=str,
        default=None,
        help="Path to a pre-trained .pth model file for the RLSolver."
    )
    parser.add_argument(
        "--training-max-n",
        type=int,
        default=None,
        help="The 'max_n_for_architecture' value that was used when a model was trained."
    )
    args = parser.parse_args()
    
    # --- 1. Create a unique name and directory for this evaluation run ---
    run_name = create_run_name(cfg)
    run_dir = os.path.join(cfg.paths.artifacts, "runs", "evaluation", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    setup_logger(run_name="evaluation_session", log_dir=run_dir)
    logger = logging.getLogger(__name__)
    
    logger.info(f"--- Starting New Evaluation Run: {run_name} ---")
    if args.model_path:
        logger.info(f"Using specified DNN model: {args.model_path}")

    # --- 2. Setup Solvers ---
    # We will test all solvers currently active in the registry.
    solvers_to_evaluate = ALGORITHM_REGISTRY.copy()  # Create a copy to avoid modifying the original registry.
    if not solvers_to_evaluate:
        logger.critical("No solvers are active in ALGORITHM_REGISTRY. Exiting.")
        sys.exit(1)
    logger.info(f"Solvers to be evaluated: {list(solvers_to_evaluate.keys())}")
        
    # --- 3. Data Loading ---
    test_data_dir = cfg.paths.data_testing
    if not os.path.exists(test_data_dir) or not os.listdir(test_data_dir):
        logger.error(f"Test data directory is empty or does not exist: {test_data_dir}")
        logger.error("Please run 'generate_data.py' to create test instances first.")
        return
    instance_files = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir) if f.endswith('.csv')]
    raw_results = []

    # If 'DNN' is set to be evaluated, the model path argument becomes mandatory.
    if "DNN" in solvers_to_evaluate and not args.model_path:
        logger.critical("CRITICAL ERROR: The 'DNN' solver is active, but no --model-path was provided.")
        logger.critical("Please specify the model path using: --model-path <path_to_your_model.pth>")
        sys.exit(1) # Exit the script immediately to prevent wasting time.

    # a. Get the baseline solver
    baseline_class = cfg.ml.baseline_algorithm
    baseline_name = None
    for name, solver_class in ALGORITHM_REGISTRY.items():
        if solver_class == baseline_class:
            baseline_name = name
            break

    # b. if a baseline solver is specified and preprocessed data exists, load it
    preprocessed_test_path = os.path.join(cfg.paths.data, "processed_testing.pt")
    if baseline_name and baseline_name in solvers_to_evaluate and os.path.exists(preprocessed_test_path):
        logger.info(f"Attempting to load pre-calculated baseline results for '{baseline_name}'...")
        try:
            preloaded_data = torch.load(preprocessed_test_path)
            
            # c. transform the preloaded data into the expected format
            for record in preloaded_data:
                raw_results.append({
                    "solver": baseline_name,
                    "instance": record['instance'],
                    "n": int(record['instance'].split('_n')[1].split('_')[0]),
                    "value": record['optimal_value'],
                    "time_seconds": record['solve_time'],
                })
            
            # d. remove the baseline solver from the evaluation list
            del solvers_to_evaluate[baseline_name]
            logger.info(f"Successfully loaded {len(preloaded_data)} results for '{baseline_name}'. It will be skipped in the main loop.")

        except Exception as e:
            logger.warning(f"Could not load or parse pre-calculated baseline results from {preprocessed_test_path}. "
                           f"The baseline solver '{baseline_name}' will be run live. Error: {e}")

    # --- 4. Run Evaluation Loop ---
    baseline_class = cfg.ml.baseline_algorithm
    baseline_name = [name for name, solver_class in ALGORITHM_REGISTRY.items() if solver_class == baseline_class][0]

    preprocessed_test_path = os.path.join(cfg.paths.data, "processed_testing.pt")

    for name, SolverClass in solvers_to_evaluate.items():
        logger.info(f"--- Evaluating Solver: {name} ---")
        # If the solver is the baseline and preprocessed file exists, load from file instead of re-solving.
        if name == baseline_name and os.path.exists(preprocessed_test_path):
            logger.info(f"Loading pre-calculated baseline results from {preprocessed_test_path}")
            preprocessed_data = torch.load(preprocessed_test_path)
        try:
            solver_instance = None # Initialize to None
            if name == "DNN":
                if not args.dnn_model_path:
                    logger.warning(f"Skipping DNN solver because no --dnn-model-path was provided.")
                    continue
                # training_max_n is used to adjust the input size of the DNN model.
                dnn_config = copy.deepcopy(cfg.ml.dnn) # create a copy to avoid modifying the original config

                if args.training_max_n:
                    logger.info(f"Overriding model input size using --training-max-n={args.training_max_n}")
                    
                    # adjust the input size based on the training_max_n
                    old_input_size = (args.training_max_n * dnn_config.hyperparams.input_size_factor) + dnn_config.hyperparams.input_size_plus
                    
                    # update the input size in the config
                    dnn_config.hyperparams.input_size = old_input_size
                    logger.info(f"Model will be loaded with input_size: {old_input_size}")

                # instantiate the solver with the DNN configuration
                solver_instance = SolverClass(config=dnn_config, device=cfg.ml.device, model_path=args.model_path)

            elif name == "PointerNet RL":
                if not args.rl_model_path:
                    logger.warning(f"Skipping PointerNet RL solver because no --rl-model-path was provided.")
                    continue
                # Instantiate RLSolver with its specific config and model path
                rl_config = cfg.ml.rl 
                solver_instance = SolverClass(config=rl_config, device=cfg.ml.device, model_path=args.rl_model_path)
            
            else:
                solver_instance = SolverClass(config={})

            if solver_instance is None:
                continue # Skip if solver wasn't instantiated

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

    # --- 5. Process Results and Calculate All Metrics ---
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

    # --- 6. Save Reports and Generate Plots ---
    logger.info("--- Finalizing Results and Plots ---")
    
    csv_path = os.path.join(run_dir, "evaluation_full_summary.csv")
    save_results_to_csv(agg_df, csv_path)

    if 'rmse' in agg_df.columns:
        plot_errors_path = os.path.join(run_dir, "evaluation_errors_vs_n.png")
        plot_evaluation_errors(agg_df, plot_errors_path)
    else:
        logger.info("Skipping error plot generation as error metrics were not calculated.")
        
    plot_times_path = os.path.join(run_dir, "evaluation_times_vs_n.png")
    plot_evaluation_times(agg_df, plot_times_path)

    logger.info("--- Evaluation script finished successfully! ---")

if __name__ == '__main__':
    main()