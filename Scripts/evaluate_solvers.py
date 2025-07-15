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
    # FIX 1: Check for dnn_model_path specifically
    if args.dnn_model_path:
        logger.info(f"Using specified DNN model: {args.dnn_model_path}")
    # Add a log for the RL model path as well
    if args.rl_model_path:
        logger.info(f"Using specified RL model: {args.rl_model_path}")

    # --- 2. Setup Solvers ---
    # We will test all solvers currently active in the registry.
    solvers_to_evaluate = ALGORITHM_REGISTRY.copy()
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

    # FIX 2: Check for dnn_model_path, not model_path
    if "DNN" in solvers_to_evaluate and not args.dnn_model_path:
        logger.critical("CRITICAL ERROR: The 'DNN' solver is active, but no --dnn-model-path was provided.")
        # Update the help message to be correct
        logger.critical("Please specify the model path using: --dnn-model-path <path_to_your_model.pth>")
        sys.exit(1)

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
    for name, SolverClass in solvers_to_evaluate.items():
        logger.info(f"--- Evaluating Solver: {name} ---")
        try:
            solver_instance = None
            
            # check if the solver is a machine learning(approximation) model
            if name in vars(cfg.ml.approximation_solvers):
                # 1. obtain the exact config key for the solver
                config_key = getattr(cfg.ml.approximation_solvers, name)
                
                # 2. obtain the model path from the command line arguments
                model_path_arg = getattr(args, f"{config_key}_model_path")
                
                if not model_path_arg:
                    logger.warning(f"Skipping {name} solver because --{config_key}-model-path was not provided.")
                    continue

                # 3. load the configuration for the solver
                config = getattr(cfg.ml, config_key)
                solver_config = copy.deepcopy(config)

                # 4. --training-max-n is only relevant for DNN
                if name == "DNN" and args.training_max_n:
                    logger.info(f"Overriding model input size using --training-max-n={args.training_max_n}")
                    old_input_size = (args.training_max_n * solver_config.hyperparams.input_size_factor) + solver_config.hyperparams.input_size_plus
                    solver_config.hyperparams.input_size = old_input_size
                    logger.info(f"Model will be loaded with input_size: {old_input_size}")
                
                # 5. create the solver instance
                solver_instance = SolverClass(config=solver_config, device=cfg.ml.device, model_path=model_path_arg)
            
            else:
                # For non-ML solvers, we can instantiate them directly
                solver_instance = SolverClass(config={})

            if solver_instance is None:
                continue

            for instance_file in tqdm(instance_files, desc=f"Solving with {name}"):
                result = solver_instance.solve(instance_file)
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
    
    agg_df = results_df.groupby(['solver', 'n']).agg(
        avg_value=('value', 'mean'),
        avg_time_ms=('time_seconds', lambda x: x.mean() * 1000)
    ).reset_index()

    baseline_class = cfg.ml.baseline_algorithm
    baseline_name = None
    for name, solver_class in ALGORITHM_REGISTRY.items():
        if solver_class == baseline_class:
            baseline_name = name
            break

    # Check if the approximation solver is in the results
    ml_solvers_in_results = set(vars(cfg.ml.approximation_solvers).keys()) & set(results_df['solver'].unique())

    if baseline_name and ml_solvers_in_results:
        # calculate error metrics first
        all_error_dfs = []
        
        # error pivot table
        error_pivot_df = results_df.pivot_table(
            index=['n', 'instance'], 
            columns='solver', 
            values='value'
        ).reset_index()

        for ml_solver_name in ml_solvers_in_results:
            logger.info(f"Calculating {ml_solver_name} error metrics against baseline '{baseline_name}'...")
            
            if ml_solver_name in error_pivot_df.columns and baseline_name in error_pivot_df.columns:
                # create a temporary DataFrame for error calculations
                temp_df = error_pivot_df.copy()
                temp_df.dropna(subset=[ml_solver_name, baseline_name], inplace=True)
                
                temp_df['absolute_error'] = (temp_df[baseline_name] - temp_df[ml_solver_name]).abs()
                temp_df['relative_error'] = (temp_df['absolute_error'] / temp_df[baseline_name].abs().replace(0, 1e-9)).fillna(0)
                temp_df['squared_error'] = temp_df['absolute_error'] ** 2
                
                error_summary_df = temp_df.groupby('n').agg(
                    mae=('absolute_error', 'mean'),
                    mre=('relative_error', lambda x: x.mean() * 100),
                    rmse=('squared_error', lambda x: np.sqrt(x.mean()))
                ).reset_index()
                
                # rename columns to include solver name
                error_summary_df = error_summary_df.add_prefix(f"{ml_solver_name}_")
                error_summary_df.rename(columns={f"{ml_solver_name}_n": "n"}, inplace=True)
                
                all_error_dfs.append(error_summary_df)
            else:
                logger.warning(f"Could not calculate error metrics for {ml_solver_name} due to incomplete data.")

        # merge all error DataFrames into the main aggregation DataFrame
        if all_error_dfs:
            for error_df in all_error_dfs:
                agg_df = pd.merge(agg_df, error_df, on='n', how='left')

    else:
        logger.warning("Skipping error metric calculation because baseline or ML solver results are missing.")

    # --- 6. Save Reports and Generate Plots ---
    logger.info("--- Finalizing Results and Plots ---")

    csv_path = os.path.join(run_dir, "evaluation_full_summary.csv")
    save_results_to_csv(agg_df, csv_path)

    # 1. list of approximation solvers that have error metrics
    solvers_with_errors = [s for s in ml_solvers_in_results if f"{s}_rmse" in agg_df.columns]

    # 2. if there are solvers with errors, generate error plots
    if solvers_with_errors:
        logger.info(f"Generating error plots for solvers: {solvers_with_errors}")
        plot_errors_path = os.path.join(run_dir, "evaluation_errors_vs_n.png")
        # 3. plot the errors
        plot_evaluation_errors(agg_df, plot_errors_path, solvers_with_errors)
    else:
        logger.info("Skipping error plot generation as no valid error metrics were calculated.")

    plot_times_path = os.path.join(run_dir, "evaluation_times_vs_n.png")
    plot_evaluation_times(agg_df, plot_times_path)

    logger.info("--- Evaluation script finished successfully! ---")

if __name__ == '__main__':
    main()