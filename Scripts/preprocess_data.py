# preprocess_data.py
import os
import torch
import logging
from tqdm import tqdm
import sys
import argparse
from collections import defaultdict

from src.utils.config_loader import cfg
from src.utils.logger import setup_logger
from src.solvers.ml.feature_extractor import extract_features_from_instance

def main():
    """
    Pre-processes the raw instance data into training-ready tensors.
    """
    # --- Argument parsing ---
    parser = argparse.ArgumentParser(description="Preprocess knapsack problem datasets.")
    parser.add_argument(
        '--dataset',
        type=str,
        nargs='+',  # one or more arguments
        required=True, # mandatory argument
        choices=['training', 'validation', 'testing', 'all'],
        help="Specify the dataset(s) to preprocess. Options: 'training', 'validation', 'testing', 'all'."
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None, # no limit by default
        help="Limit the number of instances to process for EACH problem size 'n'. Processes all if not set."
    )
    args = parser.parse_args()
    
    setup_logger(run_name="preprocessing", log_dir=cfg.paths.logs)
    logger = logging.getLogger(__name__)

    # --- Get the baseline solver from the config file ---
    try:
        # The baseline_algorithm attribute is now a CLASS, not a string.
        baseline_solver_class = cfg.ml.baseline_algorithm
        if baseline_solver_class is None:
            raise ValueError("Baseline algorithm was not found in the registry.")
        baseline_solver = baseline_solver_class()
    except (AttributeError, ValueError) as e:
        logger.critical(f"Could not initialize baseline solver. Error: {e}")
        logger.critical("Check 'ml.baseline_algorithm' in config.yaml and ALGORITHM_REGISTRY in config_loader.py.")
        sys.exit(1)

    # --- Define directories to process ---
    all_data_sources = {
        "training": cfg.paths.data_training,
        "validation": cfg.paths.data_validation,
        "testing": cfg.paths.data_testing
    }
    sources_to_process = {}
    if 'all' in args.dataset:
        sources_to_process = all_data_sources
    else:
        for name in args.dataset:
            if name in all_data_sources:
                sources_to_process[name] = all_data_sources[name]

    if not sources_to_process:
        logger.error(f"Invalid dataset names specified: {args.dataset}. Please choose from {list(all_data_sources.keys())} or 'all'.")
        sys.exit(1)

    for purpose, data_dir in sources_to_process.items():
        logger.info(f"--- Starting preprocessing for '{purpose}' dataset in '{data_dir}' ---")
    
        all_instance_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
        if not all_instance_files:
            logger.warning(f"No instance files found in {data_dir}. Skipping.")
            continue

        grouped_instances = defaultdict(list)
        for path in all_instance_files:
            try:
                # Extract the 'n' value from the filename
                n_val = int(os.path.basename(path).split('_n')[1].split('_')[0])
                grouped_instances[n_val].append(path)
            except (IndexError, ValueError):
                logger.warning(f"Could not parse 'n' value from filename: {os.path.basename(path)}. Skipping this file.")
                continue

        # 4. Limit the number of instances per 'n' if specified
        final_files_to_process = []
        for n_val in sorted(grouped_instances.keys()):
            files_for_n = grouped_instances[n_val]
            # If a limit is specified, slice the list to only include the first 'limit' files
            if args.limit is not None:
                files_for_n = files_for_n[:args.limit]
            final_files_to_process.extend(files_for_n)
        
        if args.limit is not None:
            logger.info(f"Processing a maximum of {args.limit} instances per 'n'. Total files to process: {len(final_files_to_process)}.")

        processed_data = []
        for instance_path in tqdm(final_files_to_process, desc=f"Processing {purpose} data"):
            # 1. Get normalized features for the model using the new feature extractor
            features_tensor = extract_features_from_instance(instance_path, config=cfg.ml.dnn)
            if features_tensor is None:
                logger.warning(f"Skipping instance {instance_path} due to feature extraction error.")
                continue

            # 2. Get the optimal solution (label) by running the baseline solver on the same path
            optimal_result = baseline_solver.solve(instance_path)
            if optimal_result.get('value', -1) == -1:
                logger.warning(f"Skipping instance {instance_path} due to baseline solver error.")
                continue
            
            # 3. Normalize the label
            normalized_label = optimal_result['value'] / cfg.ml.dnn.hyperparams.target_scale_factor

            # 4. Append the feature tensor and label tensor to our list
            processed_data.append({
                'instance': os.path.basename(instance_path),
                'features': features_tensor,
                'label': torch.tensor([normalized_label]).float(),
                'optimal_value': optimal_result['value'], # add original optimal value
                'solve_time': optimal_result.get('time', 0)
            })

        # 5. Save the entire list of processed data to a single, fast-loading file
        if processed_data:
            output_path = os.path.join(cfg.paths.data, f"processed_{purpose}.pt")
            torch.save(processed_data, output_path)
            logger.info(f"Successfully pre-processed {len(processed_data)} instances.")
            logger.info(f"Training-ready dataset saved to: {output_path}")
        else:
            logger.warning(f"No data was processed for '{purpose}' set.")


    logger.info("--- Preprocessing complete! ---")

if __name__ == '__main__':
    main()