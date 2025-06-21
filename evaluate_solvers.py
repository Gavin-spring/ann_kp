# evaluate_solvers.py
import logging
import os
from src.utils.config_loader import cfg
from src.utils.logger import setup_logger
from tqdm import tqdm

def main():
    """
    This script evaluates all solvers defined in the config file on a test dataset.
    """
    setup_logger(run_name="evaluation_session", log_dir=cfg.paths.logs)
    logger = logging.getLogger(__name__)

    # The config loader has already mapped names to classes
    solvers_to_test = cfg.classic_solvers.algorithms_to_test
    # Add ML solvers to the list if you want to evaluate them too
    solvers_to_test["DNN"] = cfg.ALGORITHM_REGISTRY["DNN"]

    test_data_dir = cfg.paths.data_testing
    instance_files = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir) if f.endswith('.csv')]

    results = {}

    for name, SolverClass in solvers_to_test.items():
        logger.info(f"--- Evaluating Solver: {name} ---")
        solver_instance = SolverClass()
        results[name] = []
        
        for instance_file in tqdm(instance_files, desc=f"Solving with {name}"):
            result = solver_instance.solve(instance_file)
            results[name].append(result)
    
    # --- Process and print results ---
    logger.info("--- Evaluation Complete: Summary ---")
    for name, res_list in results.items():
        avg_time = sum(r['time'] for r in res_list) / len(res_list)
        avg_value = sum(r['value'] for r in res_list) / len(res_list)
        logger.info(f"Solver: {name:<20} | Avg Value: {avg_value:<15.2f} | Avg Time: {avg_time*1000:<15.4f} ms")
    
    # Here you can add logic to save results to a CSV and generate plots comparing all solvers.

if __name__ == '__main__':
    main()