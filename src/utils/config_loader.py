# src/utils/config_loader.py
import yaml
import os
import torch
from types import SimpleNamespace
from typing import Dict, Any

# --- Import solver CLASSes here ---
from src.solvers.classic.dp_solver import DPSolver2D, DPSolver1D, DPValueSolver
from src.solvers.classic.gurobi_solver import GurobiSolver
from src.solvers.classic.heuristic_solvers import GreedySolver, BranchAndBoundSolver
from src.solvers.ml.dnn_solver import DNNSolver

# The registry now maps a name to a Solver Class.
ALGORITHM_REGISTRY = {
    "2D DP": DPSolver2D,
    "1D DP (Optimized)": DPSolver1D,
    "1D DP (on value)": DPValueSolver,
    "Gurobi": GurobiSolver,
    "Greedy": GreedySolver,
    "Branch and Bound": BranchAndBoundSolver,
    "DNN": DNNSolver,
}

def _post_process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes the raw config dict to add dynamic values and absolute paths.
    This function contains all logic that cannot be represented in a static YAML file.
    """
    # --- 1. Define Project Root and Build Absolute Paths ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # a. Build absolute paths for all entries in the 'paths' section
    for key, rel_path in config_dict['paths'].items():
        config_dict['paths'][key] = os.path.join(project_root, rel_path)
    # Add project_root to paths config for easy access elsewhere
    config_dict['paths']['root'] = project_root

    # --- 2. Dynamic Calculation of ML Hyperparameters ---
    dnn_gen_cfg = config_dict['ml']['dnn']['generation']
    dnn_hyper_cfg = config_dict['ml']['dnn']['hyperparams']
    data_gen_cfg = config_dict['data_gen']
    
    max_n = dnn_gen_cfg['end_n']
    dnn_hyper_cfg['max_n'] = max_n
    dnn_hyper_cfg['input_size'] = max_n * dnn_hyper_cfg['input_size_factor'] + dnn_hyper_cfg['input_size_plus']
    dnn_hyper_cfg['target_scale_factor'] = float(max_n * data_gen_cfg['max_value'] * dnn_hyper_cfg['target_scale_factor_multiplier'])

    # --- 3. Auto-detect Hardware Device ---
    if config_dict['ml']['device'] == 'auto':
        config_dict['ml']['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 4. Map Algorithm Names to Functions ---
    classic_cfg = config_dict['classic_solvers']
    try:
        classic_cfg['algorithms_to_test'] = {
            name: ALGORITHM_REGISTRY[name] for name in classic_cfg['algorithms_to_test']
        }
        classic_cfg['baseline_algorithm'] = ALGORITHM_REGISTRY[classic_cfg['baseline_algorithm']]
    except KeyError as e:
        raise ValueError(f"Algorithm '{e.args[0]}' is defined in config.yaml but not found in ALGORITHM_REGISTRY in config_loader.py.") from e

    return config_dict

def load_config(config_path: str = 'configs/config.yaml') -> SimpleNamespace:
    """
    Loads, processes, and returns the project configuration from a YAML file
    as a SimpleNamespace object for dot notation access.
    """
    # Define project root relative to this file's location
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    full_config_path = os.path.join(project_root, config_path)
    
    try:
        with open(full_config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at: {full_config_path}")
    
    # Process the loaded dictionary to add dynamic values
    processed_config = _post_process_config(config_dict)

    # Convert the final dictionary to a SimpleNamespace for easy attribute access
    def dict_to_namespace(d: Dict) -> SimpleNamespace:
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = dict_to_namespace(v)
        return SimpleNamespace(**d)

    return dict_to_namespace(processed_config)

# --- Create a single, global config instance for easy import across the project ---
# Other modules can simply use: from src.utils.config_loader import cfg
cfg = load_config()