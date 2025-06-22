# src/utils/run_utils.py
import datetime
from types import SimpleNamespace

def create_run_name(config: SimpleNamespace) -> str:
    """
    Creates a unique and informative name for an experiment run.

    Args:
        config (SimpleNamespace): The configuration object for the run.

    Returns:
        str: A unique name, e.g., '20250622_210000_n1000_lr0.001'
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract key parameters from the config to make the name informative
    try:
        max_n = config.ml.dnn.generation.end_n
        lr = config.ml.dnn.training.learning_rate
        run_name = f"{timestamp}_n{max_n}_lr{lr}"
    except AttributeError:
        # Fallback for general runs without dnn config
        run_name = timestamp
        
    return run_name