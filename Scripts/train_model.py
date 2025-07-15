# train_model.py
import logging
import os
# import argparse
from src.utils.config_loader import cfg
from src.utils.logger import setup_logger
# from src.solvers.ml.dnn_solver import DNNSolver
from src.utils.run_utils import create_run_name

def main():
    """
    This script initializes and runs the training process for an ML solver,
    saving all artifacts into a unique, timestamped directory.
    """
    # --- 1. Create a unique name and directory for this training run ---
    run_name = create_run_name(cfg)
    run_dir = os.path.join(cfg.paths.artifacts, "runs", "training", run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Setup logger to save to the new run-specific directory
    setup_logger(run_name="training_log", log_dir=run_dir)
    logger = logging.getLogger(__name__)
    
    logger.info(f"--- Starting New Training Run: {run_name} ---")
    logger.info(f"All artifacts for this run will be saved in: {run_dir}")

    # --- 2. Initialize Solver based on config ---
    if cfg.ml.training_mode == "DNN":
        from src.solvers.ml.dnn_solver import DNNSolver
        solver = DNNSolver(config=cfg.ml.dnn, device=cfg.ml.device)
    elif cfg.ml.training_mode == "RL":
        from src.solvers.ml.rl_solver import RLSolver
        solver = RLSolver(config=cfg.ml.rl, device=cfg.ml.device)
    else:
        raise ValueError(f"Unsupported training mode: {cfg.ml.training_mode}")
    
    # --- 3. Define artifact paths and train ---
    model_save_path = os.path.join(run_dir, "best_model.pth")
    plot_save_path = os.path.join(run_dir, "loss_curve.png")
    
    solver.train(
        model_save_path=model_save_path,
        plot_save_path=plot_save_path,
    )
    
    logger.info(f"--- Training Run {run_name} Finished. ---")

if __name__ == '__main__':
    main()