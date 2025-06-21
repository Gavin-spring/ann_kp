# train_model.py
import logging
from src.utils.config_loader import cfg
from src.utils.logger import setup_logger
from src.solvers.ml.dnn_solver import DNNSolver

def main():
    """
    This script initializes and runs the training process for an ML solver.
    """
    setup_logger(run_name="dnn_training_session", log_dir=cfg.paths.logs)
    logger = logging.getLogger(__name__)

    logger.info("Initializing DNN Solver for training...")
    # The DNNSolver class now contains all the complex training logic.
    # We just need to create an instance and call the train method.
    solver = DNNSolver()
    
    # This single call will execute the entire curriculum training loop.
    solver.train()
    
    logger.info("Script finished.")

if __name__ == '__main__':
    main()