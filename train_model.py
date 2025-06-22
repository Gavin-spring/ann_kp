# # train_model.py
# import logging
# from src.utils.config_loader import cfg
# from src.utils.logger import setup_logger
# from src.solvers.ml.dnn_solver import DNNSolver

# def main():
#     """
#     This script initializes and runs the training process for an ML solver.
#     """
#     setup_logger(run_name="dnn_training_session", log_dir=cfg.paths.logs)
#     logger = logging.getLogger(__name__)

#     logger.info("Initializing DNN Solver for training...")
    
#     # 1. Get the specific config for the DNN and "inject" it into the solver.
#     dnn_config = cfg.ml.dnn
#     device_str = cfg.ml.device
#     solver = DNNSolver(config=dnn_config, device=device_str)
    
#     # 2. Get the dependencies the train method needs from the global config.
#     baseline_solver_class = cfg.ml.baseline_algorithm
#     training_data_dir = cfg.paths.data_training
#     validation_data_dir = cfg.paths.data_validation
    
#     # 3. Call the train method with its dependencies.
#     solver.train(
#         training_data_dir=training_data_dir,
#         validation_data_dir=validation_data_dir,
#         baseline_solver_class=baseline_solver_class
#     )
    
#     logger.info("Training finished.")

# if __name__ == '__main__':
#     main()

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
    
    # We still use dependency injection for the solver's own config and device
    solver = DNNSolver(config=cfg.ml.dnn, device=cfg.ml.device)
    
    # The call to train is now extremely simple.
    solver.train()
    
    logger.info("Training finished.")

if __name__ == '__main__':
    main()