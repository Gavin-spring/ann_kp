# Scripts/train_sb3.py
"""
Train a PPO model using Stable Baselines3 with a custom policy for the Knapsack problem.
This script sets up the training environment, defines the custom policy, and starts the training process.
"""

import os
import torch
import warnings
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import VecNormalize

from src.utils.config_loader import cfg
from src.utils.run_utils import create_run_name
from src.utils.logger import setup_logger
from src.env.knapsack_env import KnapsackEnv
from src.solvers.ml.custom_policy import KnapsackActorCriticPolicy, KnapsackEncoder

torch.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore", ".*get_linear_fn().*", category=UserWarning)

def main():
    # Create a unique run name based on the current configuration
    run_name = create_run_name(cfg)
    run_dir = os.path.join("artifacts_sb3", "training", run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Define directories for models and logs
    models_dir = os.path.join(run_dir, "models")
    logs_dir = os.path.join(run_dir, "logs") # evaluation.npz
    unified_tensorboard_log_dir = os.path.join("artifacts_sb3", "tensorboard_logs")

    print(f"--- Starting new training run: {run_name} ---")
    print(f"All artifacts will be saved in: {run_dir}")
    
    # learning rate schedule
    start_lr = cfg.ml.rl.training.learning_rate
    end_lr = 0.00001
    lr_schedule = get_linear_fn(start_lr, end_lr, 1.0)

    # 1. create training and validation environments
    env_kwargs = {
        "data_dir": cfg.paths.data_training,
        "max_n": cfg.ml.rl.hyperparams.max_n,
        "max_weight": cfg.ml.generation.max_weight,
        "max_value": cfg.ml.generation.max_value,
    }
    norm_obs_keys = ["items", "capacity"]
    # Training environment
    train_env_unwrapped = make_vec_env(KnapsackEnv, n_envs=4, env_kwargs=env_kwargs)
    train_env = VecNormalize(train_env_unwrapped, 
                       norm_obs=True, 
                       norm_obs_keys=norm_obs_keys,
                       norm_reward=True, 
                       gamma=cfg.ml.rl.training.gamma)

    # Validation environment
    val_env_kwargs = env_kwargs.copy()
    val_env_kwargs["data_dir"] = cfg.paths.data_validation    
    val_env_unwrapped = make_vec_env(KnapsackEnv, n_envs=4, env_kwargs=val_env_kwargs)
    val_env = VecNormalize(val_env_unwrapped, 
                           training=False, 
                           norm_obs=True, 
                           norm_obs_keys=norm_obs_keys,
                           norm_reward=False, 
                           gamma=cfg.ml.rl.training.gamma)

    # 2. define evaluation callback
    eval_callback = EvalCallback(val_env, 
                                 best_model_save_path=models_dir,
                                 log_path=logs_dir,
                                 eval_freq=5000,
                                 deterministic=True, render=False)

    # 3. define the custom policy
    policy_kwargs = dict(
        features_extractor_class=KnapsackEncoder,
        features_extractor_kwargs=dict(
            embedding_dim=cfg.ml.rl.hyperparams.embedding_dim,
            nhead=4,
            num_layers=2,
        )
    )

    # 4. initialize the PPO model with the custom policy
    model = PPO(
        KnapsackActorCriticPolicy,
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=unified_tensorboard_log_dir,
        learning_rate=lr_schedule,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=cfg.ml.rl.training.gamma,
        gae_lambda=cfg.ml.rl.training.gae_lambda,
        clip_range=cfg.ml.rl.training.clip_param,
    )
    
    model.policy = torch.compile(model.policy) # triton compile the model for performance

    # 5. start training
    print("start training...")
    model.learn(total_timesteps=500_000, callback=eval_callback, tb_log_name=run_name)

    # 6. save the final model and VecNormalize stats
    print("Training complete. Saving VecNormalize stats and final model...")

    stats_path = os.path.join(models_dir, "vec_normalize.pkl")
    train_env.save(stats_path)
    model_path = os.path.join(models_dir, "final_model.zip")
    model.save(model_path)

    print(f"Model and stats saved successfully to {models_dir}")

if __name__ == '__main__':
    main()