# Scripts/train_sb3.py
"""
Train a PPO model using Stable Baselines3 with a custom policy for the Knapsack problem.
This script sets up the training environment, defines the custom policy, and starts the training process.
"""

import os
import torch
import warnings
import json
import argparse
import datetime
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
from src.utils.callbacks import CompiledModelSaveCallback

torch.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore", ".*get_linear_fn().*", category=UserWarning)

def main():
    # unique run name
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None, help="A descriptive name for the experiment run.")
    args = parser.parse_args()

    # 使用你的 create_run_name，但我们给它加上实验名
    base_run_name = create_run_name(cfg) # e.g., 20250803_..._RL_n20_lr0.0001
    if args.name:
        # 如果提供了实验名，就用它
        run_name = f"{args.name}-{base_run_name}"
        tb_log_name = args.name # 在TensorBoard中使用简洁的实验名
    else:
        # 否则用原来的长名字
        run_name = base_run_name
        tb_log_name = run_name

    run_dir = os.path.join("artifacts_sb3", "training", run_name)
    os.makedirs(run_dir, exist_ok=True)

    # info of this run
    run_info = {
        "run_name": run_name,
        "model_info": {
            "policy_architecture": cfg.ml.rl.ppo.hyperparams.policy_architecture,
            "critic_details": cfg.ml.rl.ppo.hyperparams.critic_details,
            "description": cfg.ml.rl.ppo.hyperparams.description,
            "data_range": cfg.ml.rl.ppo.hyperparams.data_range,
            "data_type": cfg.ml.rl.ppo.hyperparams.data_type,
            "data_paths":{
                "training": cfg.paths.data_training,
                "validation": cfg.paths.data_validation,
            },
        },
        "hyperparams": {
            "use_shaping_reward": cfg.ml.rl.ppo.hyperparams.use_shaping_reward,
            "VecNormalize_obs": cfg.ml.rl.ppo.hyperparams.VecNormalize.obs,
            "VecNormalize_reward": cfg.ml.rl.ppo.hyperparams.VecNormalize.reward,
            "train_max_n": cfg.ml.rl.ppo.hyperparams.max_n,
            "eval_max_n": cfg.ml.rl.ppo.hyperparams.eval_max_n,
            "n_glimpses": cfg.ml.rl.ppo.hyperparams.n_glimpses,
            "embedding_dim": cfg.ml.rl.ppo.hyperparams.embedding_dim,
            "nhead": cfg.ml.rl.ppo.hyperparams.nhead,
            "num_layers": cfg.ml.rl.ppo.hyperparams.num_layers,
            "n_process_block_iters": cfg.ml.rl.ppo.hyperparams.n_process_block_iters,
        },
        "params": {
            "total_timesteps": cfg.ml.rl.ppo.training.total_timesteps,
            "eval_freq": cfg.ml.rl.ppo.training.eval_freq,
            "learning_rate_initial": cfg.ml.rl.ppo.training.learning_rate_initial,
            "learning_rate_final": cfg.ml.rl.ppo.training.learning_rate_final,
            "n_steps": cfg.ml.rl.ppo.training.n_steps,
            "batch_size": cfg.ml.rl.ppo.training.batch_size,
            "n_epochs": cfg.ml.rl.ppo.training.n_epochs,
            "gamma": cfg.ml.rl.ppo.training.gamma,
            "gae_lambda": cfg.ml.rl.ppo.training.gae_lambda,
            "clip_range": cfg.ml.rl.ppo.training.clip_range,
            "vf_coef": cfg.ml.rl.ppo.training.vf_coef,
            "entropy_coef": cfg.ml.rl.ppo.training.ent_coef,
            "max_grad_norm": cfg.ml.rl.ppo.training.max_grad_norm,
        },
    }
    with open(os.path.join(run_dir, "run_info.json"), 'w') as f:
        json.dump(run_info, f, indent=4)

    # Define directories for models and logs
    models_dir = os.path.join(run_dir, "models")
    logs_dir = os.path.join(run_dir, "logs") # evaluation.npz
    unified_tensorboard_log_dir = os.path.join("artifacts_sb3", "tensorboard_logs")

    print(f"--- Starting new training run: {run_name} ---")
    print(f"All artifacts will be saved in: {run_dir}")
    
    # learning rate schedule
    start_lr = cfg.ml.rl.ppo.training.learning_rate_initial
    end_lr = cfg.ml.rl.ppo.training.learning_rate_final
    lr_schedule = get_linear_fn(start_lr, end_lr, 1.0)

    # 1. create training and validation environments
    env_kwargs = {
        "data_dir": cfg.paths.data_training,
        "max_n": cfg.ml.rl.ppo.hyperparams.max_n,
        "max_weight": cfg.ml.generation.max_weight,
        "max_value": cfg.ml.generation.max_value,
        "use_shaping_reward": cfg.ml.rl.ppo.hyperparams.use_shaping_reward,
    }
    norm_obs_keys = ["items", "capacity"]
    # Training environment
    train_env_unwrapped = make_vec_env(KnapsackEnv, n_envs=4, env_kwargs=env_kwargs)
    train_env = VecNormalize(train_env_unwrapped, 
                       norm_obs=cfg.ml.rl.ppo.hyperparams.VecNormalize.obs, 
                       norm_obs_keys=norm_obs_keys,
                       norm_reward=cfg.ml.rl.ppo.hyperparams.VecNormalize.reward,
                       gamma=cfg.ml.rl.ppo.training.gamma)

    # Validation environment
    val_env_kwargs = env_kwargs.copy()
    val_env_kwargs["data_dir"] = cfg.paths.data_validation    
    val_env_unwrapped = make_vec_env(KnapsackEnv, n_envs=4, env_kwargs=val_env_kwargs)
    val_env = VecNormalize(val_env_unwrapped, 
                           training=False, 
                           norm_obs=cfg.ml.rl.ppo.hyperparams.VecNormalize.obs, 
                           norm_obs_keys=norm_obs_keys,
                           norm_reward=cfg.ml.rl.ppo.hyperparams.VecNormalize.reward, 
                           gamma=cfg.ml.rl.ppo.training.gamma)

    # 2. define evaluation callback
    eval_callback = CompiledModelSaveCallback(val_env, 
                                 best_model_save_path=models_dir,
                                 log_path=logs_dir,
                                 eval_freq=cfg.ml.rl.ppo.training.eval_freq,
                                 deterministic=True, render=False)

    # 3. define the custom policy
    policy_kwargs = dict(
        features_extractor_class=KnapsackEncoder,
        features_extractor_kwargs=dict(
            embedding_dim=cfg.ml.rl.ppo.hyperparams.embedding_dim,
            nhead=cfg.ml.rl.ppo.hyperparams.nhead,
            num_layers=cfg.ml.rl.ppo.hyperparams.num_layers,
        ),
        n_process_block_iters=cfg.ml.rl.ppo.hyperparams.n_process_block_iters,
    )

    # 4. initialize the PPO model with the custom policy
    model = PPO(
        KnapsackActorCriticPolicy,
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=unified_tensorboard_log_dir,
        learning_rate=lr_schedule,
        n_steps=cfg.ml.rl.ppo.training.n_steps,
        batch_size=cfg.ml.rl.ppo.training.batch_size,
        n_epochs=cfg.ml.rl.ppo.training.n_epochs,
        gamma=cfg.ml.rl.ppo.training.gamma,
        gae_lambda=cfg.ml.rl.ppo.training.gae_lambda,
        clip_range=cfg.ml.rl.ppo.training.clip_range,
        vf_coef= cfg.ml.rl.ppo.training.vf_coef,
        ent_coef=cfg.ml.rl.ppo.training.ent_coef,
        max_grad_norm=cfg.ml.rl.ppo.training.max_grad_norm,
    )

    print("Compiling the model with triton...")
    model.policy = torch.compile(model.policy) # triton compile the model for performance
    print("Model compiled.")

    # 5. start training
    print("start training...")
    model.learn(total_timesteps=cfg.ml.rl.ppo.training.total_timesteps, callback=eval_callback, tb_log_name=tb_log_name)

    # 6. save the final model and VecNormalize stats
    print("Training complete. Saving VecNormalize stats and final model...")

    stats_path = os.path.join(models_dir, "vec_normalize.pkl")
    train_env.save(stats_path)
    model_path = os.path.join(models_dir, "final_model.zip")
    model.save(model_path)

    print(f"Model and stats saved successfully to {models_dir}")

if __name__ == '__main__':
    main()