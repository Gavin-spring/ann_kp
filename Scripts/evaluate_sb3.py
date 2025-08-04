# Scripts/evaluate_sb3.py

import os
import time
import pandas as pd
from tqdm import tqdm
import warnings
import argparse
import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from src.utils.config_loader import cfg, ALGORITHM_REGISTRY
from src.env.knapsack_env import KnapsackEnv
from src.utils.run_utils import create_run_name
from src.evaluation.plotting import plot_results

# --- Main Function ---
def main():
    # model path and stats path
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the training run directory to evaluate.")
    args = parser.parse_args()

    # Create a unique run name based on the current time and the training run name
    training_run_name = os.path.basename(args.run_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_run_name = f"{timestamp}-eval_on-{training_run_name}"
    eval_dir = os.path.join("artifacts_sb3", "evaluation", eval_run_name)

    os.makedirs(eval_dir, exist_ok=True)
    print(f"--- Starting new evaluation run for: {args.run_dir} ---")
    print(f"Evaluation results will be saved in: {eval_dir}")

    # evaluate the PPO agent
    print("--- 1. Evaluating PPO Agent ---")
    model_path = os.path.join(args.run_dir, "models", "best_model.zip")
    stats_path = os.path.join(args.run_dir, "models", "vec_normalize.pkl")

    env_kwargs = {
        "data_dir": cfg.paths.data_testing,
        "max_n": cfg.ml.rl.ppo.hyperparams.max_n,
        "max_weight": cfg.ml.generation.max_weight,
        "max_value": cfg.ml.generation.max_value,
    }
    env_unwrapped = make_vec_env(KnapsackEnv, n_envs=1, env_kwargs=env_kwargs)
    env = VecNormalize.load(stats_path, env_unwrapped)
    env.training = False
    env.norm_reward = False

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model = PPO.load(model_path, env=env)

    ppo_results = []
    # get all testing instances
    test_instances = env.venv.get_attr('instance_files')[0]

    for instance_path in tqdm(test_instances, desc="Evaluating PPO Agent"):
        # manually set the next instance
        env.venv.env_method('manual_set_next_instance', instance_path)
        obs = env.reset()
        done = [False]
        
        start_time = time.time()
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = env.step(action)
        end_time = time.time()
        
        final_info = info[0]
        ppo_results.append({
            "instance": final_info["instance_file"],
            "n": int(os.path.basename(final_info["instance_file"]).split('_n')[1].split('_')[0]),
            "ppo_value": final_info["total_value"],
            "ppo_time": end_time - start_time,
        })
    ppo_df = pd.DataFrame(ppo_results)

    # --- Evaluate Baseline Solver ---
    print("\n--- 2. Evaluating Baseline Solver ---")    
    BaselineSolverClass = cfg.ml.baseline_algorithm
    baseline_solver_name = None

    for name, solver_class in ALGORITHM_REGISTRY.items():
        if solver_class == BaselineSolverClass:
            baseline_solver_name = name
            break
    
    if BaselineSolverClass and baseline_solver_name:
        print(f"Found baseline solver: {baseline_solver_name}")
        solver_instance = BaselineSolverClass(config={})
        baseline_results = []
        for instance_path in tqdm(test_instances, desc=f"Solving with {baseline_solver_name}"):
            result = solver_instance.solve(instance_path)
            baseline_results.append({
                "instance": instance_path,
                "baseline_value": result.get("value", -1),
                "baseline_time": result.get("time", -1),
            })
        baseline_df = pd.DataFrame(baseline_results)
        
        # --- Combine Results ---
        # clean up instance names for merging
        baseline_df['instance'] = baseline_df['instance'].apply(os.path.basename)
        ppo_df['instance'] = ppo_df['instance'].apply(os.path.basename)
        
        merged_df = pd.merge(ppo_df, baseline_df, on="instance")
        # calculate optimality gap
        merged_df['optimality_gap'] = 1.0 - (merged_df['ppo_value'] / merged_df['baseline_value'])
        
        print("\n--- Combined Evaluation Results ---")
        print(merged_df.head())
        
        # --- Save combined results and plots ---
        final_df = merged_df
        
    else:
        print(f"Baseline solver '{baseline_solver_name}' not found. Skipping error analysis.")
        final_df = ppo_df

    # Save the final results
    results_path = os.path.join(eval_dir, "evaluation_results_full.csv")
    final_df.to_csv(results_path, index=False)
    
    plots_path = os.path.join(eval_dir, "plots")
    plot_results(final_df, save_dir=plots_path)
    
if __name__ == '__main__':
    main()