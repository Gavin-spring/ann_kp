# src/evaluation/plotting.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def plot_evaluation_errors(results_df: pd.DataFrame, save_path: str):
    """Plots MAE, MRE, and RMSE for the ML solver against problem size."""
    logger.info("Generating evaluation error plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    
    # Filter for the ML solver results and required columns
    ml_results = results_df[results_df['solver'] == 'DNN'][['n', 'mae', 'mre', 'rmse']]
    if ml_results.empty:
        logger.warning("No DNN results found to plot errors.")
        return
        
    # Melt the dataframe to make it tidy for seaborn
    df_melted = ml_results.melt(id_vars='n', var_name='Error Type', value_name='Error Value')
    
    sns.lineplot(data=df_melted, x='n', y='Error Value', hue='Error Type', style='Error Type', markers=True, dashes=False)
    
    plt.title('DNN Solver Error vs. Problem Size (n)', fontsize=16)
    plt.xlabel('Number of Items (n)', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.legend(title='Error Type')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Error plot saved to {save_path}")


def plot_evaluation_times(results_df: pd.DataFrame, save_path: str):
    """Plots a comparison of solve times for all solvers."""
    logger.info("Generating evaluation time comparison plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))

    sns.lineplot(data=results_df, x='n', y='avg_time_ms', hue='solver', style='solver', markers=True, dashes=False)
    
    plt.title('Solver Performance: Time vs. Problem Size (n)', fontsize=16)
    plt.xlabel('Number of Items (n)', fontsize=12)
    plt.ylabel('Average Time per Instance (ms)', fontsize=12)
    plt.yscale('log') # Use a log scale for time, as it can vary greatly
    plt.legend(title='Solver')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Time comparison plot saved to {save_path}")