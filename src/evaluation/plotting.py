# src/evaluation/plotting.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def plot_evaluation_errors(results_df: pd.DataFrame, save_path: str, solver_names: list):
    """
    Plots MAE, MRE, and RMSE for a list of solvers against problem size 'n'.
    
    Args:
        results_df (pd.DataFrame): The aggregated dataframe containing error columns.
        save_path (str): The path to save the plot image.
        solver_names (list): A list of solver names to plot errors for.
    """
    logger.info("Generating evaluation error comparison plot...")
    
    # create a figure with 3 subplots, one for each error metric
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle('Solver Error Metrics vs. Problem Size (n)', fontsize=16, y=0.99)
    
    # define the metrics to plot with their titles
    metrics_to_plot = {
        'mae': "Mean Absolute Error (MAE)",
        'mre': "Mean Relative Error (MRE %)",
        'rmse': "Root Mean Square Error (RMSE)"
    }
    
    # traverse each solver name
    for solver_name in solver_names:
        
        # traverse each metric and its corresponding subplot
        for i, (metric, title) in enumerate(metrics_to_plot.items()):
            # dynamically create the column name based on solver name and metric
            column_name = f"{solver_name}_{metric}"
            
            # check if the column exists in the results dataframe
            if column_name in results_df.columns:
                sns.lineplot(
                    ax=axes[i], 
                    data=results_df, 
                    x='n', 
                    y=column_name, 
                    label=solver_name # use the solver name for the legend
                )
                axes[i].set_title(title)
                axes[i].set_ylabel("Error Value")
                axes[i].legend()

    axes[-1].set_xlabel("Number of Items (n)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    try:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Error comparison plot saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save error plot: {e}")
    finally:
        plt.close()

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

def plot_evaluation_errors_DNN(results_df: pd.DataFrame, save_path: str):
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