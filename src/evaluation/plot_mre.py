import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import argparse
import os

parser = argparse.ArgumentParser(description="Plot MRE vs Problem Size: Target Solver vs Gurobi.")
parser.add_argument('--csv-path', type=str, required=True)
parser.add_argument('--solver', type=str, required=True)
args = parser.parse_args()

csv_path = args.csv_path
target_solver = args.solver

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV not found: {csv_path}")

output_dir = os.path.dirname(csv_path)
output_filename = f'mre_vs_problem_size_{target_solver.lower().replace(" ", "_")}_vs_gurobi.png'
output_path = os.path.join(output_dir, output_filename)

# Load data
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    raise RuntimeError(f"Failed to read CSV: {e}")

# Validate columns
required_cols = ['solver', 'n', 'avg_value', 'PPO_mre']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# Extract Gurobi and target solver data
gurobi_data = df[df['solver'] == 'Gurobi'].copy()
target_data = df[df['solver'] == target_solver].copy()

if gurobi_data.empty:
    raise RuntimeError("Gurobi data not found — it's required as baseline.")

# Compute MRE for target solver
# Assuming PPO_mre = V_PPO / V_opt
target_data['mre_pct'] = (1 - target_data['PPO_mre']) * 100  # Relative error in percent

# Add Gurobi as optimal (MRE = 0)
gurobi_data['mre_pct'] = 0.0  # Optimal solution has 0 error

# Combine data for plotting
plot_data = pd.concat([gurobi_data[['n', 'mre_pct']], target_data[['n', 'mre_pct']]], ignore_index=True)
plot_data['Solver'] = plot_data['n'].apply(lambda x: 'Gurobi (Optimal)' if x in gurobi_data['n'].values else target_solver)

# Plot
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(10, 6))
ax = plt.gca()

sns.lineplot(data=plot_data, x='n', y='mre_pct', hue='Solver', marker='o', ax=ax, errorbar=None)

ax.set_yscale('log')
# ax.axhline(y=30, color='r', linestyle='--', linewidth=2)
ax.yaxis.set_major_formatter(PercentFormatter())

ax.set_title(f'MRE vs Problem Size: {target_solver} vs Gurobi (Baseline)', fontsize=16, pad=20)
ax.set_xlabel('Number of Items (n)', fontsize=12)
ax.set_ylabel('Mean Relative Error (MRE) - Log Scale', fontsize=12)
ax.legend(title='Solver')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# # Add '30%' label
# xlim_left = plot_data['n'].min() - (plot_data['n'].max() - plot_data['n'].min()) * 0.1
# ax.text(xlim_left, 30, '30%', color='red', fontweight='bold', ha='right', va='center')

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Plot saved: '{output_path}'")