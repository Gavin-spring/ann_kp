# ANN Approxiamation for Dynamic Programming in Knapsack Problem

This project provides a suite of tools to generate data, train & evaluate model against other solvers on 0-1 Knapsack Problem.

## Project Requirements
torch
pandas
numpy
matplotlib
seaborn
gurobipy
tqdm
PyYAML

## Project Structure

```text
.
Project Structure:
├── src/
│  ├── utils/
│   │  ├── run_utils.py
│   │  ├── __init__.py
│   │  ├── logger.py
│   │  ├── config_loader.py
│   │  ├── generator.py
│  ├── evaluation/
│   │  ├── reporting.py
│   │  ├── plotting.py
│  ├── solvers/
│   │  ├── ml/
│   │   │  ├── feature_extractor.py
│   │   │  ├── rl_solver.py
│   │   │  ├── dnn_solver.py
│   │   │  ├── data_loader.py
│   │   │  ├── dnn_model.py
│   │  ├── classic/
│   │   │  ├── heuristic_solvers.py
│   │   │  ├── gurobi_solver.py
│   │   │  ├── dp_solver.py
│   │  ├── __init__.py
│   │  ├── interface.py
│  ├── __init__.py
├── Scripts/
│  ├── __init__.py
│  ├── preprocess_data.py
│  ├── recreate_plot_from_log.py
│  ├── train_model.py
│  ├── generate_data.py
│  ├── evaluate_solvers.py
│  ├── git-multi-status.sh
├── test/
│  ├── test_algorithms.py
│  ├── benchmark_runner.py
├── data/
│  ├── validation/
│   │   └── (*.csv)
│  ├── training/
│   │   └── (*.csv)
│  ├── testing/
│   │   └── (*.csv)
├── configs/
├── artifacts/
│  ├── runs/
│   │  ├── training/
│   │   │  ├── 20250622_214750_n1000_lr0.001/
│   │  ├── evaluation/
│   │   │  ├── 20250623_004328_n1000_lr0.001/
├── pyproject.toml
├── requirements.txt
├── readme.md
```

## How to Use

Please follow the steps in order to ensure the system works correctly.

### Step 1: Generate Data

This is the first step you need to perform. This script creates 0-1 knapsack problem instances based on the settings in `config.yaml`.

**Run the following command in your terminal:**

```bash
python Scripts/generate_data.py
```

### Step 2. Proprocess Data

```bash
python Scripts/preprocess_data.py
```

### Step 3: Train the model
```bash
python Scripts/train_model.py
```

### Step 4: Evaluate Model Performance
```bash
python Scripts/evaluate_model.py
```