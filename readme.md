# ANN Approxiamation for Dynamic Programming in Knapsack Problem

This project provides a suite of tools to generate, solve, and benchmark various instances of the 0/1 knapsack problem.

## Project Structure

After running the scripts, the project will automatically generate the following directory structure:

```text
.
Project Structure:
├── results/
│   └── (*.csv)
├── plots/
├── test/
│  ├── test_algorithms.py
├── test_cases/
│   └── (*.csv)
├── ann/
│  ├── model_train_cases/
│   │   └── (*.csv)
│  ├── saved_models/
│  ├── model_logs/
│  ├── model_plots/
│  ├── model_test_cases/
│   │   └── (*.csv)
│  ├── generate_model_testcases.py
│  ├── model.py
│  ├── train.py
│  ├── evaluation.py
│  ├── ann_solver.py
│  ├── data_loader.py
│  ├── dnn_config.py
├── generate_test_suite.py
├── config.py
├── logger_config.py
├── generator.py
├── algorithms.py
├── benchmark_runner.py
├── git-multi-status.sh
├── z-basic.ipynb
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── readme.md
```


```text
.
├── artifacts/              # ◀━ 1. 新建：所有运行时产出物（代替 plots/, results/, ann/saved_models 等）
│   ├── models/             #    - 保存训练好的模型 (.pth)
│   ├── plots/              #    - 保存所有的图表 (.png)
│   ├── logs/               #    - 保存所有的日志 (.log)
│   └── results/            #    - 保存实验结果 (.csv)
│
├── configs/                # ◀━ 2. 新建：统一管理所有配置文件
│   ├── dnn_solver.yaml     #    - DNN模型的配置
│   ├── rnn_solver.yaml     #    - 未来RNN模型的配置
│   └── classic_solvers.yaml#    - 传统算法的配置
│
├── data/                   # ◀━ 3. 新建：所有原始数据和测试用例
│   ├── training/           #    - 训练用例
│   ├── testing/            #    - 测试用例
│   └── validation/         #    - 验证用例
│
├── src/                    # ◀━ 4. 新建：所有核心 Python 源代码
│   ├── __init__.py
│   ├── solvers/            #    - 核心：所有求解器/模型算法
│   │   ├── __init__.py
│   │   ├── interface.py    #    - 定义所有 Solver 的抽象基类/接口
│   │   ├── classic/        #    - 传统算法
│   │   │   └── __init__.py #      (内容来自 algorithms.py)
│   │   └── ml/             #    - 机器学习模型
│   │       ├── __init__.py
│   │       ├── dnn_model.py#      (内容来自 ann/model.py)
│   │       └── data_loader.py # (来自 ann/data_loader.py, 但会变得更通用)
│   │
│   ├── utils/              #    - 所有可复用的工具/辅助函数
│   │   ├── __init__.py
│   │   ├── logger.py       #    - (合并 logger_config.py 和 ann/model_logger.py)
│   │   └── generator.py    #    - (合并 generator.py 和 ann/gen__instances.py)
│   │
│   └── evaluation/         #    - 评估和基准测试相关代码
│       ├── __init__.py
│       └── metrics.py      #    - 定义评估指标
│
├── tests/                  # ◀━ 5. 结构调整：测试代码应镜像 src 结构
│   ├── __init__.py
│   └── solvers/
│       ├── test_classic.py
│       └── ml/
│           └── test_dnn_model.py
│
├── train.py                # ◀━ 6. 核心脚本：训练ML模型
├── evaluate.py             # ◀━ 核心脚本：评估所有Solver的性能
├── generate_data.py        # ◀━ 核心脚本：调用 src/utils/generator.py 生成数据
│
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── readme.md
```

## How to Use

Please follow the steps in order to ensure the system works correctly.

### Step 1: Generate Test Cases

This is the first step you need to perform. This script creates a suite of knapsack problem instances based on the settings in `config.py`.

**Run the following command in your terminal:**

```bash
python generate_test_suite.py
```

This command will perform the following actions:

1. It checks for and creates the test_cases/ directory.
2. It generates a series of .csv files in that directory, with each file representing a unique problem instance.
3. Simultaneously, the logs/ directory will be created, and the generation process will be logged to benchmark.log.

### Step 2. Run the Benchmark

Once the test cases are generated, you can run the main benchmark script to test the performance of the implemented algorithms.

```bash
python benchmark_runner.py
```

This script will perform the following actions:

1. Load all problem instances from the test_cases/ directory.
2. Run every algorithm defined in the ALGORITHMS_TO_TEST dictionary in config.py on each instance.
3. Output key progress information to the console.
4. Save the detailed performance data (like execution time for each algorithm) to results/benchmark_results.csv.
5. Generate a performance comparison plot named benchmark_comparison.png in the plots/ directory, visually showing how each algorithm performs at different problem scales.
6. Append the full execution log, including timestamps, module info, and debug messages, to the logs/benchmark.log file.

### Step 3: View the Results

After all tests are complete, you can find the results in the following locations:

1. Performance Plot: Open plots/benchmark_comparison.png to see the visual performance comparison.
2. Raw Data: Open results/benchmark_results.csv with a spreadsheet program to see the exact time taken for each test.
3. Detailed Logs: Check the logs/benchmark.log file to get the most detailed program execution trace, useful for debugging or in-depth analysis.