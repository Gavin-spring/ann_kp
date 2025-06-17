# ANN Approxiamation for Dynamic Programming in Knapsack Problem

This project provides a suite of tools to generate, solve, and benchmark various instances of the 0/1 knapsack problem.

## Project Structure

After running the scripts, the project will automatically generate the following directory structure:

```text
.
├── test_cases/            # Contains the generated test cases (.csv files)
├── results/               # Contains the raw benchmark data (.csv files)
├── plots/                 # Contains the performance comparison plots (.png files)
├── logs/                  # Contains detailed execution logs (.log files)
├── algorithms.py          # Core algorithm implementations
├── benchmark_runner.py    # Benchmark runner
├── generate_test_suite.py # Test suite generator
├── config.py              # Global configuration file
└── ...                    # Other utility scripts
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