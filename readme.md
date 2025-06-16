# How to Use the System

This project provides a suite of tools to generate, solve, and benchmark knapsack problems.

## 1. Generate Test Cases (Run Once)

First, you need to create a consistent set of test problems. Open your terminal and run the generation script:

```bash
python generate_test_suite.py
```

This command will create the test_cases/ directory and fill it with .csv files, each representing a unique problem instance.

## 2. Run the Benchmark

Once the test cases are generated, you can run the main benchmark script to test the performance of the implemented algorithms.

```bash
python benchmark_runner.py
```

This script will perform the following actions:

1. Load each instance from the test_cases/ directory.
2. Run every algorithm defined in the ALGORITHMS_TO_TEST dictionary on each instance.
3. Print the timing results to the console.
4. Generate and display a comparative plot, saving it as benchmark_comparison.png.

