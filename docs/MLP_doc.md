# Neural-Guided Dynamic Programming for 0-1 Knapsack Optimization

### A Pooled Learning Approach with a Custom VDN Network

## Introduction & Problem Definition

### 1. Model Overview
* **Model:** MLP
* **Architecture:** `(512, 256, 128, 64, 1)` layers, followed by `Batch Normalization` and `ReLU`.
* **Data Correlation Type:** `uncorrelated`

### 2. Problem Definition: The 0/1 Knapsack Problem
* **Item Parameters:** `max_weight: 100`, `max_value: 100`
* **Knapsack Capacity Ratio:** Varies in range `[0.1, 0.9]`

### 3. Dataset Generation
* **Training Set:** 500 instances per size `n`, for `n` in range `[5, 1000]`, step 5.
* **Validation Set:** 100 instances per size `n`, for `n` in range `[5, 1000]`, step 5.
* **Testing Set:** 150 instances per size `n`, for `n` in range `[1101, 2001]`, step 10 (to test extrapolation).

## Model Configuration & Hyperparameters

### 1. Key Hyperparameters
* **`max_n_for_architecture`**: 2001
* **`max_weight_norm`**: 100
* **`max_value_norm`**: 100
* **`input_size_factor`**: 5
* **`input_size_plus`**: 1
* **`target_scale_factor_multiplier`**: 1.0

### 2. Training Parameters
* **`total_epochs`**: 800
* **`epochs_per_n`**: 10 *(Note: For curriculum learning, not used in Pooled Data Training)*
* **`batch_size`**: 32
* **`learning_rate`**: 0.001
* **`weight_decay`**: 0.001

## Methodology & Preprocessing

### 1. Execution Workflow
1. **Data Generation:** `generate_data.py`
2. **Preprocessing:** `preprocess_data.py`
3. **Training & Evaluation:** `train_model.py` -> `evaluate_solvers.py`

### 2. Training & Validation Process
The model is trained on the training set. Validation data is used to monitor performance and prevent overfitting.

### 3. Preprocessing & Feature Engineering
* **Tensor Conversion:** Raw data is converted into tensors for the model.
* **Engineered Features:** The input tensor is a concatenation of:
  * `weights_norm`: Normalized item weights.
  * `values_norm`: Normalized item values.
  * `value_densities`: Item value-to-weight ratios.
  * `weight_to_capacity_ratios`: Item weight-to-capacity ratios.
  * `capacity_ratio_feature`: A per-item feature derived from capacity.
  * `normalized_capacity`: A single scalar for the current knapsack capacity.
* **Output Label:**
  * `normalized_label`: The ground-truth value, scaled by `target_scale_factor`.


## Training & Evaluation

### 1. Training Model: Pooled Learning
* At each epoch, the model is trained on a randomly selected batch from the entire pooled training dataset.
* After each validation step, the training and testing loss are observed to monitor for overfitting and assess convergence.

### 2. Testing & Benchmarking
* **Ground Truth Baseline:** **Gurobi** is used as the ground truth for calculating ML model errors.
* **Compared Algorithms:** The DNN model is benchmarked against a suite of classic solvers:
  * "Gurobi"
  * "2D DP"
  * "1D DP (Optimized)"
  * "Branch and Bound"
  * "Greedy"
* **Evaluation Metrics:**
  * Mean Absolute Error (MAE)
  * Mean Relative Error (MRE)
  * Mean Squared Error (MSE)

## Results & Analysis

### Performance (Speed)
* **Excellent Speed Advantage:** The DNN solver exhibits excellent performance, maintaining a nearly constant-time inference speed (Average Time < 1ms) even as problem size (`n`) increases to 2000.
* **Avoiding the Curse of Dimensionality:** In contrast, exact solvers like **Branch and Bound** show exponential time complexity and become impractical for `n > 2000`. The DNN approach effectively bypasses this limitation.
* **Comparison:** While the Greedy algorithm is fastest, the DNN is significantly faster than Gurobi, 1D DP, and Branch and Bound for `n > 100`.

### Accuracy (Error)
* **Generalization Challenge:** The model's main weakness is its generalization ability. It was trained on problems up to `n=1000`.
* **High Error Beyond Training Range:** The error plot (MAE/RMSE) shows that as `n` increases toward 1000, the error grows significantly. Beyond `n=1000`, the error remains high and volatile, indicating the model fails to generalize to unseen problem sizes.
* **Root Cause:** This is attributed to a **truncated feature input**, which limits the model's capacity to process problems larger than its training scope.


## Conclusion & Recent Progress

### Conclusion
* The Deep Neural Network (DNN) approach is highly promising for solving the 0/1 knapsack problem, offering a **drastic speed improvement** over traditional exact solvers and avoiding the curse of dimensionality.
* However, the current model's **accuracy and generalization are poor**, especially for problem sizes beyond its training range.

### Recent Progress (Last Two Weeks)
* **Studied Reinforcement Learning (RL) Algorithms:** Focused on foundational methods for combinatorial optimization, including:
  * Temporal-Difference (TD) learning, especially **Q-Learning** and **DQN**.
  * Value Function Approximation techniques.
  * **Policy Gradient (PG)** methods.
* **Practical Implementation:** Began implementing the knapsack problem within the framework of the paper "Neural Combinatorial Optimization with Reinforcement Learning".
  * This involves working with the official code repository and adapting its **Policy Gradient** approach to our problem.
