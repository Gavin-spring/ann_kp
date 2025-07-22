# Pointer Network Technical Documentation

This document provides a detailed technical implementation of the Pointer Network model used for solving the Knapsack Problem, covering algorithm principles, model architecture, data flow, training configuration, and optimization strategies.

## 1. Algorithm Principle

The model's training and testing phases follow the classic exploration-exploitation paradigm in reinforcement learning.

### a. Training: Reinforce + Baseline

- **Core Algorithm**: Training employs the **REINFORCE with Baseline** algorithm, a fundamental yet powerful **Policy Gradient** method.
- **Objective**: The algorithm learns not from a "correct answer" but through a process of trial and error. Its goal is to optimize the policy network (the Pointer Network) to increase the probability of action sequences (item selection order) that yield higher rewards (total value in the knapsack).
- **Baseline**: An **exponential moving average** of rewards is introduced as a baseline. Its purpose is to measure the "surprise level" of the current reward (i.e., the `Advantage`) rather than its absolute value. This effectively **reduces gradient variance**, leading to a more stable and faster training process.
- **Exploration Mechanism**: A **`stochastic` search** is used for decision-making in each training step. The model samples the next item to select based on the output probability distribution. This is key to ensuring the model adequately **explores** the decision space to avoid getting stuck in local optima.

### b. Testing: Greedy & Beam Search

#### Greedy Search

- **Mechanism**: During the testing (or inference) phase, a **`greedy` search** is employed. At each decision step, the model deterministically selects the item with the highest current probability.
- **Purpose**: This represents the **exploitation** of the model's learned knowledge, aiming to achieve stable and optimal performance.

#### Beam Search - (A Better Alternative)

- **Mechanism**: As an upgrade to greedy search, Beam Search maintains the `k` most probable candidate sequences at each step. This reduces the risk of missing a globally optimal solution due to a "short-sighted" choice at an early step.
- **Trade-off**: It generally achieves better results (higher total value) than greedy search, but at the cost of computational overhead that is approximately `k` times that of greedy search. This is a classic **Effectiveness vs. Efficiency** trade-off, suitable for offline evaluation scenarios where solution quality is paramount.

## 2. Mathematical Formulas

### a. Training

The training objective is to minimize the following loss function $ L(\theta) $:

$$
L(\theta) = - \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \log \pi_\theta(a_t|s_t) A(s_t, a_t) \right]
$$

Where:

- $ \pi_\theta(a_t|s_t) $: is the probability of the policy network (Pointer Network) selecting action $ a_t $ in state $ s_t $.
- $ A(s_t, a_t) $: is the Advantage function, calculated as:
  $$
  A(s_t, a_t) = R(\tau) - b(s_t)
  $$
  - $ R(\tau) $: is the final reward (total value in the knapsack) for the entire trajectory $ \tau $.
  - $ b(s_t) $: is the baseline (exponential moving average of rewards).

### b. Testing

- **Greedy Search**: At each step $ t $, select the action $ a_t $ that maximizes the probability.
  $$
  a_t = \arg\max_{a} \pi_\theta(a|s_t)
  $$

- **Beam Search**: At each step $ t $, maintain the $ k $ candidate sequences with the highest overall probability.

## 3. Model Architecture

The model is based on an Encoder-Decoder architecture.

### a. Pointer Network (1 Encoder + 1 Decoder)

- The entire model is a Pointer Network, where the decoder outputs by "pointing" to a position in the input sequence rather than generating from a fixed vocabulary. This makes it naturally suited for solving sorting and combinatorial optimization problems.

### b. Encoder (LSTM)

- **Structure**: Uses a Long Short-Term Memory (LSTM) network.
- **Function**: It is responsible for "reading" and understanding the entire input sequence (i.e., the feature sequence of all items). It encodes the features (weight, value) and contextual relationships of each item into a set of context vectors, `context`, for the decoder to use.

### c. Decoder

- **Structure**: Uses a single-step LSTMCell for iterative decoding.
- **Function**: It is responsible for constructing the solution (the item selection order) step-by-step, based on the context provided by the Encoder and its own current state.
- **Capacity Embedding (Key Design)**: During decision-making, the decoder embeds the current remaining knapsack capacity and concatenates it with the LSTM's hidden state to form an *Augmented Query*. This design is crucial as it allows the decoder to be aware of constraints at each step when pointing to the next item, leading to more rational decisions.
- **Decoding Strategy**:
  - **i. Stochastic** — training: Performs random sampling on the output probability distribution via `torch.multinomial` to enable exploration.
  - **ii. Greedy** — testing: Selects the action with the highest probability via `torch.argmax` to enable exploitation.

## 4. Data Flow Process

The complete data flow from raw files to the final output is as follows:

1. **Input Source**: `.csv` files, each representing an independent knapsack problem instance with a variable number of items.
2. **Data Loading (`RawKnapsackDataset`)**: Reads a single `.csv` file and converts it into a dictionary of Tensors, including weights, values, and capacity. At this stage, the number of items (`N`) varies for each instance.
3. **Batch Collation (`knapsack_collate_fn`)**:
   - Pads multiple instances within a batch to a uniform length, `N_max` (the maximum length in the current batch).
   - Simultaneously generates an Attention Mask to inform the model which elements are real data versus padding.
   - Output: Well-formed batch tensors. For example, the shape of weights becomes `[B, N_max]` (where `B` is the batch size).
4. **Model Input Preparation**:
   - The weights and values tensors are stacked and permuted to form the final input tensor for the model.
   - Shape: `[B, N_max, 2]`
5. **Pointer Network Internal Flow**:
   - **Embedding Layer**: `[B, N_max, 2]` → `[B, N_max, D_emb]`
   - **Encoder (LSTM)**: Takes the permuted input `[N_max, B, D_emb]` and outputs the context `context` with a shape of `[N_max, B, D_hidden]`.
   - **Decoder (LSTMCell + Attention)**: Iterates `N_max` steps, using the `context` and capacity information at each step to point to an input position.
6. **Output**: The model outputs a sequence of action indices with a shape of `[N_max, B]`, representing the recommended item selection order for each problem instance in the batch.

## 5. Training Configuration

The following configuration from the `config.yaml` file defines key hyperparameters for the model and training loop.

```yaml
rl:
  hyperparams:
    embedding_dim: 96
    hidden_dim: 96
    n_glimpses: 1
    tanh_exploration: 10
    use_tanh: true
    max_n: 200

  training:
    total_epochs: 50
    batch_size: 128
    learning_rate: 0.001
    lr_decay_step: 500
    lr_decay_rate: 0.96
    max_grad_norm: 2.0
    baseline_beta: 0.8

  testing:
    batch_size: 32
```

## 6. Dataset Generation Configuration

The training, validation, and test datasets are generated according to rules specified in `config.yaml`.

### a. Training Set

- **Problem size `n`**: Starts from 5 items, ends at 200 items, with a step of 5.
- **Dataset size**: each `n` has 50 instances, 2000 instances in total.

### a. Validation Set

- **Problem size `n`**: Starts from 5 items, ends at 200 items, with a step of 5.
- **Dataset size**: each `n` has 10 instances, 400 instances in total.

### b. Testing Set

- **Problem size `n`**: Starts from 5 items, ends at 500 items, with a step of 10.
- **Dataset size**: each `n` has 20 instances, 1000 instances in total.

## 7. Optimizer and Training Strategy

A suite of modern optimization strategies is employed to ensure efficient and stable training.

### a. Adam Optimizer

- Uses `torch.optim.Adam` as the core parameter update algorithm, which benefits from adaptive learning rates and fast convergence.

### b. Learning Rate Scheduler

- Uses `torch.optim.lr_scheduler.MultiStepLR` to dynamically adjust the learning rate. Every `lr_decay_step` (500) batches, the learning rate is multiplied by the decay factor `lr_decay_rate` (0.96), which helps with finer convergence in later training stages.

### c. Gradient Clipping

- Employs the `torch.nn.utils.clip_grad_norm_` technique to cap the norm of gradients at `max_grad_norm` (2.0). This effectively prevents training instability caused by exploding gradients, which is particularly important for RNN/LSTM-based models.

### d. Automatic Mixed Precision (AMP) & GradScaler

- Training is accelerated using Automatic Mixed Precision, enabled via `torch.amp.GradScaler` and the `autocast` context.
- **Purpose**: This leverages half-precision floating-point numbers (`float16`) for computations on compatible GPUs (e.g., NVIDIA Tensor Core GPUs), resulting in significantly faster training speed and lower memory consumption.
- **GradScaler**: It dynamically scales the loss to prevent small gradients in `float16` from underflowing to zero, thus ensuring numerical stability during training.

### e. Execution Order in Training Loop

The optimization tools are called in a logically sound order, following PyTorch's best practices:

```python
zero_grad
-> autocast forward
-> loss
-> scaler.scale(loss).backward
-> clip_grad_norm
-> scaler.step
-> scaler.update
-> scheduler.step
```