# ann_solver.py
# -*- coding: utf-8 -*-

# TODO: Future plan, after the validation of DNN model
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np

# (KnapsackANN class definition can remain the same as you have it)
class KnapsackANN(nn.Module):
    def __init__(self, input_size):
        super(KnapsackANN, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

class KnapsackEnvironment:
    """Encapsulates the rules of the knapsack problem."""
    def __init__(self, items, capacity):
        self.items = items
        self.capacity = capacity
        self.num_items = len(items)

    def get_initial_state(self):
        # State: (current_weight, frozenset_of_available_item_indices)
        return (0, frozenset(range(self.num_items)))

    def get_possible_actions(self, state):
        current_weight, available_items = state
        return [i for i in available_items if self.items[i]['weight'] + current_weight <= self.capacity]

    def step(self, state, action):
        # Executes a decision (action) and returns the new state
        current_weight, available_items = state
        item_to_add = self.items[action]
        
        new_weight = current_weight + item_to_add['weight']
        new_available_items = available_items - {action}
        new_value_gained = item_to_add['value']
        
        return (new_weight, new_available_items), new_value_gained

class ANNSolver:
    def __init__(self, capacity, learning_rate=0.001):
        self.capacity = capacity
        # We now have a dictionary of models, one for each problem size 'n'
        self.models = {}
        self.optimizers = {}
        self.data_pool = deque(maxlen=20000)
        self.loss_fn = nn.MSELoss()
        self.learning_rate = learning_rate
        # Check if a CUDA-enabled GPU is available, otherwise fall back to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        


    def _get_or_create_model(self, num_items):
        if num_items not in self.models:
            print(f"Initializing a new model for n={num_items}")
            self.models[num_items] = KnapsackANN(input_size=num_items)
            self.optimizers[num_items] = optim.Adam(self.models[num_items].parameters(), lr=self.learning_rate)
        return self.models[num_items], self.optimizers[num_items]

    # ... ( _state_to_tensor, _generate_episode, _learn_from_batch would be refactored
    #      to use the KnapsackEnvironment and to work with models from the self.models dict)

    def train_with_curriculum(self, all_items, max_n, episodes_per_n=200, batch_size=64):
        """
        Main training loop that implements curriculum learning, inspired by Xu et al.
        """
        for n in range(3, max_n + 1): # Start from small problems (e.g., 3 items)
            print(f"\n--- Training for problem size n={n} ---")
            
            # Get the model for the current size
            current_model, current_optimizer = self._get_or_create_model(n)
            
            # Create an environment for this problem size
            current_items = {i: all_items[i] for i in range(n)}
            env = KnapsackEnvironment(current_items, self.capacity)

            epsilon = 1.0
            epsilon_decay = 0.99
            epsilon_min = 0.05
            
            for episode in range(episodes_per_n):
                # The episode generation would now use the environment 'env'
                # It would also use a cascade of previously trained models to make better decisions
                # (This part is complex, similar to FT_train_N_TSP.py)
                # For simplicity here, we just show the structure.
                
                # A simplified training step:
                # 1. Generate an episode using the current model `current_model` and `env`.
                #    (This logic would be in a refactored _generate_episode)
                #
                # 2. Add the experience to the data_pool.
                #
                # 3. Sample a batch from data_pool and train `current_model`.
                #    (This logic would be in a refactored _learn_from_batch)

                epsilon = max(epsilon_min, epsilon * epsilon_decay)
                if episode % 50 == 0:
                    print(f"  n={n}, Episode {episode}, Epsilon={epsilon:.3f}")

        print("\n--- Curriculum training complete! ---")