# File: model.py
# -*- coding: utf-8 -*-

import torch.nn as nn

class KnapsackPredictor(nn.Module):
    """Defines the DNN architecture for the knapsack problem."""
    def __init__(self, input_size):
        super(KnapsackPredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x