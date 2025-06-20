# File: ann/model.py
import torch.nn as nn

class KnapsackPredictor(nn.Module):
    def __init__(self, input_size):
        super(KnapsackPredictor, self).__init__()

        self.layer1 = nn.Linear(input_size, 512)
        # Add BatchNorm layer after the linear layer
        self.bn1 = nn.BatchNorm1d(512) 

        self.layer2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.layer3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.layer4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)

        self.layer5 = nn.Linear(64, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # The new forward pass: Linear -> BatchNorm -> ReLU
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.relu(self.bn3(self.layer3(x)))
        x = self.relu(self.bn4(self.layer4(x)))

        x = self.layer5(x) # No activation/bn on the final output layer
        return x