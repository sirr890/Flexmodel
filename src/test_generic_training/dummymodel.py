import torch
import torch.nn as nn

# Define a simple linear model


class Random(nn.Module):
    def __init__(self):
        super(Random, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
