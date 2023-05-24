import torch
from torch import nn


class SpanScorer(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 dropout: float):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        nn.init.xavier_uniform_(self.fc[1].weight)
        nn.init.uniform_(self.fc[1].bias)
        nn.init.xavier_uniform_(self.fc[3].weight)
        nn.init.uniform_(self.fc[3].bias)

    def forward(self, x):
        return self.fc(x)
