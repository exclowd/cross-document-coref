import torch
from torch import nn


class SimplePairWiseClassifier(nn.Module):
    def __init__(self,
                 bert_hidden_size: int,
                 hidden_layer_size: int,
                 embedding_dimension: int,
                 dropout: float,
                 use_mention_width: bool,
                 use_head_attention: bool):
        super().__init__()
        self.input_layer = bert_hidden_size * \
            3 if use_head_attention else bert_hidden_size * 2
        if use_mention_width:
            self.input_layer += embedding_dimension
        self.input_layer *= 3
        self.hidden_layer = hidden_layer_size
        self.pairwise_mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.input_layer, self.hidden_layer),
            nn.ReLU(),
            nn.Linear(self.hidden_layer, self.hidden_layer),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_layer, 1),
        )

    def forward(self, first, second):
        return self.pairwise_mlp(torch.cat((first, second, first * second), dim=1))
