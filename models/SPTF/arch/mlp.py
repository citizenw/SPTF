import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        hidden = self.fc(input_data)
        hidden = hidden + input_data
        return hidden


class GraphMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        x = self.fc1(input_data)
        x = self.act(x)
        x = self.dropout(x)
        return x + self.fc2(x)
