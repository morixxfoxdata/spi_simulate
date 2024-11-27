import torch
import torch.nn as nn


class BasedDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(BasedDecoder, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
