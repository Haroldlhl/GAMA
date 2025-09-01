#   NetworkModels (网络模型类) 
#   职责：定义PyTorch/TensorFlow神经网络模型。 
#   包含的类/函数： ​ - ​  GNNEncoder: 实现GAT或GCN，用于编码节点。 ​ 
#   - ​  ActorNetwork: 实现策略网络（MLP + Attention）。
#   - ​  CriticNetwork: 实现价值网络（深度MLP）。

from dataclasses import dataclass
from torch_geometric.nn import GCNConv, GATConv
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class GNNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super(GNNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.convs.append(GCNConv(hidden_dims[i-1], hidden_dims[i]))
        self.convs.append(GCNConv(hidden_dims[-1], output_dim))

    def forward(self, x, edge_index, edge_weight):
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
        return x

class ActorNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CriticNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x