import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from typing import List

class WeightedGCNConv(MessagePassing):
    """支持边权重的GCN层"""
    def __init__(self, in_channels, out_channels):
        super(WeightedGCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index, edge_weight):
        row, col = edge_index
        deg = torch.zeros(x.size(0), device=x.device)
        deg.index_add_(0, row, edge_weight)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        
        x = self.lin(x)
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class WeightedGCNEncoder(nn.Module):
    """加权GCN编码器"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.1):
        super(WeightedGCNEncoder, self).__init__()
        self.dropout = dropout
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(WeightedGCNConv(dims[i], dims[i+1]))
        
        self.convs = nn.ModuleList(layers)
    
    def forward(self, x, edge_index, edge_weight):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

def create_gcn_model(input_dim: int, hidden_dims: List[int] = [64, 32], output_dim: int = 16):
    """创建GCN模型"""
    return WeightedGCNEncoder(input_dim, hidden_dims, output_dim)