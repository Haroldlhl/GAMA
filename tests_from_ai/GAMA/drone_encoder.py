import torch
import torch.nn as nn
import torch.nn.functional as F

class DroneDecisionHead(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        """
        无人机决策头：将当前节点编码转换为目标节点注意力权重
        
        Args:
            hidden_dim: 节点编码的隐藏维度
            num_heads: 注意力头数
            dropout: Dropout率
        """
        super(DroneDecisionHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Query生成网络：学习从当前位置到目标选择的映射
        self.query_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, drone_node_encoding):
        """
        前向传播：生成目标节点注意力权重
        
        Args:
            drone_node_encoding: 无人机节点编码
            
        Returns:
            attention_weights: 目标节点注意力权重
        """
        query = self.query_net(drone_node_encoding)
        return query