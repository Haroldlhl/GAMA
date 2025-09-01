import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data
import math

class EnhancedGlobalAwareGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, dropout=0.1, 
                 pos_enc_dim=32, max_distance=100.0):
        """
        增强的全局感知GNN，包含改进的位置编码
        
        Args:
            input_dim: 输入节点特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_heads: 注意力头数
            dropout: dropout率
            pos_enc_dim: 位置编码维度
            max_distance: 最大距离（用于归一化）
        """
        super(EnhancedGlobalAwareGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.pos_enc_dim = pos_enc_dim
        self.max_distance = max_distance
        
        # 位置编码网络
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, pos_enc_dim // 2),
            nn.ReLU(),
            nn.Linear(pos_enc_dim // 2, pos_enc_dim),
            nn.ReLU()
        )
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim + pos_enc_dim, hidden_dim)
        
        # 4层GATv2卷积，使用距离感知的注意力机制
        self.conv1 = DistanceAwareGATv2(hidden_dim, hidden_dim, heads=num_heads, 
                                       dropout=dropout, pos_enc_dim=pos_enc_dim)
        self.conv2 = DistanceAwareGATv2(hidden_dim * num_heads, hidden_dim, heads=num_heads,
                                       dropout=dropout, pos_enc_dim=pos_enc_dim)
        self.conv3 = DistanceAwareGATv2(hidden_dim * num_heads, hidden_dim, heads=num_heads,
                                       dropout=dropout, pos_enc_dim=pos_enc_dim)
        self.conv4 = DistanceAwareGATv2(hidden_dim * num_heads, hidden_dim, heads=num_heads,
                                       dropout=dropout, pos_enc_dim=pos_enc_dim)
        
        # 跳跃连接
        self.skip_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim * num_heads),
            nn.Linear(hidden_dim * num_heads, hidden_dim * num_heads),
            nn.Linear(hidden_dim * num_heads, hidden_dim * num_heads)
        ])
        
        # 输出层
        self.output_proj = nn.Linear(hidden_dim * num_heads, output_dim)
        
        # 全局节点参数
        self.global_node_feature = nn.Parameter(torch.randn(1, hidden_dim))
        
    def compute_structural_positional_encoding(self, distance_matrix):
        """
        计算结构位置编码：使用到所有节点的距离分布
        
        Args:
            distance_matrix: [num_nodes, num_nodes] 距离矩阵
            
        Returns:
            pos_encoding: [num_nodes, pos_enc_dim] 位置编码
        """
        num_nodes = distance_matrix.size(0)
        
        # 1. 使用距离分布直方图
        bins = torch.linspace(0, self.max_distance, self.pos_enc_dim // 2, device=distance_matrix.device)
        hist_encoding = torch.zeros(num_nodes, self.pos_enc_dim // 2, device=distance_matrix.device)
        
        for i in range(num_nodes):
            distances = distance_matrix[i]
            hist = torch.histc(distances, bins=self.pos_enc_dim // 2, min=0, max=self.max_distance)
            hist_encoding[i] = hist / hist.sum() if hist.sum() > 0 else hist
        
        # 2. 使用统计特征
        stats_encoding = torch.stack([
            distance_matrix.mean(dim=1),  # 平均距离
            distance_matrix.std(dim=1),   # 距离标准差
            distance_matrix.min(dim=1).values,  # 最小距离
            distance_matrix.max(dim=1).values   # 最大距离
        ], dim=1)
        
        # 3. 使用PCA或类似降维（这里用MLP模拟）
        stats_encoded = self.distance_encoder(stats_encoding.unsqueeze(-1)).squeeze(1)
        
        # 合并两种编码
        pos_encoding = torch.cat([hist_encoding, stats_encoded], dim=1)
        
        return pos_encoding
    
    def add_global_node(self, x, edge_index, edge_attr, batch, distance_matrix):
        """添加虚拟全局节点"""
        num_nodes = x.size(0)
        global_feature = self.global_node_feature.expand(1, -1)
        x_with_global = torch.cat([x, global_feature], dim=0)
        
        # 创建全局节点连接
        global_idx = num_nodes
        global_edges = self.create_global_edges(num_nodes, global_idx, distance_matrix)
        
        new_edge_index = torch.cat([edge_index, global_edges], dim=1)
        
        # 更新边属性
        if edge_attr is not None:
            global_edge_attr = self.encode_global_edge_distances(distance_matrix, global_idx)
            new_edge_attr = torch.cat([edge_attr, global_edge_attr], dim=0)
        else:
            new_edge_attr = None
        
        # 更新batch
        if batch is not None:
            new_batch = torch.cat([batch, torch.tensor([batch.max() + 1], device=batch.device)])
        else:
            new_batch = None
            
        return x_with_global, new_edge_index, new_edge_attr, new_batch, global_idx
    
    def create_global_edges(self, num_nodes, global_idx, distance_matrix):
        """创建全局节点边"""
        # 全局节点到所有节点
        global_to_all = torch.stack([
            torch.full((num_nodes,), global_idx, device=distance_matrix.device),
            torch.arange(num_nodes, device=distance_matrix.device)
        ])
        
        # 所有节点到全局节点
        all_to_global = torch.stack([
            torch.arange(num_nodes, device=distance_matrix.device),
            torch.full((num_nodes,), global_idx, device=distance_matrix.device)
        ])
        
        return torch.cat([global_to_all, all_to_global], dim=1)
    
    def encode_global_edge_distances(self, distance_matrix, global_idx):
        """编码全局边的距离属性"""
        num_nodes = distance_matrix.size(0)
        mean_distances = distance_matrix.mean(dim=1)  # 每个节点到其他节点的平均距离
        
        # 全局节点到节点的距离属性
        global_out_edges = mean_distances.unsqueeze(1)
        global_in_edges = mean_distances.unsqueeze(1)
        
        return torch.cat([global_out_edges, global_in_edges], dim=0)
    
    def forward(self, data):
        x, edge_index, edge_attr, distance_matrix, batch = (
            data.x, data.edge_index, data.edge_attr, data.distance_matrix, data.batch
        )
        
        # 计算增强的位置编码
        pos_encoding = self.compute_structural_positional_encoding(distance_matrix)
        x_augmented = torch.cat([x, pos_encoding], dim=1)
        x_proj = F.relu(self.input_proj(x_augmented))
        
        # 添加全局节点
        x_with_global, edge_index_with_global, edge_attr_with_global, batch_with_global, global_idx = \
            self.add_global_node(x_proj, edge_index, edge_attr, batch, distance_matrix)
        
        # GNN前向传播
        h1 = F.elu(self.conv1(x_with_global, edge_index_with_global, edge_attr_with_global, distance_matrix))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        
        h2_input = h1 + self.skip_projs[0](x_with_global)
        h2 = F.elu(self.conv2(h2_input, edge_index_with_global, edge_attr_with_global, distance_matrix))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        
        h3_input = h2 + self.skip_projs[1](h1)
        h3 = F.elu(self.conv3(h3_input, edge_index_with_global, edge_attr_with_global, distance_matrix))
        h3 = F.dropout(h3, p=self.dropout, training=self.training)
        
        h4_input = h3 + self.skip_projs[2](h2)
        h4 = F.elu(self.conv4(h4_input, edge_index_with_global, edge_attr_with_global, distance_matrix))
        
        # 输出原始节点
        node_repr = h4[:global_idx]
        return self.output_proj(node_repr)

class DistanceAwareGATv2(nn.Module):
    """距离感知的GATv2层，在注意力计算中使用距离信息"""
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1, pos_enc_dim=32):
        super(DistanceAwareGATv2, self).__init__()
        self.heads = heads
        self.out_channels = out_channels
        
        # 标准的GATv2参数
        self.linear = nn.Linear(in_channels, heads * out_channels)
        self.attn = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels + pos_enc_dim))
        
        # 距离编码器
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, pos_enc_dim // 2),
            nn.ReLU(),
            nn.Linear(pos_enc_dim // 2, pos_enc_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn)
    
    def forward(self, x, edge_index, edge_attr, distance_matrix):
        H, C = self.heads, self.out_channels
        x_proj = self.linear(x).view(-1, H, C)
        
        # 准备注意力计算
        src, dst = edge_index
        x_src = x_proj[src]
        x_dst = x_proj[dst]
        
        # 编码距离信息
        edge_distances = distance_matrix[src, dst].unsqueeze(-1)
        distance_encoding = self.distance_encoder(edge_distances)
        
        # 拼接特征和距离编码
        alpha_input = torch.cat([x_src, x_dst, distance_encoding.unsqueeze(1).expand(-1, H, -1)], dim=-1)
        
        # 计算注意力权重（距离感知）
        alpha = (alpha_input * self.attn).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = torch.exp(alpha - alpha.max())  # 数值稳定性
        
        # 注意力归一化
        row = dst
        alpha_sum = torch.zeros(x.size(0), H, device=x.device).scatter_add_(0, row.unsqueeze(-1).expand(-1, H), alpha)
        alpha = alpha / alpha_sum[row]
        
        # 消息传递
        out = torch.zeros(x.size(0), H, C, device=x.device)
        out = out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, H, C), x_src * alpha.unsqueeze(-1))
        
        return out.view(-1, H * C)

# 使用示例
def create_enhanced_node_features(node_type_onehot, unsearched_area, searching_uav_number, 
                                 allowed_uav_number, estimated_finish_time, distance_matrix):
    """创建增强的节点特征"""
    base_features = [
        node_type_onehot,
        unsearched_area,
        searching_uav_number,
        allowed_uav_number,
        estimated_finish_time
    ]
    base_features = [f for f in base_features if f is not None]
    
    # 添加基于距离的统计特征
    distance_features = torch.stack([
        distance_matrix.mean(dim=1),  # 平均可达性
        distance_matrix.std(dim=1),   # 距离分布离散度
        1.0 / (distance_matrix.mean(dim=1) + 1e-6)  # 可达性分数
    ], dim=1)
    
    return torch.cat(base_features + [distance_features], dim=1)