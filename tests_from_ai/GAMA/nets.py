import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        output = torch.matmul(attn_probs, v)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, q, k, v, mask=None):
        q = self.split_heads(self.w_q(q))
        k = self.split_heads(self.w_k(k))
        v = self.split_heads(self.w_v(v))
        
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)
        output = self.w_o(self.combine_heads(attn_output))
        
        return output

class FFNEncoder(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FFNEncoder, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class GATEncoder(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout=0.1, alpha=0.2):
        super(GATEncoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        
        # Linear transformations for each head
        self.linear = nn.Linear(in_features, out_features)
        
        # Attention mechanism parameters
        self.a = nn.Parameter(torch.empty(size=(2 * self.head_dim, 1)))
        self.leakyrelu = nn.LeakyReLU(alpha)
        
        # Distance weighting parameters
        self.distance_weight = nn.Parameter(torch.tensor(1.0))
        self.distance_bias = nn.Parameter(torch.tensor(0.0))
        
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()

        self.test_distance_matrix = np.array([[0, 4, 7, 3, 6],
                                                [4, 0, 3, 7, 10],
                                                [7, 3, 0, 10, 13],
                                                [3, 7, 10, 0, 9],
                                                [6, 10, 13, 9, 0],
                                                ])
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.a)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, h, adj_matrix, distance_matrix=None):
        """
        Args:
            h: Node features [batch_size, num_nodes, in_features]
            adj_matrix: Adjacency matrix [batch_size, num_nodes, num_nodes]
            distance_matrix: Distance matrix between nodes [batch_size, num_nodes, num_nodes]
                           Smaller values indicate closer nodes
        """
        batch_size, num_nodes, _ = h.size()
        
        # Linear transformation
        h_transformed = self.linear(h)  # [batch_size, num_nodes, out_features]
        h_transformed = h_transformed.view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # Prepare for attention computation
        h_i = h_transformed.unsqueeze(2).repeat(1, 1, num_nodes, 1, 1)  # [batch, nodes, nodes, heads, dim]
        h_j = h_transformed.unsqueeze(1).repeat(1, num_nodes, 1, 1, 1)  # [batch, nodes, nodes, heads, dim]
        
        # Compute attention scores
        attention_input = torch.cat([h_i, h_j], dim=-1)  # [batch, nodes, nodes, heads, 2*dim]
        e = self.leakyrelu(torch.matmul(attention_input, self.a).squeeze(-1))  # [batch, nodes, nodes, heads]
        
        # Apply distance-based attention adjustment
        if distance_matrix is not None:
            # Convert distance to proximity (closer nodes get higher weights)
            # Add small epsilon to avoid division by zero
            proximity = 1.0 / (distance_matrix.unsqueeze(-1) + 1e-8)
            # Learnable scaling of distance influence
            distance_effect = self.distance_weight * proximity + self.distance_bias
            e = e + distance_effect
        
        # Mask attention scores for non-adjacent nodes
        mask = adj_matrix.unsqueeze(-1)  # [batch, nodes, nodes, 1]
        e = e.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(e, dim=2)  # [batch, nodes, nodes, heads]
        attention_weights = self.dropout(attention_weights)
        
        # Aggregate features
        h_j = h_transformed.unsqueeze(1)  # [batch, 1, nodes, heads, dim]
        output = torch.sum(attention_weights.unsqueeze(-1) * h_j, dim=2)  # [batch, nodes, heads, dim]
        
        # Combine heads
        output = output.view(batch_size, num_nodes, self.out_features)
        
        return output

# 示例用法
if __name__ == "__main__":
    # 测试MultiHeadAttention
    batch_size, seq_len, d_model = 2, 10, 64
    num_heads = 8
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    output = mha(x, x, x)
    print(f"MHA input shape: {x.shape}, output shape: {output.shape}")
    
    # 测试FFNEncoder
    d_ff = 256
    ffn = FFNEncoder(d_model, d_ff)
    output = ffn(x)
    print(f"FFN input shape: {x.shape}, output shape: {output.shape}")
    
    # 测试GATEncoder
    num_nodes = 5
    in_features = 32
    out_features = 64
    num_heads = 4
    
    gat = GATEncoder(in_features, out_features, num_heads)
    node_features = torch.randn(batch_size, num_nodes, in_features)
    adj_matrix = torch.ones(batch_size, num_nodes, num_nodes)  # 全连接图
    distance_matrix = torch.randn(batch_size, num_nodes, num_nodes)  # 随机距离矩阵
    
    output = gat(node_features, adj_matrix, distance_matrix)
    print(f"GAT input shape: {node_features.shape}, output shape: {output.shape}")