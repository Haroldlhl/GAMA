import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNetwork(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim):
        super().__init__()
        self.d_model = d_model
        
        # 1. 直接使用编码后的输入，省略初始编码层
        
        # 2. 全局上下文编码层：节点间的Self-Attention
        # 让每个节点感知到其他节点的存在（例如，一个遥远但价值极高的节点会影响其他节点的相对价值）
        self.node_self_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # 3. 交叉注意力层：无人机Query与所有节点的交互
        # 这是网络的核心：计算无人机与每个节点的“匹配度”或“相关性”
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 4. 目标节点聚焦层
        # 从交叉注意力的结果中，根据idx提取出被选择节点的特定表示
        # 我们将使用一个简单的选择方法，而不是参数层
        
        # 5. 价值预测头
        # 融合[无人机Query的表示， 被选择节点的增强表示， 它们的交互表示]
        self.value_head = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim), # 融合三种信息
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # 输出一个Q值
        )

    def forward(self, encoded_nodes, encoded_agent_query, node_idx):
        """
        Args:
            encoded_nodes: [batch_size, num_nodes, d_model] 编码后的节点特征
            encoded_agent_query: [batch_size, d_model] 编码后的无人机查询特征
            node_idx: [batch_size] 为无人机选择的节点索引
        Returns:
            q_value: [batch_size, 1] 选择的长期价值
        """
        batch_size, num_nodes, _ = encoded_nodes.shape
        
        # 2. 增强节点表征：让节点信息包含全局上下文
        attended_nodes, _ = self.node_self_attn(
            query=encoded_nodes,
            key=encoded_nodes,
            value=encoded_nodes
        ) # shape: [batch_size, num_nodes, d_model]

        # 准备无人机Query：增加序列维度以匹配Attention输入格式
        # [batch_size, d_model] -> [batch_size, 1, d_model]
        agent_query = encoded_agent_query.unsqueeze(1)

        # 3. 核心：交叉注意力
        # 让无人机Query去“审视”所有增强后的节点
        cross_attn_output, cross_attn_weights = self.cross_attention(
            query=agent_query,        # 无人机作为Query
            key=attended_nodes,       # 节点作为Key
            value=attended_nodes      # 节点作为Value
        ) # cross_attn_output shape: [batch_size, 1, d_model]
        cross_attn_output = cross_attn_output.squeeze(1) # [batch_size, d_model]

        # 4. 提取被选择节点的增强特征
        # 使用gather操作根据idx获取特定节点的特征
        # attended_nodes: [bs, num_nodes, d_model]
        # node_idx: [bs] -> [bs, 1, 1] -> [bs, 1, d_model]
        expanded_idx = node_idx.view(-1, 1, 1).expand(-1, -1, self.d_model)
        selected_node_feat = torch.gather(attended_nodes, 1, expanded_idx)
        selected_node_feat = selected_node_feat.squeeze(1) # [batch_size, d_model]

        # 5. 融合所有信息并预测Q值
        # 拼接：无人机Query， 交互表示， 被选节点特征
        combined_feat = torch.cat([
            encoded_agent_query,      # 原始无人机意图
            cross_attn_output,        # 无人机与所有节点交互后的摘要
            selected_node_feat        # 被选节点的全局增强特征
        ], dim=-1) # [batch_size, 3 * d_model]

        q_value = self.value_head(combined_feat) # [batch_size, 1]
        return q_value