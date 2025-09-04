import torch
import torch.nn.functional as F
from torch import optim

# 假设这些是已实现的组件
class GATEncoder(torch.nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super().__init__()
        # 实际实现...

class FFNEncoder(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        # 实际实现...

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # 实际实现...
        
    def forward(self, query, key, value):
        # 实际实现...
        return query  # 临时返回，仅作示例

class Actor:
    def __init__(self, lr=1e-4):
        self.GAT = GATEncoder(in_features=10, out_features=128, num_heads=4)
        self.ffn_encoder = FFNEncoder(d_model=128, d_ff=256, dropout=0.1)
        self.MultiHeadAttention = MultiHeadAttention(d_model=128, num_heads=4)
        
        # 初始化优化器，使用类自身的parameters()方法
        self.policy_optimizer = optim.Adam(self.parameters(), lr=lr)
        
    def parameters(self):
        """返回所有可训练参数的生成器"""
        # 收集所有子模块的参数
        for param in self.GAT.parameters():
            yield param
        for param in self.ffn_encoder.parameters():
            yield param
        for param in self.MultiHeadAttention.parameters():
            yield param
    
    def encode_node_states(self, node_states):
        # 实现节点状态编码逻辑
        node_ids = [node.id for node in node_states]
        node_features = torch.tensor([node.features for node in node_states], dtype=torch.float32)
        return node_ids, node_features
    
    def encode_drone_states(self, drone_states, node_feature_dict):
        # 实现无人机状态编码逻辑
        uav_ids = [drone.id for drone in drone_states]
        drone_features = torch.tensor([drone.features for drone in drone_states], dtype=torch.float32)
        return uav_ids, drone_features

    def forward(self, state, event_queue, current_time):
        node_states = state['nodes']
        drone_states = state['drones']

        node_ids, node_features = self.encode_node_states(node_states)
        node_feature = self.GAT(node_features)
        node_feature_dict = {id: node_feature[i] for i, id in enumerate(node_ids)}

        uav_ids, drone_features = self.encode_drone_states(drone_states, node_feature_dict)
        drone_features = self.ffn_encoder(drone_features)
        
        # 选出需要的无人机进行query
        task_end_time, uav = event_queue.get_next_event()
        assert task_end_time > current_time, "任务结束时间必须晚于当前时间"
        query_idx = uav_ids.index(uav.id)
        query_feature = drone_features[query_idx]
        
        # 应用多头注意力（注意：这里需要正确传递query, key, value）
        query_feature = self.MultiHeadAttention(
            query_feature.unsqueeze(0), 
            drone_features.unsqueeze(0),
            drone_features.unsqueeze(0)
        ).squeeze(0)
        
        # 计算注意力分数
        attn_scores = torch.matmul(query_feature, drone_features.transpose(0, 1))
        attn_scores = F.softmax(attn_scores, dim=1)
        
        # 计算注意力后的特征
        attn_feature = torch.matmul(attn_scores.unsqueeze(0), drone_features.unsqueeze(0)).squeeze(0)
        
        # 计算动作
        action = torch.argmax(attn_feature)
        prob = attn_scores[action]
        
        return action, prob, node_features, query_feature
