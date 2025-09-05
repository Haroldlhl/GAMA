import torch
import torch.nn.functional as F
from torch import optim
from nets import GATEncoder, FFNEncoder, MultiHeadAttention
import numpy as np

# # 假设这些是已实现的组件
# class GATEncoder(torch.nn.Module):
#     def __init__(self, in_features, out_features, num_heads):
#         super().__init__()
#         # 实际实现...

# class FFNEncoder(torch.nn.Module):
#     def __init__(self, d_model, d_ff, dropout):
#         super().__init__()
#         # 实际实现...

# class MultiHeadAttention(torch.nn.Module):
#     def __init__(self, d_model, num_heads):
#         super().__init__()
#         # 实际实现...
        
#     def forward(self, query, key, value):
#         # 实际实现...
#         return query  # 临时返回，仅作示例

class Actor(torch.nn.Module):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.GAT = GATEncoder(in_features=4, out_features=128, num_heads=4)
        self.ffn_encoder = FFNEncoder(d_model=132, d_ff=256, d_out=128, dropout=0.1)
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
    
    def get_optimizer(self):
        return self.policy_optimizer

    def encode_node_states(self, node_states):
        # 实现节点状态编码逻辑
        node_ids = [node['id'] for node in node_states]
        node_features = []
        for node in node_states:
            unsearched_area = node['unsearched_area']
            searching_uav_number = node['searching_uav_number']
            allowed_uav_number = node['allowed_uav_number']
            estimate_time = unsearched_area / (10*max(1, searching_uav_number))
            node_features.append([unsearched_area, searching_uav_number, allowed_uav_number, estimate_time])
        node_features = torch.tensor(node_features, dtype=torch.float32)
        return node_ids, node_features
    
    def encode_drone_states(self, drone_states, node_feature_dict):
        # 实现无人机状态编码逻辑
        # 先简单实现， 之后考虑用切片的方式快速索引
        drone_features = []
        uav_ids = []
        # status_idle, status_searching, status_moving
        status_idle = [1, 0, 0]
        status_searching = [0, 1, 0]
        status_moving = [0, 0, 1]
        for drone in drone_states:
            uav_ids.append(drone['drone_id'])
            status = drone['status']
            if status == 'idle':
                drone_feature = status_idle.copy()
            elif status == 'searching':
                drone_feature = status_searching.copy()
            elif status == 'moving':
                drone_feature = status_moving.copy()

            drone_feature.append(drone['task_end_time'])
            drone_feature = torch.tensor(drone_feature, dtype=torch.float32)
            taget_feature = node_feature_dict[drone['target_id']]
            drone_feature = torch.cat([drone_feature, taget_feature], dim=-1)
            drone_features.append(drone_feature)
        drone_features = torch.concat([drone_feature.unsqueeze(0) for drone_feature in drone_features], dim=0).unsqueeze(0)
        return uav_ids, drone_features

    def forward(self, state, event_queue, current_time):
        node_states = state['nodes']
        drone_states = state['drones']

        node_ids, node_features = self.encode_node_states(node_states)
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
        node_feature = self.GAT(node_features)
        node_feature_dict = {id: node_feature[0][i] for i, id in enumerate(node_ids)}  # TODO 处理高维度情况 

        uav_ids, drone_features = self.encode_drone_states(drone_states, node_feature_dict)
        drone_features = self.ffn_encoder(drone_features)
        if drone_features.dim() == 2:
            drone_features = drone_features.unsqueeze(0)
        
        # 选出需要的无人机进行query
        task_end_time, uav = event_queue.get_next_event()
        assert task_end_time >= current_time, "任务结束时间必须晚于当前时间"
        query_idx = uav_ids.index(uav['id'])
        query_feature = drone_features[:, query_idx].unsqueeze(0)
        
        # 应用多头注意力（注意：这里需要正确传递query, key, value）
        query_feature = self.MultiHeadAttention(
            query_feature, 
            drone_features,
            drone_features,
        )
        
        # 计算注意力分数
        attn_scores = torch.matmul(query_feature, drone_features.transpose(-2, -1))
        attn_scores = F.softmax(attn_scores, dim=-1)
        
        # 计算注意力后的特征
        attn_feature = torch.matmul(attn_scores, drone_features)
        
        # 计算动作
        prob, action = torch.max(attn_scores, dim=-1)
        
        return action, prob, node_feature, attn_feature
