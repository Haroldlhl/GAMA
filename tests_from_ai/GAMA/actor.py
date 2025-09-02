import torch
import torch.nn.functional as F
from nets import GATEncoder, FFNEncoder, MultiHeadAttention

class Actor:
    def __init__(self):
        self.GAT = GATEncoder()
        self.ffn_encoder = FFNEncoder()
        self.MultiHeadAttention = MultiHeadAttention()


    def forward(self, state, event_queue, current_time):
        node_states = state['nodes']
        drone_states = state['drones']

        node_ids, node_features = self.encode_node_states(node_states)
        node_feature = self.GAT(node_features)
        node_feature_dict = dict()
        for i, id in enumerate(node_ids):
            node_feature_dict[id] = node_feature[i]

        uav_ids, drone_features = self.encode_drone_states(drone_states, node_feature_dict)

        drone_features = self.ffn_encoder(drone_features)
        
        # 选出需要的无人机进行query
        task_end_time, uav = event_queue.get_next_event()
        assert task_end_time > current_time
        query_idx = uav_ids.index(uav.id)
        query_feature = drone_features[query_idx]
        query_feature = self.MultiHeadAttention(query_feature, drone_features)
        # 计算注意力分数
        attn_scores = torch.matmul(query_feature, drone_features.transpose(0, 1))
        attn_scores = F.softmax(attn_scores, dim=1)
        # 计算注意力后的特征
        attn_feature = torch.matmul(attn_scores, drone_features)
        # 计算动作, 实际是节点Idx
        action = torch.argmax(attn_feature)
        prob = attn_scores[action]
        return action, prob, node_features, query_feature