# agents/actor_critic.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from config import params
import torch.nn.functional as F
from config.params import model_config, training_config, env_config

class SimpleDroneDecisionHead(nn.Module):
    def __init__(self, temperature=1.0):
        """ 
        参数: temperature: softmax温度系数，控制概率分布的陡峭程度
        """
        super().__init__()
        self.temperature = temperature
        
    def forward(self, drone_vec, node_vecs, mask=None):
        """
        计算无人机对各房间节点的选择概率
        参数:
            drone_vec: 单个无人机的特征向量，形状为 [batch_size, 1, feat_dim]
            node_vecs: 所有房间节点的特征向量，形状为 [batch_size, num_nodes, feat_dim]
            mask: 节点掩码，用于屏蔽不可选房间，形状为 [batch_size, num_nodes]
            
        返回:
            prob_dist: 房间选择的概率分布，形状为 [batch_size, num_nodes]
        """
        # 确保无人机向量和节点向量维度匹配
        assert drone_vec.shape[-1] == node_vecs.shape[-1], \
            "无人机向量和节点向量的特征维度必须相同"
        
        # 1. 直接通过矩阵乘法计算注意力分数
        # 将无人机向量扩展维度: [batch_size, 1, feat_dim]
        if drone_vec.dim() == 2:
            drone_vec = drone_vec.unsqueeze(1)
        # 矩阵乘法计算相似度: [batch_size, 1, num_nodes] -> [batch_size, num_nodes]
        attention_scores = torch.matmul(drone_vec, node_vecs.transpose(-2, -1)).squeeze(1)
        
        # 2. 应用温度系数
        attention_scores = attention_scores / self.temperature
        
        # 3. 应用掩码（屏蔽不可选的房间）
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # 4. softmax归一化得到概率分布
        prob_dist = F.softmax(attention_scores, dim=-1)
        
        return prob_dist

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # 分割多头
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output


class QueryFusionValueHead(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 可学习的查询向量（用于聚合无人机和节点特征）
        self.feature_dim = params.model_config['feature_out']
        self.hidden_dim = params.model_config['hidden_dim']
        self.drone_query = nn.Parameter(torch.randn(1, 1, self.feature_dim))  # (1,1,128)
        self.node_query = nn.Parameter(torch.randn(1, 1, self.feature_dim))    # (1,1,128)
        
        # 2. 映射层：将融合特征转为标量价值
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.hidden_dim),  # 拼接无人机和节点的全局特征
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
    
    def forward(self, drone_state, node_state):
        # 输入形状：
        # drone_state: (batch, 3, 128)  3架无人机，每架128维特征
        # node_state: (batch, 4, 128)   4个节点，每个128维特征
        
        # --------------------------
        # 用drone_query聚合无人机特征
        # --------------------------
        # 计算查询与每个无人机特征的相似度：(batch, 3, 128) × (1,1,128) → (batch, 3, 1)
        drone_similarity = torch.matmul(drone_state, self.drone_query.transpose(1, 2))  # (batch,3,1)
        drone_attn = F.softmax(drone_similarity, dim=1)  # 注意力权重：(batch,3,1)
        
        # 加权聚合：(batch,3,128) × (batch,3,1) → (batch,1,128) → (batch,128)
        drone_global = torch.sum(drone_attn * drone_state, dim=1)  # (batch,128)
        
        # --------------------------
        # 用node_query聚合节点特征
        # --------------------------
        node_similarity = torch.matmul(node_state, self.node_query.transpose(1, 2))  # (batch,4,1)
        node_attn = F.softmax(node_similarity, dim=1)  # 注意力权重：(batch,4,1)
        node_global = torch.sum(node_attn * node_state, dim=1)  # (batch,128)
        
        # --------------------------
        # 融合并映射为标量
        # --------------------------
        merged = torch.cat([drone_global, node_global], dim=1)  # (batch, 256)
        value = self.fc(merged).squeeze(-1)  # (batch,) 最终标量价值
        
        return value


class ActorCritic(nn.Module):
    def __init__(self, node_feature_dim=None, drone_feature_dim=None, hidden_dim=None, feature_out=None, num_heads=None):
        super().__init__()
        if node_feature_dim is None:
            self.node_feature_dim = params.model_config['node_feature_in']
            self.drone_feature_dim = params.model_config['drone_feature_in']
            self.hidden_dim = params.model_config['hidden_dim']
            self.num_heads = params.model_config['num_heads']
            self.feature_out = params.model_config['feature_out']
        else:
            self.node_feature_dim = node_feature_dim
            self.drone_feature_dim = drone_feature_dim
            self.hidden_dim = hidden_dim
            self.num_heads = num_heads
            self.feature_out = feature_out
        
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.feature_out)
        )
        
        # TODO  encoder 写的太简单，没有注意力交互
        self.drone_encoder = nn.Sequential(
            nn.Linear(self.drone_feature_dim+self.feature_out-1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.feature_out)
        )

        self.drone_multihead_attention = MultiHeadAttention(self.feature_out, self.num_heads)
        self.node_multihead_attention = MultiHeadAttention(self.feature_out, self.num_heads)
        
        self.attention = MultiHeadAttention(self.feature_out, self.num_heads)
        
        self.policy_head = SimpleDroneDecisionHead()
        
        self.value_head = QueryFusionValueHead()
    
    def forward(self, node_states, drone_states, act_uav_id):
        if node_states.dim() == 2:
            node_states = node_states.unsqueeze(0)
        if drone_states.dim() == 2:
            drone_states = drone_states.unsqueeze(0)
        node_embeddings = self.node_encoder(node_states)
        node_embeddings = self.node_multihead_attention(node_embeddings, node_embeddings, node_embeddings)

        indices = drone_states[:, :, -1].long()
        expand_drone_status = node_embeddings.gather(1, indices.unsqueeze(-1).expand(-1, -1, 128))
        drone_embeddings = self.drone_encoder(torch.concat([drone_states[:, :, :-1], expand_drone_status], dim=-1))
        drone_embeddings = self.drone_multihead_attention(drone_embeddings, drone_embeddings, drone_embeddings)
    
        
        # 应用注意力机制
        batch_size = node_embeddings.shape[0]
        batch_indices = torch.arange(batch_size)

        query = drone_embeddings[batch_indices, act_uav_id]
        if batch_size != 1 and query.dim() == 2:
            query = query.unsqueeze(1)
        
        action_logits = self.policy_head(query, node_embeddings)
        values = self.value_head(drone_embeddings, node_embeddings)
        
        return action_logits, values

class PPOAgent:
    def __init__(self):  
        self.policy = ActorCritic()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=training_config['lr'])
        
        self.memory = deque(maxlen=training_config['memory_size'])
        self.batch_size = training_config['batch_size']
        self.gamma = training_config['gamma']
        self.gae_lambda = training_config['gae_lambda']
        self.clip_epsilon = training_config['clip_epsilon']
        self.ppo_epochs = training_config['ppo_epochs']
    
    def select_action(self, state):
        """选择动作"""
        node_features = state['node_features']
        drone_features = state['drone_features']
        
        with torch.no_grad():
            action_logits, value = self.policy(node_features, drone_features, state['act_uav_id'])
        
        # 创建动作分布
        action_probs = torch.softmax(action_logits, dim=0)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value
    
    def store_transition(self, state, action, log_prob, value, reward, next_state, done):
        """存储经验"""
        self.memory.append({
            'state': state,
            'action': action,
            'log_prob': log_prob,
            'value': value,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def compute_advantages(self, rewards, values, dones, next_values):
        """计算优势函数"""
        advantages = []
        gae = 0
        returns = []
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        advantages = torch.tensor(advantages, dtype=torch.float)
        returns = torch.tensor(returns, dtype=torch.float)
        
        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self):
        """更新策略"""
        if len(self.memory) < self.batch_size:
            return
        
        # 准备数据
        batch = list(self.memory)[-self.batch_size:]
        
        states = [item['state'] for item in batch]
        actions = torch.tensor([item['action'] for item in batch])
        old_log_probs = torch.stack([item['log_prob'] for item in batch])
        values = torch.stack([item['value'] for item in batch])
        rewards = torch.tensor([item['reward'] for item in batch])
        next_states = [item['next_state'] for item in batch]
        dones = torch.tensor([item['done'] for item in batch])
        
        # 计算下一个状态的值
        next_values = []
        with torch.no_grad():
            node_features = torch.stack([d['node_features'] for d in next_states], dim=0)
            drone_features = torch.stack([d['drone_features'] for d in next_states], dim=0)
            act_uav_id = torch.tensor([d['act_uav_id'] for d in next_states], dtype=torch.long)
            _, next_values = self.policy(node_features, drone_features, act_uav_id)
        
        # 计算优势函数和回报
        advantages, returns = self.compute_advantages(
            rewards.numpy(), values.mean(dim=1).detach().numpy(),
            dones.numpy(), next_values.numpy()
        )
        
        # PPO更新
        for _ in range(self.ppo_epochs):
            # 计算新的动作概率和值
            new_log_probs = []
            new_values = []
            entropy = 0
            

            node_features = torch.stack([d['node_features'] for d in states], dim=0)
            drone_features = torch.stack([d['drone_features'] for d in states], dim=0)
            act_uav_id = torch.tensor([d['act_uav_id'] for d in states], dtype=torch.long)
            action_logits, value = self.policy(node_features, drone_features, act_uav_id)
            
            action_probs = torch.softmax(action_logits, dim=0)
            dist = Categorical(action_probs)
            new_log_prob = dist.log_prob(actions[0])  # 简化处理
            entropy += dist.entropy().mean()
            
            new_log_probs.append(new_log_prob)
            new_values.append(value.mean())
            
            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.stack(new_values).squeeze()
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 计算策略损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 计算价值损失
            value_loss = nn.MSELoss()(new_values, returns)
            
            # 计算熵奖励
            entropy_bonus = entropy.mean()
            
            # 总损失
            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
            print(f"total_loss: {total_loss}")
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # 清空记忆
        self.memory.clear()
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])