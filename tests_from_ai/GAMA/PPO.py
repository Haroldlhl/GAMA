import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import random
from actor import Actor
from critic import ValueNetwork
from environment import MultiDroneSearchEnv
# from node import Node, Edge, WorldGraph

class PPO:
    def __init__(self, policy_net, value_net, env,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, 
                 clip_epsilon=0.2, ppo_epochs=4, batch_size=64):
        """
        PPO算法实现
        
        参数:
            policy_net: 策略网络
            value_net: 价值网络
            agent_encoder: 无人机编码器
            node_encoder: 节点编码器
            lr: 学习率
            gamma: 折扣因子
            gae_lambda: GAE参数
            clip_epsilon: PPO裁剪参数
            ppo_epochs: PPO更新轮数
            batch_size: 批大小
        """
        self.policy_net = policy_net
        self.value_net = value_net
        self.env = env
        # self.agent_encoder = agent_encoder
        # self.node_encoder = node_encoder
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        self.policy_optimizer = policy_net.get_optimizer()
        self.value_optimizer = optim.Adam(value_net.parameters(), lr=lr)
        
        self.memory = []
        
    
    def select_action(self, state, event_queue, current_time):
        """
        选择动作
        
        参数:
            state: 当前状态
            drone_idx: 无人机索引
            
        返回:
            动作, 动作的对数概率, 状态值
        """
        # 获取无人机特定状态
        state = self.env.get_state()
        node_states, drone_states = state['nodes'], state['drones']
        nxt_node_idx, prob, node_features, drone_query_feature = self.policy_net.forward(state, event_queue, current_time)
        
        action = torch.tensor(nxt_node_idx)
        log_prob = torch.log(prob)
        value = self.value_net(node_features, drone_query_feature, nxt_node_idx)
        return action.item(), log_prob, value
    
    def store_transition(self, state, action, log_prob, value, reward, next_state, done):
        """
        存储经验到记忆缓冲区
        
        参数:
            state: 当前状态
            action: 动作
            log_prob: 动作的对数概率
            value: 状态值
            reward: 奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.memory.append({
            'state': state,
            'action': action,
            'log_prob': log_prob,
            'value': value,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def compute_gae(self, values, rewards, dones, next_values):
        """
        计算广义优势估计(GAE)
        
        参数:
            values: 状态值数组
            rewards: 奖励数组
            dones: 结束标志数组
            next_values: 下一个状态值数组
            
        返回:
            advantages: 优势函数数组
            returns: 回报数组
        """
        advantages = []
        gae = 0
        returns = []
        
        # 反向计算GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t+1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        advantages = torch.tensor(advantages)
        returns = torch.tensor(returns)
        
        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self):
        """
        更新网络参数
        """
        if len(self.memory) < self.batch_size:
            return
        
        # 从记忆中提取数据
        states = [m['state'] for m in self.memory]
        actions = torch.tensor([m['action'] for m in self.memory])
        old_log_probs = torch.cat([m['log_prob'] for m in self.memory])
        values = torch.cat([m['value'] for m in self.memory])
        rewards = torch.tensor([m['reward'] for m in self.memory])
        next_states = [m['next_state'] for m in self.memory]
        dones = torch.tensor([m['done'] for m in self.memory])
        
        # 计算下一个状态的值
        next_values = []
        for i, next_state in enumerate(next_states):
            if dones[i]:
                next_values.append(0.0)
            else:
                # 这里需要根据你的网络结构调整
                drone_state = next_state['drones'][0]  # 假设只有一个无人机
                node_states = next_state['nodes']
                next_value = self.value_net(drone_state, node_states).detach()
                next_values.append(next_value.item())
        next_values = torch.tensor(next_values)
        
        # 计算GAE和回报
        advantages, returns = self.compute_gae(values.detach().numpy(), 
                                              rewards.numpy(), 
                                              dones.numpy(), 
                                              next_values.numpy())
        
        # 多轮PPO更新
        for _ in range(self.ppo_epochs):
            # 随机打乱数据
            indices = list(range(len(self.memory)))
            random.shuffle(indices)
            
            # 分批处理
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # 获取批次数据
                batch_states = [states[i] for i in batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 计算新的动作概率和值
                new_log_probs = []
                new_values = []
                for i in batch_indices:
                    state = states[i]
                    action = actions[i]
                    
                    # 这里需要根据你的网络结构调整
                    drone_state = state['drones'][0]  # 假设只有一个无人机
                    node_states = state['nodes']
                    
                    # 计算新的动作概率
                    action_probs = self.policy_net(drone_state, node_states)
                    dist = Categorical(action_probs)
                    new_log_prob = dist.log_prob(action)
                    new_log_probs.append(new_log_prob)
                    
                    # 计算新的值
                    value = self.value_net(drone_state, node_states)
                    new_values.append(value)
                
                new_log_probs = torch.stack(new_log_probs)
                new_values = torch.stack(new_values).squeeze()
                
                # 计算策略损失
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算值函数损失
                value_loss = nn.MSELoss()(new_values, batch_returns)
                
                # 计算熵奖励
                entropy = dist.entropy().mean()
                
                # 总损失
                total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                # 更新网络
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.policy_optimizer.step()
                self.value_optimizer.step()
        
        # 清空记忆
        self.memory = []



def create_test_env1():
    node1 = {
        'id': 'Node_1',
        'unsearched_area': 0,
        'searching_uav': 2,
        'allowed_uav_numbe': 10,
        'estimate_time': 0,
    }

    node2 = {
        'id': 'Node_2',
        'unsearched_area': 10,
        'searching_uav': 0,
        'allowed_uav_number': 1,
        'estimate_time': 0,
    }
    node3 = {
        'id': 'Node_3',
        'unsearched_area': 100,
        'searching_uav': 0,
        'allowed_uav_number': 1,
        'estimate_time': 0,
    }
    node4 = {
        'id': 'Node_4',
        'unsearched_area': 100,
        'searching_uav': 0,
        'allowed_uav_number': 1,
        'estimate_time': 0,
    }
    node5 = {
        'id': 'Node_5',
        'unsearched_area': 100,
        'searching_uav': 0,
        'allowed_uav_number': 1,
        'estimate_time': 0,
    }


    nodes = [node1, node2, node3, node4, node5, ]
    drones = []
    for i in range(2):
        drone = {
            'id': f'Drone_{i}',
            'target_id': 'Node_1',
            'status': 'idle',
            'task_end_time': 0,
        }
        drones.append(drone)

    matrix = [[0, 4, 7, 3, 6],
            [4, 0, 3, 7, 10],
            [7, 3, 0, 10, 13],
            [3, 7, 10, 0, 9],
            [6, 10, 13, 9, 0],
        ]

    distance_matrix = dict()
    for i in range(1, 6):
        distance_matrix[f"Node_{i}"] = dict()
        for j in range(1, 6):
            distance_matrix[f"Node_{i}"][f"Node_{j}"] = matrix[i-1][j-1]
    return nodes, drones, distance_matrix

def train():
    """
    训练函数 - 支持矢量奖励版本
    """
    # 初始化环境
    num_drones = 3
    num_nodes = 10
    
    # 创建无人机和节点实例
    # drones = [Drone() for _ in range(num_drones)]
    # nodes = [Node() for _ in range(num_nodes)]
    nodes, drones, distance_matrix = create_test_env1()
    
    env = MultiDroneSearchEnv(drones, nodes, distance_matrix)
    
    # 初始化网络
    policy_net = Actor()
    value_net = ValueNetwork(d_model=128, num_heads=4, hidden_dim=256)

    
    # 初始化PPO
    ppo = PPO(policy_net, value_net, env)
    
    # 训练参数
    num_episodes = 1000
    max_steps = 1000
    
    # 训练循环
    for episode in range(num_episodes):
        state = env.reset()
        episode_rewards = [0] * num_drones  # 改为矢量，每个无人机独立奖励
        
        for step in range(max_steps):
            state = env.get_state() # TODO get states 后 event 为空
            curr_state = state.copy()
            _, act_uav = env.event_queue.get_next_event()
            act_uav_idx = None
            for i in range(len(env.drones)):
                tmp_drone = env.drones[i]
                if env.drones[i]['id'] == act_uav['id']:
                    act_uav_idx = i
                    break
            assert act_uav_idx is not None, "无人机idx找不到"
            # 无人机选择动作
            action, log_prob, value = ppo.select_action(state, env.event_queue, env.current_time)

            
            # 执行动作 - 现在rewards是矢量 [drone1_reward, drone2_reward, ...]
            next_state, reward, done = env.step(action)
            
            # 更新执行动作的无人机的累计奖励
            episode_rewards[act_uav_idx] += reward
            
            # 存储经验 - 每个无人机存储自己的奖励
  
            ppo.store_transition(
                state=curr_state,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=reward,  # 使用矢量奖励中的对应分量
                next_state=next_state,
                done=done
            )
            
            state = next_state
            
            # 更新网络
            if done or step == max_steps - 1:
                ppo.update()
                break
        
        # 打印训练进度 - 显示每个无人机的独立奖励和总奖励
        if episode % 10 == 0:
            total_reward = sum(episode_rewards)
            reward_str = ", ".join([f"Drone{i}: {r:.2f}" for i, r in enumerate(episode_rewards)])
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Individual: [{reward_str}]")
    
    # 保存模型
    torch.save(policy_net.state_dict(), "policy_net.pth")
    torch.save(value_net.state_dict(), "value_net.pth")

if __name__ == "__main__":
    train()