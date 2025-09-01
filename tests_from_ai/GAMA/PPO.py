import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import random

class PPO:
    def __init__(self, policy_net, value_net, agent_encoder, node_encoder, 
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
        self.agent_encoder = agent_encoder
        self.node_encoder = node_encoder
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        self.policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(value_net.parameters(), lr=lr)
        
        self.memory = []
        
    def encode_state(self, drones, nodes):
        """
        编码状态
        
        参数:
            drones: 无人机列表
            nodes: 节点列表
            
        返回:
            编码后的状态
        """
        # 编码无人机状态
        drone_states = []
        for drone in drones:
            drone_states.append(self.agent_encoder.encode(drone))
        drone_states = torch.stack(drone_states)
        
        # 编码节点状态
        node_states = []
        for node in nodes:
            node_states.append(self.node_encoder.encode(node))
        node_states = torch.stack(node_states)
        
        # 组合状态
        state = {
            'drones': drone_states,
            'nodes': node_states
        }
        
        return state
    
    def select_action(self, state, drone_idx):
        """
        选择动作
        
        参数:
            state: 当前状态
            drone_idx: 无人机索引
            
        返回:
            动作, 动作的对数概率, 状态值
        """
        # 获取无人机特定状态
        drone_state = state['drones'][drone_idx]
        node_states = state['nodes']
        
        # 通过策略网络获取动作概率
        action_probs = self.policy_net(drone_state, node_states)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # 计算动作的对数概率
        log_prob = dist.log_prob(action)
        
        # 通过价值网络获取状态值
        value = self.value_net(drone_state, node_states)
        
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

class MultiDroneEnv:
    """
    多无人机环境类
    """
    def __init__(self, drones, nodes):
        self.drones = drones
        self.nodes = nodes
        self.current_step = 0
        self.max_steps = 1000  # 最大步数
        
    def reset(self):
        """重置环境"""
        for drone in self.drones:
            drone.reset()
        for node in self.nodes:
            node.reset()
        self.current_step = 0
        return self.get_state()
    
    def get_state(self):
        """获取当前状态"""
        # 这里需要根据你的具体实现来获取状态
        # 假设你已经实现了获取状态的方法
        state = {
            'drones': [drone.get_state() for drone in self.drones],
            'nodes': [node.get_state() for node in self.nodes]
        }
        return state
    
    def step(self, actions):
        """
        执行动作
        
        参数:
            actions: 每个无人机的动作列表
            
        返回:
            next_state: 下一个状态
            rewards: 奖励列表
            done: 是否结束
            info: 额外信息
        """
        rewards = []
        
        # 每个无人机执行动作
        for i, (drone, action) in enumerate(zip(self.drones, actions)):
            # 更新无人机状态
            drone.update(action)
            
            # 计算奖励
            reward = self.compute_reward(drone, action)
            rewards.append(reward)
        
        # 更新节点状态
        for node in self.nodes:
            node.update()
        
        # 检查是否结束
        self.current_step += 1
        done = self.is_done()
        
        next_state = self.get_state()
        info = {}  # 可以添加一些额外信息
        
        return next_state, rewards, done, info
    
    def compute_reward(self, drone, action):
        """
        计算奖励
        
        参数:
            drone: 无人机
            action: 动作
            
        返回:
            reward: 奖励值
        """
        # 这里需要根据你的任务设计奖励函数
        # 例如: 搜索到新区域的奖励, 避免碰撞的惩罚, 能量消耗的惩罚等
        reward = 0
        
        # 示例: 如果无人机搜索到了新节点，给予正奖励
        if drone.has_discovered_new_node():
            reward += 10
        
        # 示例: 如果无人机发生碰撞，给予负奖励
        if drone.has_collision():
            reward -= 5
        
        # 示例: 每一步消耗能量，给予小负奖励
        reward -= 0.1
        
        return reward
    
    def is_done(self):
        """
        检查是否结束
        
        返回:
            done: 是否结束
        """
        # 检查是否所有节点都被搜索过
        all_nodes_searched = all(node.is_searched for node in self.nodes)
        
        # 检查是否达到最大步数
        max_steps_reached = self.current_step >= self.max_steps
        
        return all_nodes_searched or max_steps_reached

def train():
    """
    训练函数
    """
    # 初始化环境
    num_drones = 3
    num_nodes = 10
    
    # 创建无人机和节点实例
    drones = [Drone() for _ in range(num_drones)]  # 假设你已经实现了Drone类
    nodes = [Node() for _ in range(num_nodes)]     # 假设你已经实现了Node类
    
    env = MultiDroneEnv(drones, nodes)
    
    # 初始化网络
    policy_net = PolicyNet()  # 假设你已经实现了PolicyNet
    value_net = ValueNet()    # 假设你已经实现了ValueNet
    agent_encoder = AgentEncoder()  # 假设你已经实现了AgentEncoder
    node_encoder = NodeEncoder()    # 假设你已经实现了NodeEncoder
    
    # 初始化PPO
    ppo = PPO(policy_net, value_net, agent_encoder, node_encoder)
    
    # 训练参数
    num_episodes = 1000
    max_steps = 1000
    
    # 训练循环
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            actions = []
            log_probs = []
            values = []
            
            # 每个无人机选择动作
            for i in range(len(env.drones)):
                action, log_prob, value = ppo.select_action(state, i)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
            
            # 执行动作
            next_state, rewards, done, _ = env.step(actions)
            episode_reward += sum(rewards)
            
            # 存储经验
            for i in range(len(env.drones)):
                ppo.store_transition(state, actions[i], log_probs[i], values[i], 
                                   rewards[i], next_state, done and i == len(env.drones)-1)
            
            state = next_state
            
            # 更新网络
            if done or step == max_steps - 1:
                ppo.update()
                break
        
        # 打印训练进度
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {episode_reward:.2f}")
    
    # 保存模型
    torch.save(policy_net.state_dict(), "policy_net.pth")
    torch.save(value_net.state_dict(), "value_net.pth")

if __name__ == "__main__":
    train()