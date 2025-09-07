# train_ppo.py
import torch
import numpy as np
from environment.graph_world import  WorldGraph, NodeState, NodeType
from environment.event_queue import EventQueue
from environment.search_env import MultiDroneSearchEnv
from agents import DroneState, DroneStatus, PPOAgent
from config.params import training_config, env_config, model_config
import json

def create_test_environment():
    """创建测试环境"""
    # 创建节点
    nodes = [
        NodeState(0, NodeType.HALLWAY, 0, 0, [], 0, 1.0),
        NodeState(1, NodeType.ROOM, 80.0, 80.0, [], 2, 1.0),
        NodeState(2, NodeType.ROOM, 50.0, 50.0, [], 2, 1.0),
        NodeState(3, NodeType.ROOM, 90.0, 90.0, [], 2, 1.0),
    ]
    
    # 创建距离矩阵
    distance_matrix = [
        [0, 10, 7, 15],
        [10, 0, 5, 8],
        [7, 5, 0, 10],
        [15, 8, 10, 0],
    ]
    
    graph = WorldGraph(nodes, distance_matrix)
    
    # 创建无人机
    drones = [
        DroneState(0, DroneStatus.PENDING, 0, velocity=2.0),
        DroneState(1, DroneStatus.PENDING, 0, velocity=2.0),
        DroneState(2, DroneStatus.PENDING, 0, velocity=2.0),
    ]
    
    return MultiDroneSearchEnv(graph, drones, env_config)

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建环境
    env = create_test_environment()
    
    # 创建智能体
    node_feature_dim = model_config['node_feature_in']  # 根据NodeState.encode()的输出维度调整
    drone_feature_dim = model_config['drone_feature_in']  # 根据DroneState.encode()  的输出维度调整
    hidden_dim = model_config['hidden_dim']
    feature_out = model_config['feature_out']
    num_heads = model_config['num_heads']
    num_nodes = len(env.graph.nodes)
    num_drones = len(env.drones)
    
    agent = PPOAgent()
    
    # 训练循环
    for episode in range(training_config['num_episodes']):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 选择动作
            action, log_prob, value = agent.select_action(state)
            
            # 执行动作（简化处理，实际需要映射到具体的无人机动作）
            act_uav_id = state['act_uav_id']
            # print(f"at time {env.current_time}, drone{act_uav_id} go to node{action} execute task")
            action_dict = (act_uav_id, action)
            state_before_action, state_after_action, next_state, reward, done, info = env.step(action_dict, state)
            
            # 存储经验
            agent.store_transition(state_before_action, action, log_prob, value, reward, state_after_action, done)
            
            # 更新状态
            state = next_state
            episode_reward += reward
            
            # 更新网络
            if len(agent.memory) >= training_config['batch_size']:
                agent.update()
        
        # 记录训练进度
        if episode % training_config['log_interval'] == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Steps: {info['steps']}")
    
    # 保存模型
    agent.save("trained_agent.pth")
    print("Training completed!")

if __name__ == "__main__":
    main()