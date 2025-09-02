#### 1. `Environment` (环境类)
# **职责**：模拟器的核心，管理整个世界的状态演进、物理规则和与智能体的交互。
# **主要属性**：
# - `graph`: 一个`WorldGraph`对象，代表地图。
# - `drones`: 一个`Drone`对象的列表，代表所有无人机。
# - `current_time`: 当前模拟时间。
# - `event_queue`: 一个优先队列（最小堆），用于管理异步事件（如无人机到达、搜索完成）。

# **主要方法**：
# - `step()`: 推进模拟时间。从`event_queue`中取出下一个事件并处理（例如，将无人机状态从`移动中`改为`到达`）。
# - `get_global_state()`: 返回当前全局状态的抽象表示，供Critic网络使用。
# - `calculate_reward(drone_id)`: 根据预定义的奖励规则，计算某个无人机在上一时间步获得的奖励。
# - `reset()`: 重置环境到初始状态，用于开始一个新的训练周期（episode）。
# - `register_event(event)`: 将一个未来事件（如预计到达时间）加入事件队列。

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from node import WorldGraph, Node, encode_node_type
from drone import Drone, encode_uav_status
from event_queue import EventQueue
import torch
import torch.nn.functional as F
from actor import Actor

def node_available(node):
    return node['allowed_uav_number'] > 0 and node['unsearched_area'] > 0

class MultiDroneSearchEnv:
    def __init__(self, drones, nodes, distance_matrix):
        """
        初始化多无人机搜索环境
        
        参数:
            drones: 无人机列表
            nodes: 节点(房间)列表
        """
        self.drones = drones
        self.nodes = nodes
        self.current_time = 0.0
        self.max_time = 1000.0
        self.episode_reward = 0
        self.event_queue = EventQueue()
        self.distance_matrix = distance_matrix
        for drone in self.drones:
            self.event_queue.add_event(drone.task_end_time, drone)
        
    def reset(self):
        """
        重置环境到初始状态
        
        返回:
            encoded_state: 编码后的初始状态
        """
        # 重置所有无人机
        for drone in self.drones:
            drone.reset()
        
        # 重置所有节点
        for node in self.nodes:
            node.reset()
        
        self.current_time = 0.0
        self.max_time = 1000.0
        self.episode_reward = 0.0
        for drone in self.drones:
            self.event_queue.add_event(drone.task_end_time, drone)
        
        return self.get_state()

    def clear(self):
        """
        清空环境
        """
        self.drones = []
        self.nodes = []
        self.event_queue = EventQueue()
        self.current_time = 0.0
        self.max_time = 1000.0
        self.episode_reward = 0.0
    
    def get_state(self):
        """
        获取当前环境的编码状态
        
        返回:
            encoded_state: 编码后的状态表示
        """
        # 需要知道无人机在搜索谁，但是不需要知道哪一个无人机在搜索该节点
        # 编码无人机状态
        drone_states = []
        for drone in self.drones:
            drone_state = {
                'drone_id': drone.id,
                'target_id': drone.target_id,
                'status': drone.status,
                'task_end_time': drone.task_end_time
            }
            drone_states.append(drone_state)
        
        # 编码节点状态
        node_states = []
        for node in self.nodes:
            node_state = {
                'node_id': node.id,
                'unsearched_area': node.unsearched_area,
                'searching_uav_number': node.searching_uav_number,
                'allowed_uav_number': node.allowed_uav_number,
                'estimate_time': node.estimate_time
            }
            node_states.append(node_state)
        
        # 组合成完整状态表示， 实际next event 也不需要，但是返回会更方便
        state = {
            'drones': drone_states,
            'nodes': node_states,
            'event': self.event_queue.get_next_event(),
        }
        
        return state
    
    def set_state(self, state, distance_matrix):
        """
        设置环境到特定状态, 需要写 event_queue
        """
        self.clear()
        # 设置节点状态
        for i, node_state in enumerate(state['nodes']):
            node = Node(
                id=node_state['id'],
                type=node_state['type'],
                unsearched_area=node_state['unsearched_area'],
                searching_uav_number=node_state['searching_uav_number'],
                allowed_uav_number=node_state['allowed_uav_number'],
                estimate_time=node_state['estimate_time']
            )
            self.nodes.append(node)
        
        # 设置无人机状态
        for i, drone_state in enumerate(state['drones']):
            drone = Drone(
                id=drone_state['drone_id'],
                target_id=drone_state['target_id'],
                status=drone_state['status'],
                task_end_time=drone_state['task_end_time']
            )
            self.drones.append(drone)
        
        for drone in self.drones:
            self.event_queue.add_event(drone.task_end_time, drone)
    
    def encode_node_states(self, node_states):
        node_features = []
        node_ids = []
        for node_state in node_states:
            one_hot_type = encode_node_type(node_state['type'])
            features = one_hot_type
            features.append(node_state['unsearched_area'])
            features.append(node_state['searching_uav_number'])
            features.append(node_state['allowed_uav_number'])
            features.append(node_state['estimate_time'])
            node_features.append(features)
            node_features = torch.tensor(node_features, dtype=torch.float)
            node_ids.append(node_state['id'])
        
        return node_ids, node_features
        
    def encode_drone_states(self, drone_states, node_states):
        drone_features, uav_ids = [], []
        for drone_state in drone_states:
            feature = node_states[drone_state['target_id']]
            feature.extend(encode_uav_status(drone_state['status']))
            feature.append(drone_state['task_end_time'])
            drone_features.append(feature)
            uav_ids.append(drone_state['id'])
        drone_features = torch.tensor(drone_features, dtype=torch.float)
        return uav_ids, drone_features

    # 输入当前状态, 返回新状态和奖励
    def act(self, state, nxt_node_id):
        current_time = self.current_time

        # 执行动作
        target_node = state['nodes'][nxt_node_id]
        _, act_uav = self.event_queue.get_next_event()
        self.event_queue.remove_event(act_uav)


        # 首先判断无人机是否在目标房间, moving 且 task end time  < current_time 则不在
        # 如果是 task_end_time ==current_time, 但是 target_id 不为 目标. 不在
        assert act_uav['task_end_time'] == current_time, "优先队列出问题了!"
        if act_uav['target_id'] != target_node['id']:
            # 无人机不在目标房间
            act_uav['status'] = 'moving'
            act_uav['task_end_time'] = current_time + self.distance_matrix[act_uav['current_node_id']][target_node['id']]
            act_uav['target_id'] = target_node['id']
            self.event_queue.add_event(act_uav['task_end_time'], act_uav)
        else:
            # 无人机在目标房间
            if target_node['allowed_uav_number'] == 0 or target_node['unsearched_area'] == 0:
                # 无法搜索, 无人机空闲
                act_uav['status'] = 'idle'
                act_uav['target_id'] = target_node['id']
                act_uav['task_end_time'] = current_time + 10
                self.event_queue.add_event(act_uav['task_end_time'], act_uav)
            elif target_node['searching_uav_number'] == 0:
                act_uav['status'] = 'searching'
                act_uav['target_id'] = target_node['id']
                target_node['searching_uav_number'] += 1
                target_node['allowed_uav_number'] -= 1
                estimate_end_time = target_node['unsearched_area'] / (10*target_node['searching_uav_number'])
                act_uav['task_end_time'] = current_time + estimate_end_time
                target_node.estimate_time = estimate_end_time+estimate_end_time
                self.event_queue.add_event(act_uav['task_end_time'], act_uav)
            else:
                act_uav['target_id'] = target_node['id']
                target_node['searching_uav_number'] += 1
                target_node['allowed_uav_number'] -= 1
                estimate_end_time = target_node['unsearched_area'] / (10*target_node['searching_uav_number'])
                act_uav['task_end_time'] = current_time + estimate_end_time
                target_node.estimate_time = estimate_end_time+estimate_end_time
                
                comp_drons = []
                for drone in state['drones']:
                    if drone['target_id'] == target_node['id'] and drone['status'] == 'searching':
                        comp_drons.append(drone)
                act_uav['status'] = 'searching'
                self.event_queue.add_event(act_uav['task_end_time'], act_uav)

                for drone in comp_drons:
                    drone['task_end_time'] = current_time + estimate_end_time
                    self.event_queue.add_event(drone['task_end_time'], drone)

        

        # 更新房间状态, 计算奖励
        nxt_time, _ = self.event_queue.get_next_event()
        elapsed_time = nxt_time - current_time
        for node in state['nodes']:
            node['unsearched_area'] -= elapsed_time * (10*node['searching_uav_number'])
        
        if act_uav['status'] == 'searching':
            reward = 5*elapsed_time
        elif act_uav['status'] == 'moving':
            reward = -2*elapsed_time
        elif act_uav['status'] == 'idle':
            reward = -10*elapsed_time
        
        return state, reward
        

    def step(self, action):
        '''
        # 1. 验证输入动作
        # 2. 保存执行前的状态（用于奖励计算）
        # 3. 执行动作并更新无人机状态
        # 4. 更新节点状态（基于无人机的新位置）
        # 5. 检查环境事件（碰撞、完成任务等）
        # 6. 更新环境步数和全局状态
        # 7. 检查终止条件
        # 8. 获取下一个状态 
        # 9. 收集信息
        '''


        state = self.get_state()
        # 执行动作
        state, reward = self.act(state, action)
        self.episode_reward = rewards

        done_flag = self._is_done()
        if done_flag:
            self.episode_reward[:] += 50

        self.current_step += 1
        return state, reward, done_flag

    
    
    def _is_done(self):
        """
        检查episode是否结束
        """
        # 所有节点都被搜索
        all_searched = all(node['unsearched_area'] == 0 for node in self.state['nodes'])
        
        # 达到最大步数
        max_steps_reached = self.current_step >= self.max_steps
        
        return all_searched or max_steps_reached
    
    # def render(self, mode='human'):
    #     """
    #     渲染环境状态
    #     """
    #     if mode == 'human':
    #         print(f"Step: {self.current_step}")
    #         print(f"Total Reward: {self.episode_reward}")
    #         print("Drones:")
    #         for i, drone in enumerate(self.drones):
    #             print(f"  Drone {i}: Pos={drone.position}, Battery={drone.battery}, "
    #                   f"Node={drone.current_node}, Status={drone.status}")
    #         print("Nodes:")
    #         for i, node in enumerate(self.nodes):
    #             print(f"  Node {i}: Pos={node.position}, Searched={node.is_searched}, "
    #                   f"Priority={node.search_priority}")
    
    def get_status(self):
        """
        获取环境状态信息（简化版本）
        """
        return {
            'step': self.current_step,
            'total_reward': self.episode_reward,
            'searched_nodes': sum(1 for node in self.nodes if node.is_searched),
            'total_nodes': len(self.nodes),
            'active_drones': sum(1 for drone in self.drones if drone.battery > 0)
        }
    
    def close(self):
        """
        清理环境资源
        """
        # 清理任何需要的资源
        pass
