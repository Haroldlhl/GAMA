# environment/search_env.py
import torch
from typing import Dict, List, Tuple, Optional, Any
from .graph_world import WorldGraph, NodeState
from .event_queue import EventQueue
from agents.drone import DroneState, DroneStatus
from config.params import model_config, env_config, training_config

# def add_event(self, time: float, drone_id: str, event_type: str, data: Any = None):
class MultiDroneSearchEnv:
    def __init__(self, graph: WorldGraph, drones: List[DroneState], config: Dict[str, Any]):
        self.graph = graph
        self.drones = drones
        self.config = config
        self.event_queue = EventQueue()
        self.current_time = 0.0
        self.total_reward = 0.0
        self.steps = 0
        
        # 初始化事件队列
        self._initialize_events()
    
    def _initialize_events(self):
        """初始化事件队列"""
        for drone in self.drones:
            if drone.status == DroneStatus.IDLE:
                self.event_queue.add_event(
                    self.current_time,
                    drone.id,
                    'pending'
                )
    
    def reset(self) -> Dict[str, Any]:
        """重置环境到初始状态"""
        self.current_time = 0.0
        self.total_reward = 0.0
        self.steps = 0
        self.event_queue.clear()
        
        # 重置无人机状态
        for drone in self.drones:
            drone.status = DroneStatus.IDLE
            drone.target_node = 0
            drone.task_end_time = 0.0
        
        # 重置节点状态
        for node in self.graph.nodes:
            node.unsearched_area = node.area
            node.searching_drones = []
        
        self._initialize_events()
        return self._get_state()
    
    def _update_time_state(self, current_time, next_time):
        if current_time == next_time:
            return
        for node in self.graph.nodes:
            node.unsearched_area = node.unsearched_area - (next_time - current_time) * node.difficulty * len(node.searching_drones)*self.config['base_search_speed']

        self.current_time = next_time
    
    def _get_state(self) -> Dict[str, Any]:
        """获取当前环境状态, 返回的是物理特征"""
        # 获取环境状态意味着要进行action， 所以此处该改变时间
        event = self.event_queue.get_next_event()
        assert event is not None, "事件队列为空"
        act_uav_id = event.drone_id
        node_features = []
        for node in self.graph.nodes:
            node_features.append(node.encode())
        node_features = torch.stack(node_features, dim=0)
        assert node_features.shape[-1] == model_config['node_feature_in']

        drone_features = []
        for drone in self.drones:
            drone_features.append(drone.encode(self.current_time))
        drone_features = torch.stack(drone_features, dim=0)
        assert drone_features.shape[-1] == model_config['drone_feature_in']
        
        return {
            'node_features': node_features,
            'drone_features': drone_features,
            'current_time': self.current_time,
            'distance_matrix': self.graph.shortest_paths,
            'act_uav_id': act_uav_id
        }
    
    def print_state(self, flag=False):
        if not flag:
            return
        print(f"current time: {self.current_time}")
        for drone in self.drones:
            print(f"drone{drone.id} status: {drone.status}, current_node: {drone.current_node}, target_node: {drone.target_node}, task_end_time: {drone.task_end_time}")
        for node in self.graph.nodes:
            # if node.unsearched_area <= 0:
            #     import pdb; pdb.set_trace()
            print(f"node{node.id} unsearched_area: {node.unsearched_area}, searching_drones: {node.searching_drones}")

    # 输入执行动作的无人机id 和 去往的目标id
    # TODO 修改逻辑   
    def step(self, action: Tuple[str, str], state: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        执行动作
        
        Args:
            action: (drone_id, target_node_id)
        
        Returns:
            state: 新状态
            reward: 预期总收益
            done: 是否结束
            info: 额外信息
        """
        drone_id, target_node_id = action
        before_state = state
        # 执行动作并获取预期收益
        reward = self._execute_action(drone_id, target_node_id)
        self.print_state()
        
        self.total_reward += reward
        self.steps += 1
        after_state = self._get_state()
        
        done = self._is_done()

        event = self.event_queue.get_next_event()
        if not event:
            raise ValueError("事件队列为空")
        if self.current_time == event.time:
            # print(f"current time is equal to event time, event, no need to update time state")
            pass
        else:
            self._update_time_state(self.current_time, event.time)
        next_state = self._get_state()
        
        return before_state, after_state, next_state, reward, done, {'steps': self.steps, 'total_reward': self.total_reward}
    
    def _execute_action(self, drone_id: int, target_node_id: int) -> float:
        """执行动作并返回预期总收益"""
        drone = self.drones[drone_id]
        target_node = self.graph.nodes[target_node_id]
        drone.current_node = drone.target_node
        drone.target_node = target_node_id
        
        # 移除该无人机的所有现有事件
        self.event_queue.remove_events_for_drone(drone_id)
        
        # 根据当前状态和目标任务决定行为
        if drone.status == DroneStatus.MOVING:
            # 如果正在移动，先完成移动（到达目标节点）
            self._complete_movement(drone)
        
        elif drone.status == DroneStatus.SEARCHING:
            # 如果正在搜索，先完成搜索
            self._complete_search(drone)
        
        # 现在无人机处于IDLE状态
        if drone.current_node == target_node_id:
            # 已经在目标节点
            if target_node.is_available():
                return self._start_searching(drone)
            else:
                return self._go_idle(drone)
        else:
            # 需要移动到目标节点
            return self._start_moving(drone)

    def _complete_movement(self, drone: DroneState,) -> float:
        """完成移动，到达目标节点"""
        # 更新无人机位置
        drone.status = DroneStatus.IDLE
        

    def _complete_search(self, drone: DroneState) -> float:
        """完成搜索"""
        drone.status = DroneStatus.IDLE
        if self.graph.nodes[drone.current_node].unsearched_area != 0:
            i = 3
            i += 1
        assert self.graph.nodes[drone.current_node].unsearched_area == 0, "无人机搜索完成后，节点未被搜索完或者剩余面积小于0"
        self.graph.nodes[drone.current_node].remove_searching_drone(drone.id)

    def _start_moving(self, drone: DroneState) -> float:
        """开始移动到目标节点"""
        target_node = self.graph.nodes[drone.target_node]
        distance = self.graph.get_distance(drone.current_node, target_node.id)
        move_time = distance / drone.velocity
        
        # 移动惩罚
        move_penalty = -move_time * self.config['move_penalty_factor']
        
        # 预期搜索收益
        expected_search_reward = 0.0
        if target_node.is_available():
            expected_search_reward = self._calculate_expected_search_reward(target_node, 1) * self.config['move_to_search_bonus']
        
        # 更新无人机状态
        drone.status = DroneStatus.MOVING
        drone.task_end_time = self.current_time + move_time
        
        # 添加到达事件
        self.event_queue.add_event(drone.task_end_time, drone.id)
        
        return move_penalty + expected_search_reward

    def _start_searching(self, drone: DroneState) -> float:
        """开始在节点搜索"""
        target_node = self.graph.nodes[drone.target_node]
        target_node.add_searching_drone(drone.id)
        expected_search_time = target_node.unsearched_area / (
            self.config['base_search_speed'] * target_node.difficulty * len(target_node.searching_drones)
        )
        expected_end_time = self.current_time + expected_search_time
        
        expected_search_reward = expected_search_time * self.config['search_reward_factor']
        
        # 更新现有搜索无人机的事件
        if len(target_node.searching_drones) > 0:
            self._update_existing_searching_drones(target_node, expected_end_time) # 内部完成事件更新
        
        # 更新状态
        drone.status = DroneStatus.SEARCHING
        return expected_search_reward

    def _go_idle(self, drone: DroneState) -> float:
        """进入空闲状态"""
        idle_time = self.config['idle_timeout']
        idle_penalty = -idle_time * self.config['idle_penalty_factor']
        
        drone.status = DroneStatus.IDLE
        drone.task_end_time = self.current_time + idle_time
        
        self.event_queue.add_event(drone.task_end_time, drone.id)
        
        return idle_penalty
            
    def _calculate_expected_search_reward(self, node: NodeState, additional_drones: int = 0) -> float:
        """计算预期搜索收益"""
        if node.unsearched_area == 0:
            return 0.0
        elif node.unsearched_area < 0:
            raise ValueError("节点剩余面积小于0")
        
        # 预期搜索时间
        total_drones = len(node.searching_drones) + additional_drones
        expected_search_time = node.unsearched_area / (
            self.config['base_search_speed'] * node.difficulty * total_drones
        )
        
        # 预期搜索收益 = 搜索时间 * 奖励系数
        return expected_search_time * self.config['search_reward_factor']
    
    def _update_existing_searching_drones(self, target_node: NodeState, new_end_time: float):
        """更新正在搜索该节点的无人机事件"""
        for drone_id in target_node.searching_drones:
            drone = self.drones[drone_id]
            drone.task_end_time = new_end_time
            # 移除旧的事件
            # self.event_queue.remove_events_for_drone(drone_id)
            # 添加新的事件
            self.event_queue.add_event(drone.task_end_time, drone.id)
    
    
    def _handle_arrival(self, drone_id: str):
        """处理到达事件（现在统一由决策网络处理）"""
        # 到达事件现在只是一个决策点，由策略网络决定下一步动作
        # 不需要特殊处理，等待下一次决策即可
        pass
    
    def _handle_search_completion(self, drone_id: str, node_id: str):
        """处理搜索完成事件"""
        drone = self.drones[drone_id]
        node = self.graph.nodes[node_id]
        
        # 计算实际搜索量
        actual_search_time = drone.task_end_time - self.current_time
        actual_search_amount = (self.config['base_search_speed'] * node.difficulty * actual_search_time)
        
        # 更新节点状态
        node.unsearched_area = max(0, node.unsearched_area - actual_search_amount)
        node.update_searching_drones(drone.id, add=False)
        
        # 无人机进入空闲状态，等待下一次决策
        drone.status = DroneStatus.IDLE
        drone.target_node = None
        
        # 不需要添加事件，因为搜索完成本身就是一个决策点
    
    def _is_done(self) -> bool:
        """检查是否结束"""
        all_searched = all(node.unsearched_area <= 0 for node in self.graph.nodes)
        max_steps_reached = self.steps >= self.config['max_steps']
        max_time_reached = self.current_time >= self.config['max_time']
        
        return all_searched or max_steps_reached or max_time_reached