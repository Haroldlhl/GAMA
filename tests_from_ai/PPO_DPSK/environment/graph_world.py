# environment/graph_world.py
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch
from config import params

class NodeType(Enum):
    ROOM = auto()
    HALLWAY = auto()
    STAIR = auto()
    OTHER = auto()

@dataclass
class NodeState:
    id: int
    node_type: NodeType
    area: float
    unsearched_area: float
    searching_drones: List[int]
    max_drones: int
    difficulty: float = 1.0
    
    def encode(self) -> torch.Tensor:
        """编码节点状态为特征向量"""
        type_onehot = torch.zeros(len(NodeType))
        type_onehot[self.node_type.value - 1] = 1.0
        
        
        features = torch.cat([
            type_onehot, # 归一化
            torch.tensor([self.unsearched_area / params.env_config['max_area']]),
            torch.tensor([len(self.searching_drones)]),
            torch.tensor([self.max_drones - len(self.searching_drones)]),
            torch.tensor([self.difficulty]) 
        ])

        return features
    
    def is_available(self) -> bool:
        """检查节点是否可用（可搜索）"""
        return (self.unsearched_area > 0 and 
                len(self.searching_drones) < self.max_drones)

    def remove_searching_drone(self, drone_id: int):
        if drone_id in self.searching_drones:
            self.searching_drones.remove(drone_id)
    
    def add_searching_drone(self, drone_id: int):
        if self.id == 0:
            pass
        if drone_id in self.searching_drones:
            return
        assert len(self.searching_drones) < self.max_drones, "节点最大无人机数量已满"
        self.searching_drones.append(drone_id)

class WorldGraph:
    def __init__(self, nodes: List[NodeState], distance_matrix: List[List[float]]):
        self.nodes = nodes
        self.distance_matrix = distance_matrix
        self._build_shortest_paths()
    
    # TODO 检查算法正确性
    def _build_shortest_paths(self):
        # """使用Floyd-Warshall算法计算最短路径"""
        # node_ids = range(len(self.nodes))
        # n = len(node_ids)
        
        # # 初始化距离矩阵
        # dist = {i: {j: float('inf') for j in node_ids} for i in node_ids}
        # for i in node_ids:
        #     dist[i][i] = 0
        #     for j in node_ids:
        #         if j in self.distance_matrix[i]:
        #             dist[i][j] = self.distance_matrix[i][j]
        
        # # Floyd-Warshall算法
        # for k in node_ids:
        #     for i in node_ids:
        #         for j in node_ids:
        #             if dist[i][j] > dist[i][k] + dist[k][j]:
        #                 dist[i][j] = dist[i][k] + dist[k][j]
        
        # self.shortest_paths = dist
        self.shortest_paths = self.distance_matrix
    
    def get_distance(self, node1_id: int, node2_id: int) -> float:
        """获取两个节点之间的最短距离"""
        return self.shortest_paths[node1_id][node2_id]
    
    def get_neighbors(self, node_id: int) -> List[int]:
        """获取节点的邻居"""
        return [nid for nid, dist in self.distance_matrix[node_id].items() 
                if dist < float('inf') and nid != node_id]
    
    def update_node(self, node_id: str, **kwargs):
        """更新节点状态"""
        if node_id in self.nodes:
            for key, value in kwargs.items():
                if hasattr(self.nodes[node_id], key):
                    setattr(self.nodes[node_id], key, value)