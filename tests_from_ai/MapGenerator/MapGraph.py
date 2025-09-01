import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from heapq import heappush, heappop

@dataclass
class MapNode:
    """地图节点，代表一个可搜索的区域（如房间）"""
    id: int
    name: str  # 可选，用于调试
    area: float  # 区域面积 (m²)
    search_cost: float  # 搜索该区域所需时间成本
    risk_level: float = 0.0  # 风险等级
    node_type: str = "room"  # room, hallway, stairwell等
    grid_positions: List[Tuple[int, int]] = None  # 在底层网格上的代表点坐标
    
    def __post_init__(self):
        if self.grid_positions is None:
            self.grid_positions = []
        # 搜索成本可以与面积成正比
        if self.search_cost <= 0:
            self.search_cost = self.area * 0.5  # 默认系数

class MapGraph:
    """
    拓扑地图图结构
    使用节点表示房间，边权重表示实际最短路径距离（包含走廊等路径成本）
    """
    
    def __init__(self):
        self.nodes: Dict[int, MapNode] = {}  # node_id -> MapNode
        self.edges: Dict[Tuple[int, int], float] = {}  # (node_i, node_j) -> distance
        self.distance_matrix: Optional[np.ndarray] = None  # 全节点距离矩阵
        self._node_index_map: Dict[int, int] = {}  # node_id -> matrix_index
        
    def add_node(self, node_id: int, area: float, search_cost: float = 0,
                 risk_level: float = 0.0, node_type: str = "room", 
                 name: str = "") -> MapNode:
        """添加一个新节点"""
        if search_cost <= 0:
            search_cost = area * 0.5  # 默认搜索成本
        
        node = MapNode(
            id=node_id,
            name=name or f"node_{node_id}",
            area=area,
            search_cost=search_cost,
            risk_level=risk_level,
            node_type=node_type
        )
        self.nodes[node_id] = node
        return node
    
    def add_edge(self, node_i: int, node_j: int, distance: float, bidirectional: bool = True):
        """添加一条边"""
        self.edges[(node_i, node_j)] = distance
        if bidirectional:
            self.edges[(node_j, node_i)] = distance
    
    def get_distance(self, from_node: int, to_node: int) -> float:
        """获取两节点间距离"""
        if from_node == to_node:
            return 0
        return self.edges.get((from_node, to_node), float('inf'))
    
    def build_distance_matrix(self):
        """构建全节点距离矩阵，使用Floyd-Warshall算法处理稀疏连接"""
        node_ids = sorted(self.nodes.keys())
        n_nodes = len(node_ids)
        self._node_index_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        # 初始化距离矩阵
        dist_matrix = np.full((n_nodes, n_nodes), float('inf'))
        np.fill_diagonal(dist_matrix, 0)
        
        # 填充直接连接的边
        for (i, j), dist in self.edges.items():
            if i in self._node_index_map and j in self._node_index_map:
                idx_i = self._node_index_map[i]
                idx_j = self._node_index_map[j]
                dist_matrix[idx_i, idx_j] = dist
        
        # Floyd-Warshall算法计算所有节点对最短路径
        for k in range(n_nodes):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if dist_matrix[i, j] > dist_matrix[i, k] + dist_matrix[k, j]:
                        dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]
        
        self.distance_matrix = dist_matrix
        return dist_matrix
    
    def get_full_cost(self, from_node: int, to_node: int) -> float:
        """获取从from_node移动到to_node并完成搜索的总成本"""
        if self.distance_matrix is None:
            self.build_distance_matrix()
        
        if from_node not in self._node_index_map or to_node not in self._node_index_map:
            return float('inf')
        
        idx_i = self._node_index_map[from_node]
        idx_j = self._node_index_map[to_node]
        travel_cost = self.distance_matrix[idx_i, idx_j]
        search_cost = self.nodes[to_node].search_cost
        
        return travel_cost + search_cost
    
    def to_json(self, filename: str):
        """导出地图到JSON文件"""
        data = {
            "nodes": [],
            "edges": []
        }
        
        for node in self.nodes.values():
            node_data = {
                "id": node.id,
                "name": node.name,
                "area": node.area,
                "search_cost": node.search_cost,
                "risk_level": node.risk_level,
                "type": node.node_type,
                "grid_positions": node.grid_positions
            }
            data["nodes"].append(node_data)
        
        for (from_node, to_node), dist in self.edges.items():
            edge_data = {
                "from": from_node,
                "to": to_node,
                "distance": dist
            }
            data["edges"].append(edge_data)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def from_json(cls, filename: str) -> 'MapGraph':
        """从JSON文件加载地图"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        map_graph = cls()
        
        for node_data in data["nodes"]:
            node = map_graph.add_node(
                node_data["id"],
                node_data["area"],
                node_data["search_cost"],
                node_data["risk_level"],
                node_data["type"],
                node_data["name"]
            )
            node.grid_positions = [tuple(pos) for pos in node_data["grid_positions"]]
        
        for edge_data in data["edges"]:
            map_graph.add_edge(
                edge_data["from"],
                edge_data["to"],
                edge_data["distance"]
            )
        
        return map_graph