from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List
import torch
from torch_geometric.data import Data

class RoomType(Enum):
    NORMAL = auto()
    HALLWAY = auto()
    CUT_HALLWAY = auto()
    STAIRS = auto()

class ConnectionType(Enum):
    DOOR = auto()
    WIDE_DOOR = auto()
    NARROW_PASSAGE = auto()
    STAIRS = auto()
    OPEN_SPACE = auto()

@dataclass
class SpaceNode:
    id: str
    area: float
    room_type: RoomType
    discount_coefficient: float
    floor: int
    connection_count: int = 0

@dataclass
class SpaceEdge:
    source: str
    target: str
    length: float
    connection_type: ConnectionType
    discount: float

class BuildingGraph:
    def __init__(self):
        self.nodes: Dict[str, SpaceNode] = {}
        self.edges: Dict[str, List[SpaceEdge]] = {}
    
    def add_node(self, node: SpaceNode):
        self.nodes[node.id] = node
        self.edges[node.id] = []
    
    def add_edge(self, edge: SpaceEdge):
        if edge.source in self.edges:
            self.edges[edge.source].append(edge)

def create_sample_building_graph() -> BuildingGraph:
    """创建示例建筑图"""
    building = BuildingGraph()
    
    nodes = [
        SpaceNode("room_101", 25.0, RoomType.NORMAL, 0.8, 1, 1),
        SpaceNode("room_102", 20.0, RoomType.NORMAL, 0.8, 1, 2),
        SpaceNode("hallway_1f", 50.0, RoomType.HALLWAY, 0.3, 1, 3),
        SpaceNode("stairs_1_2", 15.0, RoomType.STAIRS, 0.6, 1, 2),
        SpaceNode("room_201", 30.0, RoomType.NORMAL, 0.8, 2, 1)
    ]
    
    for node in nodes:
        building.add_node(node)
    
    edges = [
        SpaceEdge("room_101", "hallway_1f", 2.0, ConnectionType.DOOR, 1.2),
        SpaceEdge("room_102", "hallway_1f", 5.0, ConnectionType.DOOR, 1.2),
        SpaceEdge("hallway_1f", "stairs_1_2", 3.0, ConnectionType.OPEN_SPACE, 1.0),
        SpaceEdge("stairs_1_2", "room_201", 2.5, ConnectionType.DOOR, 1.2)
    ]
    
    for edge in edges:
        building.add_edge(edge)
    
    return building

def convert_to_pyg_data(building_graph: BuildingGraph, flight_speed: float = 1.0) -> Data:
    """将BuildingGraph转换为PyG的Data对象"""
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(building_graph.nodes.keys())}
    num_nodes = len(building_graph.nodes)
    
    # 构建节点特征
    node_features = []
    for node_id, node in building_graph.nodes.items():
        numerical_features = [
            node.area,
            node.discount_coefficient,
            float(node.connection_count),
            float(node.floor)
        ]
        
        room_type_onehot = [0] * len(RoomType)
        room_type_onehot[node.room_type.value - 1] = 1
        
        features = numerical_features + room_type_onehot
        node_features.append(features)
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # 构建边索引和边权重
    edge_list = []
    weight_list = []
    
    for source_id, edges in building_graph.edges.items():
        for edge in edges:
            if edge.target in node_id_to_idx:
                source_idx = node_id_to_idx[source_id]
                target_idx = node_id_to_idx[edge.target]
                
                travel_time = (edge.length * edge.discount) / flight_speed
                weight = 1.0 / (travel_time + 1e-8)
                
                edge_list.append([source_idx, target_idx])
                weight_list.append(weight)
                
                # 无向图，添加反向边
                edge_list.append([target_idx, source_idx])
                weight_list.append(weight)
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(weight_list, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes)