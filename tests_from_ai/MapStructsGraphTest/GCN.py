import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, MessagePassing
from typing import List, Dict
import numpy as np

# 使用您之前的数据结构
from dataclasses import dataclass
from enum import Enum, auto

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

class WeightedGCNConv(MessagePassing):
    """支持边权重的GCN层"""
    def __init__(self, in_channels, out_channels):
        super(WeightedGCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index, edge_weight):
        # 归一化：使用边权重而不是简单的度数
        row, col = edge_index
        deg = torch.zeros(x.size(0), device=x.device)
        deg.index_add_(0, row, edge_weight)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        
        x = self.lin(x)
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class WeightedGCNEncoder(nn.Module):
    """支持边权重的GCN编码器"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.1):
        super(WeightedGCNEncoder, self).__init__()
        self.dropout = dropout
        
        # 创建加权GCN层
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(WeightedGCNConv(dims[i], dims[i+1]))
        
        self.convs = nn.ModuleList(layers)
    
    def forward(self, x, edge_index, edge_weight):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

def build_weighted_graph_data(building_graph: BuildingGraph, flight_speed: float = 1.0) -> Data:
    """构建带权重的图数据"""
    # 创建节点到索引的映射
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(building_graph.nodes.keys())}
    num_nodes = len(building_graph.nodes)
    
    # 1. 构建节点特征矩阵
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
    
    # 2. 构建边索引和边权重
    edge_list = []
    weight_list = []
    
    for source_id, edges in building_graph.edges.items():
        for edge in edges:
            if edge.target in node_id_to_idx:
                source_idx = node_id_to_idx[source_id]
                target_idx = node_id_to_idx[edge.target]
                
                # 计算通行时间（距离越近，时间越短，权重越大）
                travel_time = (edge.length * edge.discount) / flight_speed
                # 将时间转换为权重：时间越短，权重越大（使用倒数）
                weight = 1.0 / (travel_time + 1e-8)  # 加小值防止除零
                
                edge_list.append([source_idx, target_idx])
                weight_list.append(weight)
                
                # 无向图，添加反向边（相同权重）
                edge_list.append([target_idx, source_idx])
                weight_list.append(weight)
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(weight_list, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes)

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
        SpaceEdge("room_101", "hallway_1f", 2.0, ConnectionType.DOOR, 1.2),      # 短距离
        SpaceEdge("room_102", "hallway_1f", 5.0, ConnectionType.DOOR, 1.2),      # 长距离
        SpaceEdge("hallway_1f", "stairs_1_2", 3.0, ConnectionType.OPEN_SPACE, 1.0),
        SpaceEdge("stairs_1_2", "room_201", 2.5, ConnectionType.DOOR, 1.2)
    ]
    
    for edge in edges:
        building.add_edge(edge)
    
    return building

def main():
    """主函数：使用加权GCN对建筑图进行编码"""
    # 1. 创建示例建筑图
    building_graph = create_sample_building_graph()
    
    # 2. 转换为带权重的PyG Data格式
    graph_data = build_weighted_graph_data(building_graph, flight_speed=0.5)
    
    print(f"图数据信息:")
    print(f"  节点数量: {graph_data.num_nodes}")
    print(f"  边数量: {len(graph_data.edge_weight)}")
    print(f"  边权重范围: {graph_data.edge_weight.min():.3f} - {graph_data.edge_weight.max():.3f}")
    
    # 显示边权重信息（距离越近，权重越大）
    print(f"\n边权重示例（距离越近，权重越大）:")
    edge_weights = graph_data.edge_weight.numpy()
    for i, weight in enumerate(edge_weights[:6:2]):  # 显示前3条边（跳过反向边）
        print(f"  边{i+1}权重: {weight:.3f}")
    
    # 3. 创建加权GCN编码器
    input_dim = graph_data.x.shape[1]
    hidden_dims = [64, 32]
    output_dim = 16
    
    model = WeightedGCNEncoder(input_dim, hidden_dims, output_dim)
    
    # 4. 进行编码
    model.eval()
    with torch.no_grad():
        node_embeddings = model(graph_data.x, graph_data.edge_index, graph_data.edge_weight)
    
    # 5. 输出结果
    print(f"\n加权GCN编码结果:")
    print(f"  编码向量形状: {node_embeddings.shape}")
    
    return node_embeddings, graph_data

if __name__ == "__main__":
    embeddings, graph_data = main()
    
    # 验证权重效果：距离近的节点应该有更相似的编码
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 计算节点间的余弦相似度
    similarity_matrix = cosine_similarity(embeddings.numpy())
    
    print(f"\n节点间相似度矩阵:")
    node_ids = list(create_sample_building_graph().nodes.keys())
    for i, id_i in enumerate(node_ids):
        for j, id_j in enumerate(node_ids):
            if i < j:  # 只显示上三角
                sim = similarity_matrix[i, j]
                print(f"  {id_i} - {id_j}: {sim:.3f}")