from MapGenerator import MapGenerator
from MapGraph import MapGraph, MapNode
import numpy as np

# 1. 创建生成器
generator = MapGenerator(grid_width=50, grid_height=50)

# 2. 生成一批训练地图
train_maps = generator.generate_batch(num_maps=100, rooms_per_map=8)

# 3. 保存地图
for i, map_graph in enumerate(train_maps):
    map_graph.to_json(f"train_map_{i}.json")

# 4. 为DRL准备输入
def prepare_drl_input(map_graph: MapGraph):
    """准备DRL模型的输入：节点特征和距离矩阵"""
    node_features = []
    for node in map_graph.nodes.values():
        features = [
            node.area,
            node.search_cost, 
            node.risk_level,
            1.0 if node.node_type == "room" else 0.0  # 类型特征
        ]
        node_features.append(features)
    
    # 确保已构建距离矩阵
    if map_graph.distance_matrix is None:
        map_graph.build_distance_matrix()
    
    return np.array(node_features), map_graph.distance_matrix

# 为每个地图准备训练数据
train_data = [prepare_drl_input(map_graph) for map_graph in train_maps]