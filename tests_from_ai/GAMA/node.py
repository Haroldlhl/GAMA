# 定义图结构中的节点类，代表一个房间
# 包含 1.节点id：str  2.节点类型：枚举(普通， 过道，楼梯， 其他)
# 3. 节点面积：float  4. 难度因子：float  5. 未搜索面积：float  6. 节点的度：int
# 允许uav 数量 int， 位置（float, float） 在执行任务的无人机数量int 预估任务完成时间float
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

Node_Type = Enum('Node_Type', ['room', 'hallway', 'stair', 'other'])
def encode_node_type(node_type: Node_Type):
    node_type_onehot = [0.0] * len(Node_Type)
    node_type_onehot[node_type.value - 1] = 1
    return node_type_onehot

@dataclass
class Node:
    id: str
    type: Node_Type
    area: float
    floor: int
    config: Config
    unsearched_area: float=-1.0
    position: Tuple[float, float]=(0.0, 0.0)
    neigbors: List[str]=[]
    degree: int=0
    search_speed: float=0.0
    difficulty_factor: float=1.0
    allowed_uav_number: int = 3 # 最大允许无人机数量
    searching_uav: []
    last_change_time: float=0.0

    def updata_unsearched_area(self, global_time):
        self.update_search_speed()
        self.unsearched_area = self.unsearched_area - (global_time - self.last_change_time) * self.search_speed
        self.last_change_time = global_time

    def is_available(self):
        if self.unsearched_area > 0 and self.allowed_uav_number > len(self.searching_uav):
            return True
        return False
    
    def update_search_speed(self):
        self.search_speed = 10 * len(self.searching_uav)

    def get_feature(self, global_time):
        # 1.0 版本 返回： 类型：onehot, 待搜索面积：float, 正在搜索无人机数量int, 可搜索无人机数量int, 预计任务完成时间

        node_type_onehot = [0] * len(Node_Type)
        node_type_onehot[self.type.value - 1] = 1

        self.updata_unsearched_area(global_time)
        unsearched_area = self.unsearched_area / self.config.node.max_area
        searching_uav_number = len(self.searching_uav)
        allowed_uav_number = self.allowed_uav_number - len(self.searching_uav)
        if self.search_speed == 0.0:
            estimate_time = self.remaining_area / self.config.uav.search_speed
        else:
            estimate_time = self.remaining_area / self.search_speed

        return node_type_onehot, unsearched_area, searching_uav_number, allowed_uav_number, estimate_time

# 边，包含 1.起点：str  2.终点：str  3.长度：float  4.权重：float， 时间成本：float
@dataclass
class Edge:
    id: str
    source: str
    target: str
    length: float
    weight: float
    time_cost: float


# #### 2. `WorldGraph` (世界图类)
# **职责**：封装地图的图结构及其所有属性。
# **主要属性**：
# - `nodes`: 字典，`{node_id: Node}`。
# - `edges`: 字典，`{(node_id_i, node_id_j): Edge}`。
# - `shortest_paths`: 缓存节点间的最短路径（时间代价），可用Floyd-Warshall或Dijkstra算法预处理。

# **主要方法**：
# - `get_neighbors(node_id)`: 返回一个节点的所有邻居节点ID。
# - `get_shortest_path_time(start_id, end_id)`: 返回两节点间的最短移动时间。
# - `get_node_embedding(node_id)`: 返回节点的最新编码（结合静态和动态特征）。
# - `update_node_dynamic_attr(node_id, **kwargs)`: 更新节点的动态属性（如`remaining_area`）。

@dataclass
class WorldGraph:
    nodes: Dict[str, Node]
    edges: Dict[Tuple[str, str], Edge]
    shortest_paths: Dict[Tuple[str, str], float]

    def __init__(self, nodes: Dict[str, Node], edges: Dict[Tuple[str, str], Edge]):
        self.nodes = nodes
        self.edges = edges
        self.node_id_to_edge_id = {}
        self.set_node_id_to_edge_id()
        self.shortest_paths = {}
        self.set_nodes_neigbors_degree()
        self.build_shortest_paths()

    def set_node_id_to_edge_id(self):
        for edge_id, edge in self.edges.items():
            self.node_id_to_edge_id[(edge.source, edge.target)] = edge_id
            self.node_id_to_edge_id[(edge.target, edge.source)] = edge_id

    def set_nodes_neigbors_degree(self):
        for edge_id, edge in self.edges.items():
            self.nodes[edge.source].neigbors.append(edge.target)
            self.nodes[edge.target].neigbors.append(edge.source)
            self.nodes[edge.source].degree += 1
            self.nodes[edge.target].degree += 1

    def get_neighbors(self, node_id: str) -> List[str]:
        return self.nodes[node_id].neigbors
    
    def get_shortest_path_time(self, start_id: str, end_id: str) -> float:
        return self.shortest_paths[(start_id, end_id)]

    def build_shortest_paths(self):
        for edge_id, edge in self.edges.items():
            self.shortest_paths[(edge.source, edge.target)] = edge.time_cost
            self.shortest_paths[(edge.target, edge.source)] = edge.time_cost
        
        # 使用BFS搜索，构建所有节点之间的最短路径
        for node_id in self.nodes:
            self.build_distance(node_id)
    
    def format_shortest_paths(self):
        paths = [[] for _ in range(len(self.nodes))]
        for i in range(len(self.nodes)):
            node_id1 = f"Node_{i+1}"
            for j in range(len(self.nodes)):
                node_id2 = f"Node_{j+1}"
                paths[i].append(self.shortest_paths[(node_id1, node_id2)])
        return paths


    def build_distance(self, start_id: str):
        open_set = set()
        open_set.add(start_id)
        close_set = set()
        distance = {start_id: 0}
        # 层序遍历
        for i in range(len(self.nodes)):
            current_id = open_set.pop()
            close_set.add(current_id)
            for neighbor_id in self.nodes[current_id].neigbors:
                edge_id = self.node_id_to_edge_id[(current_id, neighbor_id)]
                if neighbor_id not in close_set and neighbor_id not in distance:
                    open_set.add(neighbor_id)
                    distance[neighbor_id] = distance[current_id] + self.edges[edge_id].time_cost
                else:
                    distance[neighbor_id] = min(distance[neighbor_id], distance[current_id] + self.edges[edge_id].time_cost)
        for end_id, dist in distance.items():
            self.shortest_paths[(start_id, end_id)] = dist
            self.shortest_paths[(end_id, start_id)] = dist


def test1():
    nodes = {
        "Node_1": Node(id="Node_1", type="room", area=100, floor=1, unsearched_area=100, position=(0, 0), neigbors=[], degree=0, estimated_finish_time=0, difficulty_factor=1.0, allowed_uav_number=3, searching_uav_number=0),
        "Node_2": Node(id="Node_2", type="room", area=100, floor=1, unsearched_area=100, position=(0, 0), neigbors=[], degree=0, estimated_finish_time=0, difficulty_factor=1.0, allowed_uav_number=3, searching_uav_number=0),
        "Node_3": Node(id="Node_3", type="room", area=100, floor=1, unsearched_area=100, position=(0, 0), neigbors=[], degree=0, estimated_finish_time=0, difficulty_factor=1.0, allowed_uav_number=3, searching_uav_number=0)
    }
    edges = {
        "edge1": Edge(id="edge1", source="Node_1", target="Node_2", length=10, weight=1.0, time_cost=10),
        "edge2": Edge(id="edge2", source="Node_2", target="Node_3", length=10, weight=1.0, time_cost=10)
    }
    graph = WorldGraph(nodes, edges)
    print(graph.shortest_paths)

def test2():
    # 1-2:2  1-3:4 2-4:3 2-6:1 6-5:3 3-5:2
    nodes = {
        "Node_1": Node(id="Node_1", type="room", area=100, floor=1, unsearched_area=100, position=(0, 0), neigbors=[], degree=0, estimated_finish_time=0, difficulty_factor=1.0, allowed_uav_number=3, searching_uav_number=0),
        "Node_2": Node(id="Node_2", type="room", area=100, floor=1, unsearched_area=100, position=(0, 0), neigbors=[], degree=0, estimated_finish_time=0, difficulty_factor=1.0, allowed_uav_number=3, searching_uav_number=0),
        "Node_3": Node(id="Node_3", type="room", area=100, floor=1, unsearched_area=100, position=(0, 0), neigbors=[], degree=0, estimated_finish_time=0, difficulty_factor=1.0, allowed_uav_number=3, searching_uav_number=0),
        "Node_4": Node(id="Node_4", type="room", area=100, floor=1, unsearched_area=100, position=(0, 0), neigbors=[], degree=0, estimated_finish_time=0, difficulty_factor=1.0, allowed_uav_number=3, searching_uav_number=0),
        "Node_5": Node(id="Node_5", type="room", area=100, floor=1, unsearched_area=100, position=(0, 0), neigbors=[], degree=0, estimated_finish_time=0, difficulty_factor=1.0, allowed_uav_number=3, searching_uav_number=0),
        "Node_6": Node(id="Node_6", type="room", area=100, floor=1, unsearched_area=100, position=(0, 0), neigbors=[], degree=0, estimated_finish_time=0, difficulty_factor=1.0, allowed_uav_number=3, searching_uav_number=0),
    }
    edges = {
        "Edge_1": Edge(id="Edge_1", source="Node_1", target="Node_2", length=2, weight=1.0, time_cost=2),
        "Edge_2": Edge(id="Edge_2", source="Node_1", target="Node_3", length=4, weight=1.0, time_cost=4),
        "Edge_3": Edge(id="Edge_3", source="Node_2", target="Node_4", length=3, weight=1.0, time_cost=3),
        "Edge_4": Edge(id="Edge_4", source="Node_2", target="Node_6", length=1, weight=1.0, time_cost=1),
        "Edge_5": Edge(id="Edge_5", source="Node_6", target="Node_5", length=3, weight=1.0, time_cost=3),
        "Edge_6": Edge(id="Edge_6", source="Node_3", target="Node_5", length=2, weight=1.0, time_cost=2),
    }
    graph = WorldGraph(nodes, edges)
    print(graph.format_shortest_paths())

if __name__ == "__main__":
    test2() 