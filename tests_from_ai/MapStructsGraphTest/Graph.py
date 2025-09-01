from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

# 使用枚举类型确保数据一致性
class RoomType(Enum):
    NORMAL = auto()         # 普通房间
    HALLWAY = auto()        # 通道
    CUT_HALLWAY = auto()    # 被切割的通道
    STAIRS = auto()         # 楼梯

class ConnectionType(Enum):
    DOOR = auto()           # 标准门
    WIDE_DOOR = auto()      # 双开门/宽门
    NARROW_PASSAGE = auto() # 狭窄通道
    STAIRS = auto()         # 楼梯连接
    OPEN_SPACE = auto()     # 开阔空间连接

@dataclass
class SpaceNode:
    """空间节点（房间）类"""
    id: str  # 节点唯一标识符
    area: float  # 面积（平方米）
    room_type: RoomType  # 房间类型
    discount_coefficient: float  # 折扣面积系数
    floor: int  # 所在楼层
    connection_count: int = 0  # 联通的节点数量
    search_time: float = area * discount_coefficient  # 搜索时间
    

@dataclass
class SpaceEdge:
    """空间边（连接）类"""
    source: str  # 起始节点ID
    target: str  # 目标节点ID
    length: float  # 长度（米）
    connection_type: ConnectionType  # 连接类型
    discount: float  # 通行折扣系数
    
    def get_discounted_time(self, flight_speed: float = 1.0) -> float:
        """计算折扣后的通行时间"""
        return (self.length * self.discount) / flight_speed

class BuildingGraph:
    """建筑图类，用于管理节点和边"""
    def __init__(self):
        self.nodes: Dict[str, SpaceNode] = {}
        self.edges: Dict[str, List[SpaceEdge]] = {}
        self.reverse_edges: Dict[str, List[SpaceEdge]] = {}
    
    def add_node(self, node: SpaceNode) -> None:
        """添加节点"""
        self.nodes[node.id] = node
        self.edges[node.id] = []
        self.reverse_edges[node.id] = []
    
    def add_edge(self, edge: SpaceEdge) -> None:
        """添加边（无向图，添加双向关系）"""
        # 添加到正向边列表
        if edge.source in self.edges:
            self.edges[edge.source].append(edge)
        
        # 添加到反向边列表
        if edge.target in self.reverse_edges:
            reverse_edge = SpaceEdge(
                source=edge.target,
                target=edge.source,
                length=edge.length,
                connection_type=edge.connection_type,
                discount=edge.discount
            )
            self.reverse_edges[edge.target].append(reverse_edge)
    
    def update_connection_counts(self) -> None:
        """更新所有节点的连接数量"""
        for node_id in self.nodes:
            # 对于无向图，连接数 = 出边数（因为我们已经建立了双向关系）
            self.nodes[node_id].connection_count = len(self.edges[node_id])
    
    def get_connected_nodes(self, node_id: str) -> Set[str]:
        """获取与指定节点相连的所有节点ID"""
        connected = set()
        for edge in self.edges.get(node_id, []):
            connected.add(edge.target)
        for edge in self.reverse_edges.get(node_id, []):
            connected.add(edge.source)
        return connected
    
    def find_edge_between(self, node1_id: str, node2_id: str) -> Optional[SpaceEdge]:
        """查找两个节点之间的边"""
        # 检查正向边
        for edge in self.edges.get(node1_id, []):
            if edge.target == node2_id:
                return edge
        # 检查反向边
        for edge in self.edges.get(node2_id, []):
            if edge.target == node1_id:
                return edge
        return None

def calculate_total_cost(graph: BuildingGraph, path: List[str], flight_speed: float = 1.0) -> float:
    """计算路径的总成本（搜索时间 + 飞行时间）"""
    total_cost = 0.0
    
    for i, node_id in enumerate(path):
        # 添加节点搜索时间
        node = graph.nodes[node_id]
        total_cost += node.search_time
        
        # 添加边通行时间（如果不是最后一个节点）
        if i < len(path) - 1:
            next_node_id = path[i + 1]
            edge = graph.find_edge_between(node_id, next_node_id)
            if edge:
                total_cost += edge.get_discounted_time(flight_speed)
    
    return total_cost

def test1():
    """测试函数"""
    # 创建建筑图实例
    building = BuildingGraph()

    # 创建节点
    room_101 = SpaceNode(
        id="room_101",
        area=25.0,
        room_type=RoomType.NORMAL,
        discount_coefficient=0.8,
        floor=1
    )

    hallway_1f = SpaceNode(
        id="hallway_1f",
        area=50.0,
        room_type=RoomType.HALLWAY,
        discount_coefficient=0.3,
        floor=1
    )

    stairs_1_2 = SpaceNode(
        id="stairs_1_2",
        area=15.0,
        room_type=RoomType.STAIRS,
        discount_coefficient=0.6,
        floor=1
    )

    # 添加节点
    building.add_node(room_101)
    building.add_node(hallway_1f)
    building.add_node(stairs_1_2)

    # 创建边
    door_1 = SpaceEdge(
        source="room_101",
        target="hallway_1f",
        length=2.0,
        connection_type=ConnectionType.DOOR,
        discount=1.2
    )

    stairs_connection = SpaceEdge(
        source="hallway_1f",
        target="stairs_1_2",
        length=1.5,
        connection_type=ConnectionType.STAIRS,
        discount=2.0
    )

    # 添加边
    building.add_edge(door_1)
    building.add_edge(stairs_connection)

    # 更新连接数量
    building.update_connection_counts()

    # 计算路径成本
    path = ["room_101", "hallway_1f", "stairs_1_2"]
    total_time = calculate_total_cost(building, path, flight_speed=0.5)
    print(f"路径总时间: {total_time:.2f} 单位时间")

    # 显示节点信息
    print(f"\n房间101信息:")
    print(f"  面积: {room_101.area}m²")
    print(f"  类型: {room_101.room_type.name}")
    print(f"  搜索时间: {room_101.search_time:.2f}")
    print(f"  连接数量: {room_101.connection_count}")

    # 显示边信息
    print(f"\n门连接信息:")
    print(f"  长度: {door_1.length}m")
    print(f"  类型: {door_1.connection_type.name}")
    print(f"  通行时间: {door_1.get_discounted_time(flight_speed=0.5):.2f}")

if __name__ == "__main__":
    test1()