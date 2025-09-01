import Graph

# 创建建筑图实例
building = BuildingGraph()

# 创建一些节点
room_101 = SpaceNode(
    id="room_101",
    area=25.0,
    room_type=RoomType.NORMAL,
    discount_coefficient=0.8,  # 普通房间搜索系数
    floor=1
)

hallway_1f = SpaceNode(
    id="hallway_1f",
    area=50.0,
    room_type=RoomType.HALLWAY,
    discount_coefficient=0.3,  # 通道搜索系数较低
    floor=1
)

stairs_1_2 = SpaceNode(
    id="stairs_1_2",
    area=15.0,
    room_type=RoomType.STAIRS,
    discount_coefficient=0.6,  # 楼梯搜索系数
    floor=1  # 楼梯通常跨越多个楼层，这里以主要楼层为准
)

# 添加节点到图中
building.add_node(room_101)
building.add_node(hallway_1f)
building.add_node(stairs_1_2)

# 创建边（连接）
door_1 = SpaceEdge(
    source="room_101",
    target="hallway_1f",
    length=2.0,
    connection_type=ConnectionType.DOOR,
    discount=1.2  # 通过门需要额外时间
)

stairs_connection = SpaceEdge(
    source="hallway_1f",
    target="stairs_1_2",
    length=1.5,
    connection_type=ConnectionType.STAIRS,
    discount=2.0  # 楼梯通行时间加倍
)

# 添加边到图中
building.add_edge(door_1)
building.add_edge(stairs_connection)

# 更新所有节点的连接数量
building.update_connection_counts()

# 示例：计算某个路径的总成本
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
            edges = graph.find_edges_between(node_id, next_node_id)
            if edges:
                # 取第一条边的通行时间（假设节点间只有一条边）
                total_cost += edges[0].discounted_time(flight_speed)
    
    return total_cost

# 使用示例
path = ["room_101", "hallway_1f", "stairs_1_2"]
total_time = calculate_total_cost(building, path, flight_speed=0.5)
print(f"路径总时间: {total_time:.2f} 单位时间")

# 查看节点信息
print(f"\n房间101信息:")
print(f"  面积: {room_101.area}m²")
print(f"  类型: {room_101.room_type.name}")
print(f"  搜索时间: {room_101.search_time:.2f}")
print(f"  连接数量: {room_101.connection_count}")

# 查看边信息
print(f"\n门连接信息:")
print(f"  长度: {door_1.length}m")
print(f"  类型: {door_1.connection_type.name}")
print(f"  通行时间: {door_1.discounted_time(flight_speed=0.5):.2f}")