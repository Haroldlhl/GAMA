import numpy as np
from typing import List, Tuple
from MapGraph import MapGraph

class MapGenerator:
    """生成具有物理合理性的随机拓扑地图"""
    
    def __init__(self, grid_width: int = 100, grid_height: int = 100):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid = None
        
    def generate_grid_map(self, obstacle_density: float = 0.3) -> np.ndarray:
        """生成一个随机网格地图，包含障碍物和自由空间"""
        # 初始化全自由空间网格
        self.grid = np.zeros((self.grid_height, self.grid_width))
        
        # 添加随机障碍物
        num_obstacles = int(self.grid_width * self.grid_height * obstacle_density)
        for _ in range(num_obstacles):
            x = np.random.randint(0, self.grid_width)
            y = np.random.randint(0, self.grid_height)
            # 创建一些不规则形状的障碍物
            size = np.random.randint(1, 5)
            for dx in range(-size, size + 1):
                for dy in range(-size, size + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        if np.random.random() < 0.7:  # 概率创建障碍物
                            self.grid[ny, nx] = 1  # 1表示障碍物
        
        # 确保边界是障碍物
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        
        return self.grid
    
    def generate_room_positions(self, num_rooms: int) -> List[Tuple[int, int]]:
        """在自由空间中生成房间位置"""
        if self.grid is None:
            self.generate_grid_map()
        
        free_positions = []
        for y in range(1, self.grid_height - 1):
            for x in range(1, self.grid_width - 1):
                if self.grid[y, x] == 0:  # 自由空间
                    free_positions.append((x, y))
        
        if len(free_positions) < num_rooms:
            raise ValueError("Not enough free space for rooms")
        
        return np.random.choice(free_positions, num_rooms, replace=False)
    
    def calculate_shortest_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> float:
        """使用波传播（Dijkstra）计算两点间最短路径距离"""
        if self.grid is None:
            self.generate_grid_map()
        
        # 初始化距离场
        distance_field = np.full_like(self.grid, float('inf'), dtype=float)
        sx, sy = start
        distance_field[sy, sx] = 0
        
        # 优先级队列
        heap = [(0, sx, sy)]
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4连通
        
        while heap:
            dist, x, y = heappop(heap)
            
            if (x, y) == end:
                return dist
            
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.grid_width and 0 <= ny < self.grid_height and 
                    self.grid[ny, nx] == 0):  # 可通行
                    new_dist = dist + np.sqrt(dx*dx + dy*dy)
                    if new_dist < distance_field[ny, nx]:
                        distance_field[ny, nx] = new_dist
                        heappush(heap, (new_dist, nx, ny))
        
        return float('inf')  # 不可达
    
    def generate_topological_map(self, num_rooms: int = 10) -> MapGraph:
        """生成一个完整的拓扑地图"""
        # 1. 生成底层网格地图
        self.generate_grid_map()
        
        # 2. 生成房间位置
        room_positions = self.generate_room_positions(num_rooms)
        
        # 3. 创建地图图结构
        map_graph = MapGraph()
        
        # 4. 添加节点
        for i, pos in enumerate(room_positions):
            area = np.random.uniform(10, 50)  # 随机面积
            risk = np.random.uniform(0, 1)    # 随机风险等级
            node = map_graph.add_node(i, area, risk_level=risk)
            node.grid_positions = [pos]  # 记录在网格上的位置
        
        # 5. 创建连接：基于空间邻近性，但验证可达性
        connection_probability = 0.3  # 连接概率
        
        for i in range(num_rooms):
            for j in range(i + 1, num_rooms):
                pos_i = room_positions[i]
                pos_j = room_positions[j]
                
                # 基于距离的概率连接
                distance = np.sqrt((pos_i[0]-pos_j[0])**2 + (pos_i[1]-pos_j[1])**2)
                max_distance = min(self.grid_width, self.grid_height) / 3
                
                if distance < max_distance and np.random.random() < connection_probability:
                    # 计算实际最短路径距离
                    actual_dist = self.calculate_shortest_path(pos_i, pos_j)
                    if actual_dist < float('inf'):  # 如果可达
                        map_graph.add_edge(i, j, actual_dist)
        
        # 6. 确保图的连通性
        self._ensure_connectivity(map_graph, room_positions)
        
        # 7. 构建距离矩阵
        map_graph.build_distance_matrix()
        
        return map_graph
    
    def _ensure_connectivity(self, map_graph: MapGraph, positions: List[Tuple[int, int]]):
        """确保地图是连通的，通过添加必要的边"""
        # 使用并查集检查连通分量
        parent = {i: i for i in map_graph.nodes}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            parent[find(x)] = find(y)
        
        # 基于现有边构建连通分量
        for (i, j) in map_graph.edges:
            union(i, j)
        
        # 检查是否所有节点都在同一连通分量中
        root = find(0)
        disconnected_nodes = [i for i in map_graph.nodes if find(i) != root]
        
        # 为不连通的节点添加边
        for node_id in disconnected_nodes:
            # 找到最近的已连通节点
            min_dist = float('inf')
            closest_node = None
            pos_i = positions[node_id]
            
            for connected_id in map_graph.nodes:
                if find(connected_id) == root:
                    pos_j = positions[connected_id]
                    dist = self.calculate_shortest_path(pos_i, pos_j)
                    if dist < min_dist:
                        min_dist = dist
                        closest_node = connected_id
            
            if closest_node is not None and min_dist < float('inf'):
                map_graph.add_edge(node_id, closest_node, min_dist)
                union(node_id, closest_node)
    
    def generate_batch(self, num_maps: int, rooms_per_map: int = 8) -> List[MapGraph]:
        """批量生成多个地图用于训练"""
        maps = []
        for i in range(num_maps):
            try:
                map_graph = self.generate_topological_map(rooms_per_map)
                maps.append(map_graph)
                print(f"Generated map {i+1}/{num_maps}")
            except Exception as e:
                print(f"Failed to generate map {i+1}: {e}")
        return maps