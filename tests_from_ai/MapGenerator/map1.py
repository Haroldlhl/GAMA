import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.patches import Rectangle
from scipy.spatial import Voronoi
import os

class MapGenerator:
    def __init__(self, width=100, height=100, num_rooms=10, seed=42):
        self.width = width
        self.height = height
        self.num_rooms = num_rooms
        self.seed = seed
        self.rooms = []
        self.doors = []
        self.corridors = []
        self.map_grid = np.zeros((height, width), dtype=int)
        
        # 设置随机种子以确保可重复性
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_points(self):
        """生成随机点作为房间的中心"""
        points = []
        for _ in range(self.num_rooms):
            x = np.random.randint(self.width * 0.1, self.width * 0.9)
            y = np.random.randint(self.height * 0.1, self.height * 0.9)
            points.append((x, y))
        return points
    
    def points_to_voronoi(self, points):
        """使用Voronoi图将点转换为区域"""
        vor = Voronoi(points)
        return vor
    
    def create_rooms_from_voronoi(self, vor, min_size=5, max_size=15):
        """从Voronoi区域创建矩形房间"""
        rooms = []
        
        for point_idx, region_idx in enumerate(vor.point_region):
            vertices = vor.regions[region_idx]
            if -1 in vertices or len(vertices) == 0:
                continue
            
            # 获取区域的边界
            region_vertices = [vor.vertices[i] for i in vertices if i != -1]
            if not region_vertices:
                continue
                
            x_coords = [v[0] for v in region_vertices]
            y_coords = [v[1] for v in region_vertices]
            
            # 计算区域的中心
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            
            # 确定房间大小
            width = np.random.randint(min_size, max_size)
            height = np.random.randint(min_size, max_size)
            
            # 确保房间在边界内
            x = int(max(0, min(center_x - width/2, self.width - width)))
            y = int(max(0, min(center_y - height/2, self.height - height)))
            
            # 确保宽度和高度为正
            width = max(1, width)
            height = max(1, height)
            
            rooms.append({
                'x': x, 
                'y': y, 
                'width': width, 
                'height': height,
                'center': (int(center_x), int(center_y))
            })
            
            # 在网格地图上标记房间区域
            self.map_grid[y:y+height, x:x+width] = 1
        
        return rooms
    
    def find_adjacent_rooms(self):
        """找到相邻的房间对"""
        adjacent_pairs = []
        for i, room1 in enumerate(self.rooms):
            for j, room2 in enumerate(self.rooms[i+1:], i+1):
                # 检查两个房间是否相邻
                x1, y1, w1, h1 = room1['x'], room1['y'], room1['width'], room1['height']
                x2, y2, w2, h2 = room2['x'], room2['y'], room2['width'], room2['height']
                
                # 检查水平相邻
                if (y1 <= y2 + h2 and y1 + h1 >= y2 and 
                    (abs(x1 + w1 - x2) <= 3 or abs(x2 + w2 - x1) <= 3)):
                    adjacent_pairs.append((i, j, 'horizontal'))
                
                # 检查垂直相邻
                elif (x1 <= x2 + w2 and x1 + w1 >= x2 and 
                      (abs(y1 + h1 - y2) <= 3 or abs(y2 + h2 - y1) <= 3)):
                    adjacent_pairs.append((i, j, 'vertical'))
        
        return adjacent_pairs
    
    def create_doors(self, adjacent_pairs, door_width=1):
        """在相邻房间之间创建门"""
        doors = []
        
        for i, j, orientation in adjacent_pairs:
            room1 = self.rooms[i]
            room2 = self.rooms[j]
            
            if orientation == 'horizontal':
                # 确定两个房间的垂直重叠区域
                y_overlap_start = max(room1['y'], room2['y'])
                y_overlap_end = min(room1['y'] + room1['height'], room2['y'] + room2['height'])
                
                if y_overlap_end > y_overlap_start + 2:  # 确保有足够的空间放置门
                    # 在重叠区域内随机选择门的位置
                    door_y = np.random.randint(y_overlap_start + 1, y_overlap_end - 1)
                    
                    # 确定门的x位置
                    if room1['x'] + room1['width'] < room2['x']:
                        door_x = room1['x'] + room1['width']
                    else:
                        door_x = room2['x'] + room2['width']
                    
                    # 确保门在边界内
                    if door_x < 0 or door_x >= self.width or door_y < 0 or door_y >= self.height:
                        continue
                    
                    # 创建门
                    doors.append({
                        'x': door_x,
                        'y': door_y,
                        'width': 1,
                        'height': door_width,
                        'connects': (i, j)
                    })
                    
                    # 在地图上标记门
                    for dy in range(door_width):
                        if 0 <= door_y + dy < self.height:
                            self.map_grid[door_y + dy, door_x] = 2
            
            else:  # vertical
                # 确定两个房间的水平重叠区域
                x_overlap_start = max(room1['x'], room2['x'])
                x_overlap_end = min(room1['x'] + room1['width'], room2['x'] + room2['width'])
                
                if x_overlap_end > x_overlap_start + 2:  # 确保有足够的空间放置门
                    # 在重叠区域内随机选择门的位置
                    door_x = np.random.randint(x_overlap_start + 1, x_overlap_end - 1)
                    
                    # 确定门的y位置
                    if room1['y'] + room1['height'] < room2['y']:
                        door_y = room1['y'] + room1['height']
                    else:
                        door_y = room2['y'] + room2['height']
                    
                    # 确保门在边界内
                    if door_x < 0 or door_x >= self.width or door_y < 0 or door_y >= self.height:
                        continue
                    
                    # 创建门
                    doors.append({
                        'x': door_x,
                        'y': door_y,
                        'width': door_width,
                        'height': 1,
                        'connects': (i, j)
                    })
                    
                    # 在地图上标记门
                    for dx in range(door_width):
                        if 0 <= door_x + dx < self.width:
                            self.map_grid[door_y, door_x + dx] = 2
        
        return doors
    
    def create_corridors(self):
        """创建连接房间的过道 - 简化版本"""
        corridors = []
        # 这里可以添加更复杂的过道生成逻辑
        return corridors
    
    def generate_map(self):
        """生成完整的地图"""
        # 生成随机点
        points = self.generate_points()
        
        # 创建Voronoi图
        vor = self.points_to_voronoi(points)
        
        # 从Voronoi区域创建房间
        self.rooms = self.create_rooms_from_voronoi(vor)
        
        # 找到相邻的房间
        adjacent_pairs = self.find_adjacent_rooms()
        
        # 创建门
        self.doors = self.create_doors(adjacent_pairs)
        
        # 创建过道
        self.corridors = self.create_corridors()
        
        return self.map_grid, self.rooms, self.doors, self.corridors
    
    def save_map_visualization(self, output_dir="map_output"):
        """保存地图可视化到文件"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # 绘制房间
        for room in self.rooms:
            rect = Rectangle((room['x'], room['y']), room['width'], room['height'],
                             linewidth=1, edgecolor='blue', facecolor='lightblue', alpha=0.7)
            ax.add_patch(rect)
            
            # 标记房间中心
            ax.plot(room['center'][0], room['center'][1], 'ro', markersize=4)
        
        # 绘制门
        for door in self.doors:
            rect = Rectangle((door['x'], door['y']), door['width'], door['height'],
                             linewidth=1, edgecolor='red', facecolor='red', alpha=0.7)
            ax.add_patch(rect)
        
        # 设置图形属性
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.set_title(f'Generated Map with {len(self.rooms)} Rooms and {len(self.doors)} Doors')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        
        # 保存图形
        plt.savefig(os.path.join(output_dir, 'map_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存网格地图
        plt.figure(figsize=(10, 10))
        plt.imshow(self.map_grid, cmap='tab20c', origin='lower')
        plt.title('Grid Representation of the Map')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.colorbar(label='Cell Type (0=Empty, 1=Room, 2=Door)')
        plt.savefig(os.path.join(output_dir, 'map_grid.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"地图可视化已保存到 {output_dir} 目录")
    
    def save_map_data(self, output_dir="map_output"):
        """保存地图数据到文件"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存网格地图
        np.save(os.path.join(output_dir, 'map_grid.npy'), self.map_grid)
        
        # 保存房间信息
        with open(os.path.join(output_dir, 'rooms.txt'), 'w') as f:
            for i, room in enumerate(self.rooms):
                f.write(f"Room {i}: x={room['x']}, y={room['y']}, width={room['width']}, height={room['height']}\n")
        
        # 保存门信息
        with open(os.path.join(output_dir, 'doors.txt'), 'w') as f:
            for i, door in enumerate(self.doors):
                f.write(f"Door {i}: x={door['x']}, y={door['y']}, connects rooms {door['connects']}\n")
        
        print(f"地图数据已保存到 {output_dir} 目录")

# 使用示例
if __name__ == "__main__":
    # 创建地图生成器实例
    map_gen = MapGenerator(width=50, height=50, num_rooms=8, seed=42)
    
    # 生成地图
    map_grid, rooms, doors, corridors = map_gen.generate_map()
    
    # 保存可视化结果和数据
    map_gen.save_map_visualization()
    map_gen.save_map_data()
    
    # 打印生成信息
    print(f"生成了 {len(rooms)} 个房间和 {len(doors)} 个门")
    print("地图网格形状:", map_grid.shape)
    print("地图网格内容示例:")
    for i in range(50):
        print(map_grid[i, :])  # 显示前10x10的网格内容