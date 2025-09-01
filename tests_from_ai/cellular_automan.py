import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.use('TkAgg')

class CellularAutomatonMap:
    def __init__(self, width=50, height=50, initial_density=0.45, 
                 birth_limit=4, death_limit=3, iterations=5):
        """
        初始化元胞自动机地图生成器
        :param width: 地图宽度
        :param height: 地图高度
        :param initial_density: 初始障碍物密度 (0-1)
        :param birth_limit: 出生阈值，周围活细胞数达到此值时死细胞复活
        :param death_limit: 死亡阈值，周围活细胞数低于此值时活细胞死亡
        :param iterations: 迭代次数
        """
        self.width = width
        self.height = height
        self.initial_density = initial_density
        self.birth_limit = birth_limit
        self.death_limit = death_limit
        self.iterations = iterations
        
        # 初始化地图
        self.map = self._initialize_map()
        self.history = [self.map.copy()]  # 保存每一步的历史用于动画
        
    def _initialize_map(self):
        """初始化地图，随机生成障碍物"""
        # 创建随机地图，边界设为全是障碍物
        map_grid = np.random.rand(self.height, self.width) < self.initial_density
        
        # 确保边界都是障碍物
        map_grid[0, :] = 1
        map_grid[-1, :] = 1
        map_grid[:, 0] = 1
        map_grid[:, -1] = 1
        
        return map_grid.astype(int)
    
    def _count_live_neighbors(self, x, y):
        """计算某个细胞周围的活细胞数量"""
        # 确保不越界
        x_min, x_max = max(0, x-1), min(self.width-1, x+1)
        y_min, y_max = max(0, y-1), min(self.height-1, y+1)
        
        # 计算周围活细胞总数，减去自身
        neighbors = self.map[y_min:y_max+1, x_min:x_max+1].sum() - self.map[y, x]
        return neighbors
    
    def step(self):
        """执行一次元胞自动机迭代"""
        new_map = self.map.copy()
        
        # 遍历每个细胞(跳过边界)
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                neighbors = self._count_live_neighbors(x, y)
                
                # 应用规则
                if self.map[y, x] == 1:  # 当前是活细胞(障碍物)
                    if neighbors < self.death_limit:
                        new_map[y, x] = 0  # 死亡
                else:  # 当前是死细胞(通路)
                    if neighbors > self.birth_limit:
                        new_map[y, x] = 1  # 复活
        
        self.map = new_map
        self.history.append(new_map.copy())
        return new_map
    
    def generate(self):
        """执行完整的地图生成过程"""
        for _ in range(self.iterations):
            self.step()
        return self.map
    
    def visualize(self, show_history=False):
        """可视化地图"""
        # 创建自定义颜色映射：0=通路(深色)，1=障碍物(浅色)
        colors = [(0.1, 0.1, 0.3), (0.8, 0.8, 0.8)]  # 深蓝和浅灰
        cmap = LinearSegmentedColormap.from_list('map_cmap', colors, N=2)
        
        if show_history and len(self.history) > 1:
            # 显示生成过程动画
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(self.history[0], cmap=cmap, interpolation='nearest')
            ax.set_title(f'元胞自动机地图生成 - 迭代 0/{len(self.history)-1}')
            ax.axis('off')
            
            def update(frame):
                im.set_data(self.history[frame])
                ax.set_title(f'元胞自动机地图生成 - 迭代 {frame}/{len(self.history)-1}')
                return [im]
            
            ani = animation.FuncAnimation(
                fig, update, frames=len(self.history),
                interval=300, blit=True
            )
            plt.show()
        else:
            # 只显示最终结果
            plt.figure(figsize=(8, 8))
            plt.imshow(self.map, cmap=cmap, interpolation='nearest')
            plt.title('元胞自动机生成的地图')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    # 创建地图生成器实例
    map_generator = CellularAutomatonMap(
        width=80, 
        height=60, 
        initial_density=0.48,  # 初始障碍物比例
        birth_limit=4,         # 复活阈值
        death_limit=3,         # 死亡阈值
        iterations=15          # 迭代次数
    )
    
    # 生成地图
    print("正在生成地图...")
    final_map = map_generator.generate()
    
    # 可视化结果，设置show_history=True可以看到生成过程
    print("显示地图生成过程...")
    map_generator.visualize(show_history=True)
