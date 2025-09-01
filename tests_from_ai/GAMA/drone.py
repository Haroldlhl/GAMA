# **职责**：代表一个无人机智能体，维护其自身状态。
# **主要属性**：
# - `id`, `current_node_id`, `status` (状态机), `battery_level`, `velocity`, `target_node_id`。
# - `current_action`: 当前正在执行的动作（如“前往节点5”）。
# - `estimated_completion_time`: 当前动作的预计完成时间（用于事件队列）。

# **主要方法**：
# - `get_observation(graph)`: 根据当前所在节点和自身状态，计算局部观察（用于Actor网络输入）。
# - `encode_state()`: 根据自身属性，计算其状态编码向量。
# - `set_target(target_node_id, graph)`: 设置目标，计算路径和预计到达时间，并向环境注册到达事件。

from dataclasses import dataclass
from enum import Enum
from event_queue import EventQueue
from congig_parse import Config

UAV_Status = Enum('UAV_Status', ['idle', 'searching', 'moving', 'pending'])
# 只会给无人机下发节点任务，自行判断会转化为什么状态
# pending 是一个瞬时状态(不编码），转化为空闲后， 会有一个空闲倒计时，空闲状态结束后会继续进入pending
event_queue = EventQueue()

def encode_uav_status(uav_status: UAV_Status):
    uav_status_onehot = [0.0] * len(UAV_Status)
    uav_status_onehot[uav_status.value - 1] = 1
    return uav_status_onehot

@dataclass
class Drone:
    id: str
    status: UAV_Status
    current_node_id: str
    target_node_id:str
    config: Config
    velocity: float=0.0
    search_speed: float=10.0
    task_end_time: float=0.0

    def get_observation(self):
        return self.target_node_id, self.status, self.task_end_time

    def update_status(self, graph, target_node_id, task_queue, global_time):
        if global_time < self.task_end_time:
            return
        if global_time > self.task_end_time:
            raise ValueError("Task end time is in the past")

        # 结束搜索任务， 需要节点中删除无人机
        if self.status == UAV_Status.searching:
            current_searching_node = graph.nodes[self.current_node_id]
            current_searching_node.updata_unsearched_area(global_time)
            current_searching_node.searching_uav.remove(self)

        self.current_node_id = self.target_node_id
        self.target_node_id = target_node_id
        node = graph.nodes[target_node_id]
        distance_matrix = graph.shortest_paths
        # 已到达目标节点
        if target_node_id == self.current_node_id:
            if not node.is_available():# 变为idle
                self.status = UAV_Status.idle
                self.task_end_time = global_time + self.config.uav.idle_time
                event_queue.add_event(self.task_end_time, self)
                return 
            else:
                # 搜索， 区分是否有别的无人机
                if len(node.searching_uav) == 0:
                    # 搜索，先变自己，再变node, 不需要删除有限队列中节点
                    self.status = UAV_Status.searching
                    node.searching_uav.append(self)
                    node.update_unsearched_area(global_time)
                    self.task_end_time = global_time + node.unsearched_area / node.search_speed
                    event_queue.add_event(self.task_end_time, self)
                    return 
                else:# 有其他无人机， 要改变优先队列，以及他们的时间
                    node.updata_unsearched_area(global_time)
                    old_uavs = [uav for uav in node.searching_uav if uav != self]
                    node.searching_uav.append(self)
                    node.update_search_speed()
                    self.task_end_time = global_time + node.unsearched_area / node.search_speed
                    
                    # 删除并维护新的队列 
                    for uav in old_uavs:
                        uav.task_end_time = self.task_end_time
                        event_queue.add_event(uav.task_end_time, uav)
                    return 
        
        else: #未到达， 移动
            self.status = UAV_Status.moving
            self.task_end_time = global_time + distance_matrix[self.current_node_id][target_node_id]
            event_queue.add_event(self.task_end_time, self)
            return 

