# environment/event_queue.py
import heapq
from typing import Any, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Event:
    time: float
    drone_id: str
    data: Any = None

    def __lt__(self, other):
        return self.time < other.time


class EventQueue:
    def __init__(self):
        self.queue: List[Event] = []
        self.current_time = 0.0
    
    def add_event(self, time: float, drone_id: str, data: Any = None):
        # 先移除该无人机的所有现有事件
        self.remove_events_for_drone(drone_id)
        
        # 添加新事件
        event = Event(time, drone_id, data)
        heapq.heappush(self.queue, event)
    
    def get_next_event(self) -> Optional[Event]:
        """返回时间最小的事件"""
        if not self.queue:
            return None
        return self.queue[0]
    
    def pop_event(self) -> Optional[Event]:
        """弹出时间最小的事件"""
        if not self.queue:
            return None
        event = heapq.heappop(self.queue)
        self.current_time = event.time
        return event
    
    def remove_events_for_drone(self, drone_id: str):
        """移除特定无人机的所有事件"""
        self.queue = [e for e in self.queue if e.drone_id != drone_id]
        heapq.heapify(self.queue)
    
    def clear(self):
        """清除所有事件"""
        self.queue.clear()
        self.current_time = 0.0
    
    def __len__(self):
        return len(self.queue)