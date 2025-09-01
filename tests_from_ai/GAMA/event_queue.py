from sortedcontainers import SortedDict

class EventQueue:
    def __init__(self):
        self.time_to_events = SortedDict()  # 键为时间（自动排序），值为事件集合
        self.event_to_time = {}  # 反向映射：{事件: 时间}，用于快速查询
    
    def add_event(self, time, event):
        """添加或更新事件"""
        # 如果事件已存在，先移除旧的
        if event in self.event_to_time:
            old_time = self.event_to_time[event]
            events = self.time_to_events[old_time]
            events.remove(event)
            if not events:  # 如果该时间点没有事件了，移除这个时间
                del self.time_to_events[old_time]
        
        # 添加新的映射
        self.event_to_time[event] = time
        if time not in self.time_to_events:
            self.time_to_events[time] = set()
        self.time_to_events[time].add(event)
    
    def remove_event(self, event):
        """根据事件删除"""
        if event in self.event_to_time:
            time = self.event_to_time[event]
            del self.event_to_time[event]
            events = self.time_to_events[time]
            events.remove(event)
            if not events:
                del self.time_to_events[time]
    
    def get_next_event(self):
        """获取最早发生的事件"""
        if self.time_to_events:
            earliest_time = self.time_to_events.keys()[0]
            return earliest_time, self.time_to_events[earliest_time]
        return None, None