from sortedcontainers import SortedDict
import uuid

class EventQueue:
    def __init__(self):
        # 键为时间（自动排序），值为事件列表（存储不可哈希的事件）
        self.time_to_events = SortedDict()
        self.event_to_time = {}  # 反向映射：{事件标识: 时间}
        self.event_identifiers = []  # 存储 (事件, 标识) 元组
    
    def _get_event_id(self, event):
        """为事件生成唯一可哈希的标识"""
        if isinstance(event, dict):
            # 遍历列表查找已有标识
            for existing_event, event_id in self.event_identifiers:
                if existing_event == event:  # 比较字典内容
                    return event_id
            # 生成新标识并存储
            new_id = uuid.uuid4().hex
            self.event_identifiers.append((event, new_id))
            return new_id
        elif isinstance(event, (list, set)):
            # 转换为元组作为标识
            return tuple(event)
        else:
            # 可哈希类型直接返回自身
            return event
    
    def add_event(self, time, event):
        """添加或更新事件，若时间冲突则自动将新事件时间+0.001"""
        # 检查时间是否已存在，若存在则调整时间
        adjusted_time = time
        while adjusted_time in self.time_to_events:
            adjusted_time += 0.001  # 每次冲突增加0.001
        
        event_id = self._get_event_id(event)
        
        # 如果事件已存在，先移除旧的
        if event_id in self.event_to_time:
            old_time = self.event_to_time[event_id]
            events = self.time_to_events[old_time]
            # 从列表中移除事件（处理不可哈希类型）
            if event in events:
                events.remove(event)
            if not events:  # 如果该时间点没有事件了，移除这个时间
                del self.time_to_events[old_time]
        
        # 添加新的映射（使用调整后的时间）
        self.event_to_time[event_id] = adjusted_time
        if adjusted_time not in self.time_to_events:
            self.time_to_events[adjusted_time] = []  # 改用列表存储事件
        self.time_to_events[adjusted_time].append(event)  # 列表支持添加不可哈希类型
    
    def remove_event(self, event):
        """根据事件删除"""
        event_id = self._get_event_id(event)
        if event_id in self.event_to_time:
            time = self.event_to_time[event_id]
            del self.event_to_time[event_id]
            
            # 从事件标识列表中移除
            for i, (e, e_id) in enumerate(self.event_identifiers):
                if e == event and e_id == event_id:
                    del self.event_identifiers[i]
                    break
            
            if time in self.time_to_events:
                events = self.time_to_events[time]
                if event in events:
                    events.remove(event)
                if not events:
                    del self.time_to_events[time]
    
    def get_next_event(self):
        """获取最早发生的事件"""
        if self.time_to_events:
            earliest_time = next(iter(self.time_to_events.keys()))
            events = self.time_to_events[earliest_time].copy()  # 复制列表
            del self.time_to_events[earliest_time]
            
            # 清理相关映射
            for event in events:
                event_id = self._get_event_id(event)
                if event_id in self.event_to_time:
                    del self.event_to_time[event_id]
                
                # 从事件标识列表中移除
                for i, (e, e_id) in enumerate(self.event_identifiers):
                    if e == event and e_id == event_id:
                        del self.event_identifiers[i]
                        break
            
            return earliest_time, events
        return None, None
    
    def clear(self):
        self.time_to_events = SortedDict()
        self.event_to_time = {}
        self.event_identifiers = []
