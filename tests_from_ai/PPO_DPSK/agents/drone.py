# agents/drone.py
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
import torch
from config import params

class DroneStatus(Enum):
    IDLE = auto()
    MOVING = auto()
    SEARCHING = auto()
    PENDING = auto()

@dataclass
class DroneState:
    id: int
    status: DroneStatus
    current_node: int
    target_node: Optional[int] = None
    velocity: float = 1.0
    task_end_time: float = 0.0 # 记录的是绝对时间
    
    def encode(self, current_time) -> torch.Tensor:
        """编码无人机状态为特征向量"""
        status_onehot = torch.zeros(len(DroneStatus))
        status_onehot[self.status.value - 1] = 1.0
        
        features = torch.cat([
            status_onehot,
            torch.tensor([self.task_end_time - current_time / params.env_config['max_task_time']]),  # 归一化
            torch.tensor([self.target_node]),
        ])

        return features
    
    def update(self, **kwargs):
        """更新无人机状态"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)