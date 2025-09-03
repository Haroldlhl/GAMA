#### 8. `Config` (配置类)
# **职责** 以 argparse的形式管理所有超参数和设置。
# **主要属性**：
# - `training_params`: `{lr, gamma, batch_size, ...}`
# - `reward_params`: `{area_reward, time_penalty, ...}`
# - `env_params`: `{map_file, num_drones, ...}`
# - `model_params`: `{gnn_hidden_dim, actor_hidden_dims, ...}`

import argparse

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--lr', type=float, default=0.001)
        self.parser.add_argument('--gamma', type=float, default=0.99)
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.args = self.parser.parse_args()     